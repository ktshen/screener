import datetime
import seaborn as sns
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import gc
import warnings
import optuna # type: ignore
import time
import math
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
from vectorbtpro.portfolio.enums import SizeType
import vectorbtpro as vbt


pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

class BackTester():
    """
        信號建構 & 視覺化信號
    """

    def __init__(self, **kwargs):
        self.df = pd.DataFrame()
        self.configs = {}
        self.freq = '5min'
        
        self.fee = 0.000   # % 
        self.fixed_fee = 0 # $
        self.slippage = 0  # %
        self.lag = 1
        self.__dict__.update(kwargs)

    # def _strategy(self,df,side,**params):
    #     '''
    #     自訂策略
    #     '''
    #     pass
    
    def resample_df(self,df,freq = '1h'):
        cols = ['open', 'high', 'low', 'close','volume']
        agg =  ['first','max',  'min', 'last', 'sum']

        df = df[cols]
        df = df.resample(freq).agg(dict(zip(cols,agg)))
        return df.dropna()

    def strategy(self,side,params={}):
        _pf, _ = self._strategy(self.df,side,**params) # type: ignore
        return _pf
    
    def _strategy_setting(self):
        # 策略設定
        vbt.settings.portfolio['init_cash'] = 10000
        vbt.settings.portfolio['size'] = 10000
        vbt.settings.portfolio['size_type'] = SizeType.Value
        vbt.settings.wrapping['freq'] = self.freq
        
        # 手續費、滑價
        vbt.settings.portfolio['fees'] = self.fee
        vbt.settings.portfolio['fixed_fees'] = self.fixed_fee
        vbt.settings.portfolio['slippage'] = self.slippage
    
    def for_loop_backtest(self, df, params = {}, print_info = False):
                
        self.for_loop_cache = {}
        self.for_loop_cache['print_info'] = print_info
        self.for_loop_cache['time_index'] = df.index
        self.for_loop_cache['direction'] = None
        self.for_loop_cache['entry_price'] = None
        self.for_loop_cache['entry_kbar'] = None
        self.for_loop_cache['exit_kbar'] = None

        self.for_loop_cache['tradetimes'] = 0
        self.for_loop_cache['trade_record'] = {}
        self.for_loop_cache['holdings'] = np.full((len(df), ), 0.)

        self._for_loop_backtest(df,params) # type: ignore

        if self.for_loop_cache['direction'] != None: # 補齊最後一單用 close 出掉
            self.for_loop_cache['trade_record'][self.for_loop_cache['tradetimes']] = {
                "direction": self.for_loop_cache['direction'],
                "entry_time": self.for_loop_cache['entry_time'],
                "entry_price": self.for_loop_cache['entry_price'],
                'entry_kbar':self.for_loop_cache['entry_kbar'],
                "exit_time": df.index[-1],
                "exit_price": df['close'][-1],
                'exit_kbar': len(df)
                }
            
        self.for_loop_cache['trade_record'] = pd.DataFrame(self.for_loop_cache['trade_record']).T
        self.for_loop_trade_record = self.for_loop_cache['trade_record']
        if len(self.for_loop_trade_record) == 0:
            return False, False, False, False
        
        long_entry_time = self.for_loop_trade_record[self.for_loop_trade_record['direction'] == 'long']['entry_time'].astype(str).to_list()
        long_exit_time = self.for_loop_trade_record[self.for_loop_trade_record['direction'] == 'long']['exit_time'].astype(str).to_list()
        short_entry_time = self.for_loop_trade_record[self.for_loop_trade_record['direction'] == 'short']['entry_time'].astype(str).to_list()
        short_exit_time = self.for_loop_trade_record[self.for_loop_trade_record['direction'] == 'short']['exit_time'].astype(str).to_list()
        
        long_entry = pd.Series(df.index,index=df.index).isin(long_entry_time)
        long_exit = pd.Series(df.index,index=df.index).isin(long_exit_time)
        short_entry = pd.Series(df.index,index=df.index).isin(short_entry_time)
        short_exit = pd.Series(df.index,index=df.index).isin(short_exit_time)
        self.for_loop_cache = {}
        return long_entry, long_exit, short_entry, short_exit

    def upon_forloop_signal(self, direction, position_type, time, price, idx):
        if self.for_loop_cache['print_info']:
            print_direction = "多" if direction == 'long' else "空"
            print_position_type = "開" if position_type == 'open' else "平"
            print(time, "訊號{}{}，下根K進場價格：{}".format(print_position_type,print_direction,price))

        if position_type == 'open':
            
            if direction == 'long':
                self.for_loop_cache['holdings'][idx] = 1
            else:
                self.for_loop_cache['holdings'][idx] = -1
            
            self.for_loop_cache['direction'] = direction
            self.for_loop_cache['entry_time'] = time
            self.for_loop_cache['entry_price'] = price
            self.for_loop_cache['entry_kbar'] = idx
        else :
            self.for_loop_cache['holdings'][idx] = 0
            self.for_loop_cache['trade_record'][self.for_loop_cache['tradetimes']] = {
                "direction": self.for_loop_cache['direction'],
                "entry_time": self.for_loop_cache['entry_time'],
                "entry_price": self.for_loop_cache['entry_price'],
                'entry_kbar':self.for_loop_cache['entry_kbar'],
                "exit_time": time,
                "exit_price": price,
                "exit_kbar": idx
            }
            self.for_loop_cache['direction'] = None
            self.for_loop_cache['entry_time'] = None
            self.for_loop_cache['entry_price'] = None
            self.for_loop_cache['entry_kbar'] = None
            self.for_loop_cache['tradetimes'] += 1 

    def upon_kbar_loop(self, i):
        self.for_loop_cache['holdings'][i] = self.for_loop_cache['holdings'][i - 1]
        return self.for_loop_cache['holdings'][i]
    
    def optimize(
            self,
            side='both',
            params={},
            target='Calmar Ratio',
            direction='max',
            opt_type='joblib',
            start = '',
            end = ''
            ):
        
        # train set  
        train = self.df      
        if len(start) != 0:
            train = train.loc[start:]
        
        if len(end) != 0:
            train = train.loc[:end]

        _parmas = {}
        # generate all params conbination
        for key in params.keys():
            p = params[key]
            _parmas[key] = list(np.arange(p[0], p[1], p[2]))

        keys, values = zip(*_parmas.items())
        params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        result_list =  []
        print('Optimization trails:',len(params_list))

        t = time.time()
        if len(params_list) > 250:            
            def func(_strategy, train, side, p): # type: ignore
                pf, p = _strategy(train, side, **p) # type: ignore
                return pf, p, pf.stats(), pf.value  # type: ignore

            result_list = Parallel(n_jobs=12,prefer=opt_type,verbose=0)(delayed(func)(self._strategy, train, side, p) for p in tqdm(params_list)) # type: ignore
            result_list = [[result[0],result[1],result[2],result[3]] for result in result_list] # type: ignore
        else:
            for p in tqdm(params_list):
                pf, p = self._strategy(train, side, **p) # type: ignore
                result_list.append([pf,p,pf.stats(),pf.value])

        print(f'Optimization time: {round(time.time()-t,2)} sec')
        result_df = pd.DataFrame(result_list,columns=['pf','params','stat','value'])
        for col in result_df['stat'].iloc[0].index:
            result_df[col] = result_df['stat'].apply(lambda x : x[col])
        result_df = result_df.drop(columns=['stat'])
        
        if direction=='max':
            try:
                result_df = result_df.sort_values(target,ascending=False)
            except:
                result_df = result_df.sort_values('End Value',ascending=False)
        elif direction=='min':
            result_df = result_df.sort_values(target,ascending=True)
        result_df['side'] = side
        return result_df.drop(columns=['pf'])
        del result_df
        gc.collect()


    def get_best_index(self,value_df,_start,_sep,target,direction):
        temp = value_df.loc[_start:_sep]
        # total return 要補calmar
        temp = temp.iloc[-1] - temp.iloc[0]
        best_index = temp.sort_values(ascending=False).index[0]
        return best_index
    
    def rolling_optimize(
            self,
            side='both',
            params={},
            intervals=[12,4],
            expanding = False,
            opt_type='for',
            target='Total Return [%]',
            direction='max',
            start = '',
            end = ''
            ):
        
        train_interval,test_interval = intervals
        df = self.df
        
        record_df = self.optimize(
                        side=side,
                        params=params,
                        opt_type=opt_type,
                        target=target,
                        direction=direction,
                        start = start,
                        end = end
                        )
        
        record_df['value'] = record_df['params'].apply(lambda p : self.strategy(side = side,params=p))
        record_df['value'] = record_df['value'].map(lambda x : x.value)

        if len(start) != 0:
            df = df.loc[start:] # type: ignore

        if len(end) != 0:
            df = df.loc[:end] # type: ignore

        start,end = str(df.index[0].date()),str(df.index[-1].date()) # type: ignore
        train_start_list = pd.date_range(start=start,end=end,freq=f'{test_interval}W')
        train_end_list = [start + pd.Timedelta(f'{train_interval}W') for start in train_start_list]
        test_end_list = [start + pd.Timedelta(f'{test_interval}W') for start in train_end_list]

        if expanding:
            exp_start = train_start_list[0]
            train_start_list = [exp_start for start in train_start_list]

        rolling_list = []

        for _start,_sep,_end in zip(train_start_list,train_end_list,test_end_list):
            if _end.timestamp() > pd.Timestamp(end).timestamp():
                rolling_list.append([_start,_sep,pd.Timestamp(end)])
                break
            rolling_list.append([_start,_sep,_end])
        print('Rolling trails: ',len(rolling_list))
        value_df = record_df['value'].apply(lambda x : x).T
        rolling_results = []
        for rolling_range in rolling_list:
            _start,_sep,_end = rolling_range
            best_index = self.get_best_index(value_df,_start,_sep,target,direction)
            params = record_df['params'].loc[best_index]
            train_value = value_df.loc[_start:_sep][best_index]
            test_value = value_df.loc[_sep:_end][best_index]
            rolling_results.append([best_index,params,train_value,test_value])
        
        rolling_df = pd.DataFrame(rolling_results,columns=['index','params','train_value','test_value'])
        rolling_df['start'] = rolling_df['train_value'].apply(lambda x :x.index[0])
        rolling_df['sep'] = rolling_df['train_value'].apply(lambda x :x.index[-1])
        rolling_df['end'] = rolling_df['test_value'].apply(lambda x :x.index[-1])
        rolling_ret = pd.concat([value.diff().iloc[1:] for value in rolling_df['test_value'].to_list()])
        rolling_value = rolling_ret.loc[rolling_ret.index.drop_duplicates()].cumsum()
        
        for p in rolling_df['params'].iloc[0].keys():
            rolling_df[p] = rolling_df['params'].apply(lambda x : x[p])

        # trades 
        _trades = []
        for index,row in rolling_df.iterrows():
            param,start,end = row['params'],row['sep'],row['end']
            _pf, _params = self._strategy(df.loc[start:end], side=side, **param) # type: ignore
            _trades.append(_pf.trades.records_readable)
        rolling_trades = pd.concat(_trades)

        rolling_value = rolling_value[~rolling_value.index.duplicated(keep="first")]
        return rolling_df,rolling_value,rolling_trades
    








    # ### new optimizatoin method
    # def _objective(self, trial, _train, _eval, side, params, target='Calmar Ratio'):
    #     trail_params = {}
    #     for key in params.keys():
    #         p = params[key]
    #         if type(params[key][2]) == int:
    #             trail_params[key] = trial.suggest_int(key, p[0], p[1])

    #         elif type(params[key][2]) == float :
    #             trail_params[key] = trial.suggest_float(key, p[0], p[1],step=p[2])
        
    #     _, _params = self._strategy(_train,side,**trail_params) # type: ignore
    #     outsample_pf, _ = self._strategy(_eval,side,**trail_params) # type: ignore

    #     if target != 'return':
    #         return 0 if math.isnan(outsample_pf.stats()[target]) else outsample_pf.stats()[target]
    #     else:
    #         return -999 if math.isnan(target.total_return()) else target.total_return() # type: ignore
    
    # def optimizatoin(
    #                 self, 
    #                 params={},
    #                 side='both',
    #                 target='Calmar Ratio',
    #                 n_trials=10, 
    #                 train_start='', 
    #                 eval_start='', 
    #                 test_start='', 
    #                 test_end='',
    #                 direction='maximize',
    #                 opt_type = 'optuna'
    #                 ):
        
    #     if opt_type == 'optuna':
    #         def func(trial): 
    #             return self._objective(trial, _train, _eval,side=side, params=params, target=target)

    #         _train = self.df[(self.df.index >= train_start) & ((self.df.index < eval_start))]
    #         _eval = self.df[(self.df.index >= eval_start) & ((self.df.index < test_start))]
    #         _test = self.df[(self.df.index >= test_start) & ((self.df.index < test_end))]
    #         print(train_start, eval_start, test_start, test_end)

    #         optuna.logging.set_verbosity(optuna.logging.ERROR)
    #         study = optuna.create_study(direction=direction)
    #         study.optimize(func, n_trials=n_trials, n_jobs=8)
            
    #         # print('best: ',study.best_value, study.best_trial.params)

    #         _pf, _params = self._strategy(_test, side=side, **study.best_trial.params) # type: ignore

    #         return _pf, _params, _pf.trades.records_readable, study
    #     else:
    #         pass

    # def rolling_optimizatoin(
    #         self,
    #         params={},
    #         side='both',
    #         target='Calmar Ratio', 
    #         direction='maximize',
    #         intervals=[0,0,0], 
    #         n_trials=100,
    #         train_start_date='', 
    #         test_end_date=''
    #         ):
        
    #     """
    #     Args:
    #         optimize (bool, optional): 最佳化. Defaults to False.
    #         rolling (bool, optional): 滾動. Defaults to False.
    #         intervals (list, optional): [train, eval, test] 單位為周. Defaults to [0, 0, 0].
    #         n_trials (int, optional): 最佳化iter數. Defaults to 10.
    #         train_start_date (str, optional): Train start date. Defaults to ''.
    #         eval_start_date (str, optional): Eval start date. Defaults to ''.
    #         test_start_date (str, optional): Test start date. Defaults to ''.
    #         test_end_date (str, optional): Train end date. Defaults to ''.
    #         params (dict, optional): Necessary if _optimize=False. Defaults to {}.

    #     Returns:
    #         _pf: vbt portfolio,
    #         _params: 參數組合,
    #         _trades: 交易流水
    #     """

    #     if len(train_start_date) == 0:
    #         train_start_date = str(self.df.index[0].date()) # type: ignore
        
    #     if len(test_end_date) == 0:
    #         test_end_date = str(self.df.index[-1].date()) # type: ignore

    #     _total_df = self.df[(self.df.index >= train_start_date) & ((self.df.index < test_end_date))]
    #     _train_interval, _eval_interval, _test_interval = intervals

    #     _pf_list = []
    #     _param_list = []
    #     _trade_list = []
    #     _start = train_start_date

    #     while True:

    #         _pf, _params, _trades, _ = self.optimizatoin(
    #                                             params=params,
    #                                             side=side,
    #                                             n_trials=n_trials,
    #                                             target=target,
    #                                             train_start=_start, # type: ignore
    #                                             eval_start=pd.to_datetime(_start)+pd.Timedelta(_train_interval, 'W'), # type: ignore
    #                                             test_start=pd.to_datetime(_start)+pd.Timedelta(_train_interval+_eval_interval, 'W'), # type: ignore
    #                                             test_end=pd.to_datetime(_start)+pd.Timedelta(_train_interval + _eval_interval+_test_interval, 'W'), # type: ignore
    #                                             direction=direction
    #                                             )
            
    #         _pf_list.append(_pf)
    #         _param_list.append(_params)
    #         _trade_list.append(_trades)
    #         _start = pd.to_datetime(_start)+pd.Timedelta(_test_interval, 'W')

    #         if _start+pd.Timedelta(_train_interval + _eval_interval+_test_interval, 'W') >= pd.to_datetime(test_end_date):
    #             break

    #     return _pf_list, _param_list, _trade_list
