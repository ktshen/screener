# from .BackTester import BackTester
from .Analyzer import Analyzer
from tabulate import tabulate
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pandas_ta
import sys
import json


class MultiTester():
    
    def __init__(
        self,
        Strategy,
        params,
        df_dict,
        get_data_func,
        config={'freq':'5min','fee': 0.0005},
        symbol_list=['BTC','ETH','SOL','DOGE'],
        start='',
        end='',
        save_path=''
        ):
        
        self.Strategy = Strategy
        self.params = params
        self.get_data_func = get_data_func
        self.symbol_list = symbol_list
        self.config = config
        self.start = start
        self.end = end
        self.save_path = save_path
        self.df_cache = {}
        self.optimize_cache = {}
        self.rolling_cache = {}
        self.df_dict = df_dict

    def get_data(self,symbol,use_cache=True):
        if symbol in list(self.df_cache.keys()) and use_cache==True:
            return self.df_cache[symbol]
        # df = self.get_data_func(symbol)
        df = self.df_dict[symbol]
        if len(self.start)!=0:
            df = df.loc[self.start:]
        if len(self.end)!=0:
            df = df.loc[:self.end]
        self.df_cache[symbol] = df
        return df

    def run_optimize_test(self,symbol,side='L/S',sep='',df_use_cache=True):
        self.optimize_cache[symbol] = {}
        df = self.get_data(symbol,use_cache=df_use_cache)
        _sep = sep
        if _sep == '':
            _sep = str(df.index[-1])
        print(f'======================= {symbol.upper()} {side.upper()} Optimize Test =======================')
        strategy = self.Strategy(df=df, configs=self.config)
        analyze = Analyzer(strategy)
        if side == 'L/S':
            freq = self.config['freq']
            if not os.path.exists(f"{self.save_path}{freq}/{symbol}"):
                os.makedirs(f"{self.save_path}{freq}/{symbol}")
            print(f'\n---------- {symbol} Long ----------')
            long_record_df = strategy.optimize(
                                        side='long',
                                        params=self.params,
                                        opt_type='processes',
                                        target = 'Calmar Ratio',
                                        direction='max',
                                        end=sep
                                        )
            print(long_record_df['params'].iloc[0])
            analyze.show_pf_analysis(long_record_df['params'].iloc[0], 'long', symbol, axv_index=[_sep])
            self.optimize_cache[symbol]['long'] = long_record_df
            
            print(f'\n---------- {symbol} Short ----------')
            short_record_df = strategy.optimize(
                                        side='short',
                                        params=self.params,
                                        opt_type='processes',
                                        target='Calmar Ratio',
                                        direction='max',
                                        end=sep
                                        )
            print(short_record_df['params'].iloc[0])
            analyze.show_pf_analysis(short_record_df['params'].iloc[0], 'short', symbol, axv_index=[_sep])
            self.optimize_cache[symbol]['short'] = short_record_df

            print(f'-------- {symbol} L/S --------')
            long_pf = strategy.strategy(side = 'long', params=long_record_df.iloc[0]['params'])
            short_pf = strategy.strategy(side = 'short', params=short_record_df.iloc[0]['params'])
            value = (long_pf.value + short_pf.value - 2* long_pf.init_cash) * 100 / long_pf.init_cash
            analyze.show_value_analyze(value,f'{symbol} L/S',axv_index=[_sep])
            long_trades = long_pf.trades.records_readable 
            short_trades = short_pf.trades.records_readable
            trades = pd.concat([long_trades,short_trades]).sort_values('Entry Index')
            period_df = analyze.show_period_analysis(trades,period='Q')
            print(tabulate(period_df, headers='keys', tablefmt='psql')) # type: ignore
            if not os.path.exists(f"{self.save_path}{freq}/{symbol}"):
                os.makedirs(f"{self.save_path}{freq}/{symbol}")
            long_record_df.to_csv(f'{self.save_path}{freq}/{symbol}/long_record_df.csv')
            short_record_df.to_csv(f'{self.save_path}{freq}/{symbol}/short_record_df.csv')
        else:
            record_df = strategy.optimize(
                            side=side,
                            params=self.params,
                            opt_type='processes',
                            target = 'Calmar Ratio',
                            direction='max',
                            end=sep
                            )
            print(f'---------- {symbol} {side.upper()} ----------')
            print(record_df['params'].iloc[0])
            pf = strategy.strategy(side = side, params=record_df.iloc[0]['params'])
            trades = pf.trades.records_readable
            self.optimize_cache[symbol][side] = record_df
            analyze.show_pf_analysis(record_df['params'].iloc[0], side, symbol, axv_index=[_sep])
            period_df = analyze.show_period_analysis(trades,period='Q')
            print(tabulate(period_df, headers='keys', tablefmt='psql'))
            freq = self.config['freq']
            if not os.path.exists(f"{self.save_path}{freq}/{symbol}"):
                os.makedirs(f"{self.save_path}{freq}/{symbol}")
            record_df.to_csv(f"{self.save_path}{freq}/{symbol}/{side}_record_df.csv")

    def run_rolling_test(self,symbol,side='L/S',intervals=[16,4],expanding=True):
        self.rolling_cache[symbol] = {}
        df = self.get_data(symbol)
        print(f'======================= {symbol.upper()} {side.upper()} Rolling Test =======================')
        strategy = self.Strategy(df=df, configs=self.config)
        analyze = Analyzer(strategy)
        if side == 'L/S':
            long_rolling_df,long_value,long_trades= strategy.rolling_optimize(
                                                    side='long',
                                                    params=self.params,                                
                                                    intervals=intervals,
                                                    expanding = expanding,
                                                    opt_type='processes',
                                                    target='Sharpe Ratio',
                                                    direction='max',
                                                        )
            long_rolling_df['side'] = 'Long'
            self.rolling_cache[symbol]['long'] = {}
            # self.rolling_cache[coin]['long']['rolling_df'] = long_rolling_df
            self.rolling_cache[symbol]['long']['rolling_value'] = long_value
            self.rolling_cache[symbol]['long']['rolling_trades'] = long_trades

            print(f'\n---------- {symbol} Long Rolling ----------')
            long_value = (long_value/long_value.iloc[0])*100
            analyze.show_value_analyze((long_value),f'{symbol} - Long Rolling {intervals} {str(expanding)}')
            short_rolling_df,short_value,short_trades= strategy.rolling_optimize(
                                                    side='short',
                                                    params=self.params,                                
                                                    intervals=intervals,
                                                    expanding = expanding,
                                                    opt_type='processes',
                                                    target='Calmar Ratio',
                                                    direction='max',
                                                )
            self.rolling_cache[symbol]['short'] = {}
            # self.rolling_cache[coin]['short']['rolling_df'] = short_rolling_df
            self.rolling_cache[symbol]['short']['rolling_value'] = short_value
            self.rolling_cache[symbol]['short']['rolling_trades'] = short_trades
            short_rolling_df['side'] = 'Short'
            print(f'\n---------- {symbol} Short Rolling ----------')
            short_value = (short_value/short_value.iloc[0])*100
            analyze.show_value_analyze((short_value),f'{symbol} - Short Rolling {intervals} {str(expanding)}')
            rolling_df = pd.concat([long_rolling_df,short_rolling_df])
            trades = pd.concat([long_trades,short_trades])
            value = long_value+short_value
            print(f'\n---------- {symbol} L/S Rolling ----------')
            value = (value/value.iloc[0])*100
            analyze.show_value_analyze((value),f'{symbol} - L/S Rolling {intervals} {str(expanding)}')
        else:
            rolling_df,value,trades= strategy.rolling_optimize(
                                        side=side,
                                        params=self.params,                                
                                        intervals=intervals,
                                        expanding = expanding,
                                        opt_type='processes',
                                        target='Max Drawdown Duration',
                                        direction='min',
                                    )
            print(f'\n---------- {symbol} {side.upper()} Rolling ----------')
            value = (value/value.iloc[0])*100
            analyze.show_value_analyze((value),f'{symbol} - {side} Rolling {intervals} {str(expanding)}')

            self.rolling_cache[symbol][side] = {}
            # self.rolling_cache[coin][side]['rolling_df'] = rolling_df
            self.rolling_cache[symbol][side]['rolling_value'] = value
            self.rolling_cache[symbol][side]['rolling_trades'] = trades
    
    # def show_portofolio_analysis(self,symbol_list = [], _type = 'optimize' ,sep = ''):
    #     value_df = pd.DataFrame()
    #     strategy = None
    #     if len(symbol_list):
    #         symbol_list = self.optimize_cache
    #     for symbol in symbol_list:
    #         params = self.optimize_cache[symbol]
    #         for side in params:
    #             p = params[side]
    #             dd = self.get_data(symbol)
    #             strategy = self.Strategy(df=dd, configs=self.config)
    #             value_df[f'{symbol}-{side}'] = strategy.strategy(side=side,params=p).value
        
    #     value = value_df.mean(axis=1)
    #     value = 100*(value/value.iloc[0])
    #     analyze = Analyzer(strategy)
    #     analyze.show_value_analyze(value, f'Optimized Portofolio',axv_index=[sep])
    #     return value

    def run(self,optimize=True,rolling=False,side_list=['both','L/S','short'],sep='',intervals=[26,4],expanding=False,df_use_cache=True):
        for symbol in self.symbol_list:
            if optimize==True:
                for side in side_list:
                    try:
                        self.run_optimize_test(symbol,side=side,sep=sep,df_use_cache=df_use_cache)
                    except Exception as e:
                        traceback.print_exc()


            if rolling == True:
                for side in side_list:
                    self.run_rolling_test(symbol,side=side,intervals=intervals,expanding=expanding)

    def multi_symbols(self,sep):
        symbol_select_list = []
        if not self.optimize_cache:
            self.run(optimize=True,rolling=False,side_list=['L/S'],sep=sep)
        
        analyze = None
        for symbol in self.optimize_cache:
            temp_df = self.df_cache[symbol]
            strategy = self.Strategy(df=temp_df, configs=self.config)
            analyze = Analyzer(strategy)
            value_df = pd.DataFrame()
            trades = pd.DataFrame()
            for side in self.optimize_cache[symbol]:
                p = self.optimize_cache[symbol][side]
                _pf = strategy.strategy(side=side,params=p)
                value_df[side] = _pf.value
                trades = pd.concat([trades,_pf.trades.records_readable])
                
            value = value_df.mean(axis=1)
            value = (value-value.iloc[0])*100/value.iloc[0]
            calmar = analyze.get_value_attribute(value, 'Calmar Ratio')
            temp = analyze.show_period_analysis(trades)
            if calmar > 3 and len(temp[temp['收益率%'] < 0]) == 0: # type: ignore
                symbol_select_list.append(symbol)
        
        return symbol_select_list

    def multi_params(self, symbol_list, sample_sets = [['2022-01-01','2023-08-31']], direction='L/S'):
        all_params = {}
        origin_symbol_list = self.symbol_list
        origin_start = self.start
        origin_end = self.end
        orgin_opt_cache = self.optimize_cache
        self.symbol_list = symbol_list

        for sample,pid in zip(sample_sets,range(len(sample_sets))):
            self.optimize_cache = {}
            all_params[pid] = {}
            start = sample[0]
            end = sample[1]
            self.start = start 
            self.end = end
            self.run(optimize=True,rolling=False,side_list=[direction],sep='',df_use_cache=False)
            all_params[pid] = self.optimize_cache

        self.symbol_list = origin_symbol_list
        self.start = origin_start
        self.end = origin_end
        self.optimize_cache = orgin_opt_cache
        
        direction_list = []
        if direction == 'L/S':
            direction_list = ['long','short']
        else:
            direction_list = [direction]

        strategy_params = {}
        for symbol in symbol_list:
            strategy_params[symbol] = {}
            for side in direction_list:
                strategy_params[symbol][side] = []
                for pid in range(len(sample_sets)):
                    params_record = all_params[pid][symbol][side]['params']
                    for i in range(len(params_record)):
                        if params_record.iloc[i] not in strategy_params[symbol][side]:
                            strategy_params[symbol][side].append(params_record.iloc[i])
                            break

        for symbol in symbol_list:
            for side in direction_list:
                params_list = strategy_params[symbol][side]
                strategy_params[symbol][side] = {}
                for pid in range(len(sample_sets)):
                    strategy_params[symbol][side][pid] = params_list[pid]
        
        return strategy_params
    
    def multi_symbols_mulit_params(self, sep, sample_sets = [['2022-01-01','2023-08-31'],['2022-01-01','2022-12-31'],['2023-01-01','2023-08-31']]):
        symbol_list = self.multi_symbols(sep)
        all_params = self.multi_params(symbol_list, sample_sets)
        return all_params
    
    def multi_params_result(self,all_params):
        all_trades = pd.DataFrame()
        value_df = pd.DataFrame()
        for symbol in all_params:
            temp_df = self.get_data(symbol, use_cache=False)
            for side in all_params[symbol]:
                for pid in all_params[symbol][side]:
                    params = all_params[symbol][side][pid]
                    strategy = self.Strategy(df=temp_df, configs=self.config)
                    _pf = strategy.strategy(side = side,params=params)
                    value = _pf.value
                    value = (value-value.iloc[0])*100/value.iloc[0]
                    value_df[f'{symbol}-{side}-{pid}'] = value

                    trades = _pf.trades.records_readable
                    trades['symbol'] = symbol
                    trades['side'] = side
                    trades['pid'] = pid
                    all_trades = pd.concat([all_trades,trades])

        return all_trades, value_df
    
    def save_version_result(self,params,value_df,trades,version):
        if os.path.exists(f"v{version}"):
            print(f"v{version} dictory exists!!")
            return 
        
        
        def tran_np_to_float(data: dict):
            for key, value in data.copy().items():
                if type(value) == np.float64 or type(value) == np.int64:
                    data[key] = float(value)
                if type(value) == dict:
                    data[key] = tran_np_to_float(value)
            return data

        save_json = {}
        save_json['config'] = self.config
        save_json['params'] = params
        save_json = tran_np_to_float(save_json)
        json_object = json.dumps(save_json, indent=4)

        if not os.path.exists(f"v{version}"):
            os.makedirs(f"v{version}")

        with open(f"v{version}/params.json", "w") as outfile:
            outfile.write(json_object)
        value_df.to_pickle(f'v{version}/_values.pkl')
        trades.to_pickle(f'v{version}/_trades.pkl')


    def get_specific_value_df(self, value_df, symbol = None, side = None, pid = None):
        cols = value_df.columns # type: ignore
        if symbol is not None:
            cols =  [col for col in cols if symbol in col]
        if side is not None:
            cols =  [col for col in cols if side in col]
        if pid is not None:
            cols =  [col for col in cols if f'{pid}' in col]
        return value_df[cols]