import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numba as nb
import pandas_ta 
import os
import datetime
import json
from tabulate import tabulate
import sys
sys.path.append('../..')
import vectorbtpro as vbt
from vectorbtpro.portfolio.enums import SizeType
from src.utils import fu
from src.utils import plot_return_mdd
from src.strategy.BackTester import BackTester
from src.strategy.Analyzer import Analyzer
# from src.strategy.PositionSizer import PositionSizer
from src.strategy.MultiTester import MultiTester
from src.utils import plot_return_mdd,twinx_plot # as utils
import pytz


def get_data(coin):
    try:
        pair = f'{coin}USDT'
        df = pd.read_hdf(f'Y:\\price_data\\binance\\1m\\{pair}_PERPETUAL.h5')
    except:
        df = pd.read_hdf(f'/Users/johnsonhsiao/Desktop/data/{pair}_PERPETUAL.h5')
    return df

class Strategy(BackTester):

    def __init__(self, df, configs, **kwargs):
        super().__init__(**kwargs)
        self.configs = configs
        self.freq = self.configs['freq']
        self.fee = self.configs['fee']
        self.weekend_filter = self.configs['weekend_filter']
        self.df = self.resample_df(df=df, freq=self.freq)
        self._strategy_setting()
        self.indicator = pd.DataFrame()
    
    def resample_df(self,df,freq = '1h'):
        cols = ['open', 'high', 'low', 'close','volume']
        agg =  ['first','max',  'min', 'last', 'sum']
        df = df[cols]
        df = df.resample(freq).agg(dict(zip(cols,agg)))
        return df.dropna()
    
    def _strategy(self, df, side='both', **params):
        df['timestamp'] = pd.to_datetime(df.index)

        eastern = pytz.timezone('America/New_York')
        df['timestamp_est'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(eastern)
        df['is_weekday'] = df['timestamp_est'].dt.dayofweek < 5  # 0-4代表周一到周五

        # params
        window_k = int(params['window_k'])
        window_d = int(params['window_d'])

        df.ta.stoch(high='high', low='low', close='close', k=window_k, d=window_d, append=True)
          
        df['double_d'] = df[f'STOCHd_{window_k}_{window_d}_3'].ewm(span=window_d, adjust=False).mean()
        df['double_dd'] = df['double_d'].ewm(span=window_d, adjust=False).mean()
        

        long_entry = (df[f'STOCHd_{window_k}_{window_d}_3'] > df['double_dd']) & \
                    (df[f'STOCHd_{window_k}_{window_d}_3'].shift(1) < df['double_dd'].shift(1)) 
        long_exit = (df[f'STOCHd_{window_k}_{window_d}_3'] < df['double_d']) 

        short_entry = (df[f'STOCHd_{window_k}_{window_d}_3'] < df['double_dd']) & \
                    (df[f'STOCHd_{window_k}_{window_d}_3'].shift(1) > df['double_dd'].shift(1)) 
        short_exit = (df[f'STOCHd_{window_k}_{window_d}_3'] > df['double_d'])
        
        if side == 'long':
            short_entry = False
            short_exit = False

        elif side == 'short':
            long_entry = False
            long_exit = False

        price = df['open'].shift(-self.lag)
        pf = vbt.Portfolio.from_signals(price, # type: ignore
                                        open = df['open'],
                                        high = df['high'],
                                        low  = df['low'],
                                        entries=long_entry,
                                        exits=long_exit,
                                        short_entries=short_entry,
                                        short_exits=short_exit,
                                        # sl_stop= 0.1,
                                        upon_opposite_entry='reverse'
                                        )
        return pf, params




# window_entry_k = int(params['window_entry_k'])
#         window_entry_d = int(params['window_entry_d'])
#         window_exit_k = int(params['window_exit_k'])
#         window_exit_d = int(params['window_exit_d'])

#         df.ta.stoch(high='high', low='low', close='close', k=window_entry_k, d=window_entry_d, append=True)
#         df.ta.stoch(high='high', low='low', close='close', k=window_exit_k, d=window_exit_d, append=True)

#         df['double_entry_d'] = df[f'STOCHd_{window_entry_k}_{window_entry_d}_3'].ewm(span=window_entry_d, adjust=False).mean()
#         df['double_entry_dd'] = df['double_entry_d'].ewm(span=window_entry_d, adjust=False).mean()
#         df['double_exit_d'] = df[f'STOCHd_{window_exit_k}_{window_exit_d}_3'].ewm(span=window_exit_k, adjust=False).mean()
#         df['double_exit_dd'] = df['double_exit_d'].ewm(span=window_exit_k, adjust=False).mean()
        
#         long_entry = (df[f'double_exit_d'] > df['double_entry_dd']) & \
#                      (df[f'double_exit_d'].shift(1) < df['double_entry_dd'].shift(1)) 
#         long_exit = (df[f'STOCHd_{window_exit_k}_{window_exit_d}_3'] < df['double_exit_dd']) 

#         short_entry = (df[f'double_exit_d'] < df['double_entry_dd']) & \
#                       (df[f'double_exit_d'].shift(1) > df['double_entry_dd'].shift(1))
#         short_exit = (df[f'STOCHd_{window_exit_k}_{window_exit_d}_3'] > df['double_exit_dd']) 
