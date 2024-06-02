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

import json
# config_f = open('..\configs\config.json')
# config = json.load(config_f)# %%

def get_data(coin):
    pair = f'{coin}USDT'
    try:
        df = pd.read_hdf(f'C:\\Users\\Intern\\Desktop\\{pair}_PERPETUAL.h5')
    except:
        df = pd.read_hdf(f'/Users/johnsonhsiao/Desktop/{pair}_PERPETUAL.h5')
    return df

class Strategy(BackTester):

    def __init__(self, df, configs, **kwargs):
        super().__init__(**kwargs)
        self.configs = configs
        self.freq = self.configs['freq']
        self.fee = self.configs['fee']
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
        # params
        vol_threshold = params['vol_threshold'] / 100
        ret_threshold = params['ret_threshold'] / 100
        # window = int(params['window'])
        # ma = df['close'].rolling(window).mean()
        
        df['weekday'] = df.index.weekday+1
        df['hour'] = df.index.hour
        df['return'] = df['close'] / df['open'] - 1
        
        df['weekend_vol'] = 0
        ret = 0
        i = 0
        for idx, row in df.iterrows():
            if row['weekday'] == (6 or 7):
                i += 1
                ret += abs(row['return'])
            elif row['weekday'] != (6 or 7):
                try:
                    df['weekend_vol'].loc[idx] = ret / i
                    if row['weekday'] == 1:
                        ret = 0
                        i = 0
                except:
                    pass
                    
        df['weekend_ret'] = 0
        for idx, row in df.iterrows():
            if (row['weekday'] == 6) and (row['hour'] == 0):
                o = row['open']
            if (row['weekday'] == 7) and (row['hour'] == 23):
                c = row['close']
            elif row['weekday'] != (6 or 7):
                try:
                    df['weekend_ret'].loc[idx] = c / o - 1
                except:
                    pass
                
        long_entry = (df['weekend_vol'] > vol_threshold) & (df['weekend_ret'] > ret_threshold) 
        long_exit = (df['weekday'] == 2) & (df['hour'] == 0)

        short_entry = (df['weekend_vol'] > vol_threshold) & (df['weekend_ret'] < -ret_threshold)
        short_exit = (df['weekday'] == 2) & (df['hour'] == 0)

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
                                        sl_stop= np.nan/100,
                                        upon_opposite_entry='reverse'
                                        )
        return pf, params