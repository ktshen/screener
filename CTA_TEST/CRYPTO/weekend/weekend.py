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
        window = int(params['window'])
        K = int(params['hour'])
        
        df['weekday'] = df.index.weekday+1
        df['hour'] = df.index.hour
        weekend = ((df['weekday'] == K) & (df['hour'] >= 24-K)) | (df['weekday'] == 6) | ((df['weekday'] == 7) & (df['hour'] < 24-K))
        
        donchian_high = df['high'].rolling(window).max().shift()
        donchian_low = df['low'].rolling(window).min().shift()
        
        # 進場
        long_entry = (df['close'] < donchian_low)  & (weekend == True) 
        long_exit =  (df['close'] > donchian_high) | (weekend == False)

        # 出場
        short_entry = (df['close'] > donchian_high)  & (weekend == True)
        short_exit =  (df['close'] < donchian_low) | (weekend == False)

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