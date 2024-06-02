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
    try:
        pair = f'{coin}USDT'
        df = pd.read_hdf(f'Y:\\price_data\\binance\\1m/{pair}_PERPETUAL.h5')
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
        short_ma = int(params['short_ma'])
        long_ma = int(params['long_ma']) + short_ma
        sl = int(params['sl'])
        
        if short_ma > long_ma:
            short_ma = 12
            long_ma = 12
        
        K = int(params['hour'])
        
        df['weekday'] = df.index.weekday+1
        df['hour'] = df.index.hour
        
        weekend = ((df['weekday'] == 5) & (df['hour'] >= 24-K)) | (df['weekday'] == 6) | ((df['weekday'] == 7) & (df['hour'] < 24-K))
        
        df['long_ma'] = df['close'].rolling(long_ma).mean()
        df['short_ma'] = df['close'].rolling(short_ma).mean()
        df['uband'] = df['short_ma'] + 2 * df['close'].rolling(window=short_ma).std()
        df['lband'] = df['short_ma'] - 2 * df['close'].rolling(window=short_ma).std()
        
        long_entry = (df['long_ma'].shift(1) < df['long_ma']) & (df['close'] < df['lband']) & weekend
        long_exit = (df['long_ma'].shift(1) < df['long_ma']) & (df['close'] > df['short_ma']) & ~weekend

        short_entry = (df['long_ma'].shift(1) > df['long_ma']) & (df['close'] > df['uband']) & weekend
        short_exit = (df['long_ma'].shift(1) > df['long_ma']) & (df['close'] < df['short_ma']) & ~weekend

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
                                        sl_stop= sl/100,
                                        upon_opposite_entry='reverse'
                                        )
        return pf, params