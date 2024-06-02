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
        
        # params
        short_window = int(params['short_window'])
        middle_window = int(params['middle_window']) + short_window
        long_window = int(params['long_window']) + middle_window

        df['short_ma'] = df['close'].rolling(window=short_window).mean()
        df['middle_ma'] = df['close'].rolling(window=middle_window).mean()
        df['long_ma'] = df['close'].rolling(window=long_window).mean()
        
        # 多單
        long_entry = (df['short_ma'].shift(1) < df['middle_ma'].shift(1)) & \
                    (df['short_ma'] > df['middle_ma']) & (df['long_ma'] > df['long_ma'].shift(1)) #& RV_filter
        long_exit = ((df['short_ma'].shift(1) > df['middle_ma'].shift(1)) & (df['short_ma'] < df['middle_ma'])) | \
                    ((df['middle_ma'].shift(1) > df['long_ma'].shift(1)) & (df['middle_ma'] < df['long_ma']))

        df['short_ma'] = df['close'].rolling(window=short_window).mean()
        df['middle_ma'] = df['close'].rolling(window=middle_window).mean()
        df['long_ma'] = df['close'].rolling(window=long_window).mean()

        # 空單
        short_entry = (df['short_ma'].shift(1) > df['middle_ma'].shift(1)) & \
                    (df['short_ma'] < df['middle_ma']) & (df['long_ma'] < df['long_ma'].shift(1)) #& RV_filter
        short_exit = ((df['short_ma'].shift(1) < df['middle_ma'].shift(1)) & (df['short_ma'] > df['middle_ma'])) | \
                    ((df['middle_ma'].shift(1) < df['long_ma'].shift(1)) & (df['middle_ma'] > df['long_ma']))

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