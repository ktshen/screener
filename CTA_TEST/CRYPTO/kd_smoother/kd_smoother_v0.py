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
    df = pd.read_hdf(f'Y:\\price_data\\binance\\1m\\{pair}_PERPETUAL.h5')
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
        window_k = int(params['window_k'])
        window_d = int(params['window_d'])

        low_min = df['low'].rolling(window=window_k).min()
        high_max = df['high'].rolling(window=window_k).max()
        df['%K'] = ((df['close'] - low_min) / (high_max - low_min)) * 100

        # 計算 %D
        df['%D'] = df['%K'].rolling(window=window_d).mean()

        # 計算雙重平滑的 %D
        df['double_d'] = df['%D'].ewm(span=window_d, adjust=False).mean()
        df['double_dd'] = df['double_d'].ewm(span=window_d, adjust=False).mean()

        # 定義進場和出場訊號
        long_entry = (df['%D'] > df['double_dd']) & (df['%D'].shift(1) < df['double_dd'].shift(1))
        long_exit = (df['%D'] < df['double_d']) & (df['%D'].shift(1) > df['double_d'].shift(1))
        short_entry = (df['%D'] < df['double_dd']) & (df['%D'].shift(1) > df['double_dd'].shift(1))
        short_exit = (df['%D'] > df['double_d']) & (df['%D'].shift(1) < df['double_d'].shift(1))


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