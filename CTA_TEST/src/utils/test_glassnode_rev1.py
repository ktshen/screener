# %%
import simplejson
import pandas.io.json
import json
import requests
import os
import numpy as np
import pandas as pd
import tqdm
import warnings
import datetime
warnings.filterwarnings("ignore")


# monkeypatch using standard python json module
pd.io.json._json.loads = lambda s, *a, **kw: json.loads(s)

# monkeypatch using faster simplejson module
pd.io.json._json.loads = lambda s, *a, **kw: simplejson.loads(s)

# normalising (unnesting) at the same time (for nested jsons)
pd.io.json._json.loads = lambda s, * \
    a, **kw: pandas.io.json.json_normalize(simplejson.loads(s))

# insert your API key here
API_KEY = '2GdBkvVpi3E9FeK1HyQAe846zBg'
URI = 'https://api.glassnode.com/v1/metrics/'
DIR_PATH = 'X:\\glassnode'
TARGET_ENDPOINTS = [
    # 'derivatives/options_atm_implied_volatility_6_months',
    'derivatives/options_25delta_skew_6_months'
]
TRAGET_TIMEFRAME = [
    '10m'
]
# %%


def make_request(endpoint, params):
    print(f'{URI}{endpoint}')
    res = requests.get(f'{URI}{endpoint}', params=params)
    # convert to pandas dataframe
    # print(res.text)
    return pd.read_json(res.text, convert_dates=['t'])


def check_dir(title):
    _path = os.path.join(DIR_PATH, title.split('/')[0])
    if not os.path.exists(_path):
        os.makedirs(_path)
    _path = os.path.join(_path, title.split('/')[1])
    if not os.path.exists(_path):
        os.makedirs(_path)
    return _path


# %%
import time
if __name__ == '__main__':
    _last_timestmap = 1690506600
    _start_min = [0, 6, 8]
    _agg = 2
    while True:
        if datetime.datetime.now().minute % 10 in _start_min:
            for _name in TARGET_ENDPOINTS:
                for _time in TRAGET_TIMEFRAME:
                    _params = {
                        'api_key': API_KEY,
                        'a': 'ETH',
                        's': _last_timestmap,
                        'i': _time,
                        'e': 'deribit',
                    }
                    # print(_params)
                    resp = make_request(_name, _params)
                    if _last_timestmap != (_timestamp:=(resp['t'].values[-1].astype(np.int64) // 10**9)):
                        _last_timestmap = _timestamp
                        print(resp, datetime.datetime.now())
            time.sleep(60)


# %%
