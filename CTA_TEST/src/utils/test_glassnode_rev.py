# %%
import simplejson
import pandas.io.json
import json
import requests
import os
import pandas as pd
import tqdm
import warnings
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
    'derivatives/options_atm_implied_volatility_6_months',
    'derivatives/options_25delta_skew_6_months'
]
TRAGET_TIMEFRAME = [
    '10m',
    '1h',
    '24h'
]
# %%


def make_request(endpoint, params):
    print(f'{URI}{endpoint}')
    res = requests.get(f'{URI}{endpoint}', params=params)
    # convert to pandas dataframe
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
if __name__ == '__main__':
    for _name in TARGET_ENDPOINTS:
        for _time in TRAGET_TIMEFRAME:
            _path = os.path.join(DIR_PATH, _name, 'deribit', _time)
            for _file in os.listdir(_path):
                _df = pd.read_hdf((_target_file:=os.path.join(_path, _file)))
                _params = {
                    'api_key': API_KEY,
                    'a': _file.split('_')[0],
                    's': int(_df.index[-1].timestamp()),
                    'i': _time,
                    'e': 'deribit',
                }
                # print(_params)
                resp = make_request(_name, _params)
                pd.concat([_df[:-1], resp.set_index('t')]).to_hdf(_target_file, key='_', format='t')
            
    # endpoints = make_request('/v2/metrics/endpoints', {'api_key': API_KEY})
    # # %%
    # import itertools
    # stop = True
    # for idx, row in endpoints.iterrows():
    #     if row['path'].split('metrics/')[-1].split('/')[0] == 'market':
    #         if stop:
    #             print(row['path'])
    #             dir_path = check_dir(row['path'].split('metrics/')[-1])
    #             symbols = [d['symbol'] for d in row['assets']]
    #             exchanges = []
    #             for d in row['assets']:
    #                 if 'exchanges' in d.keys():
    #                     exchanges += d['exchanges']
    #             exchanges = list(set(exchanges))
    #             if len(exchanges) == 0:
    #                 params = [symbols, row['currencies'], row['resolutions']]
    #                 for symbol, currency, resolution in itertools.product(*params):
    #                     dir_path_exchange = os.path.join(dir_path, resolution)
    #                     if not os.path.exists(dir_path_exchange):
    #                         os.makedirs(dir_path_exchange)
    #                     if not os.path.exists(os.path.join(dir_path_exchange, f'{symbol}_{currency}.h5')):
    #                         df = make_request(
    #                             row['path'], {'api_key': API_KEY, 'a': symbol, 'i': resolution, 'c': currency})

    #                         try:
    #                             df.set_index('t').to_hdf(os.path.join(
    #                                 dir_path_exchange, f'{symbol}_{currency}.h5'), key='_', format='t')
    #                         except Exception as e:
    #                             print(symbol, currency, resolution)
    #                             print(e)
    #                             df.to_hdf(os.path.join(
    #                                 dir_path_exchange, f'{symbol}_{currency}.h5'), key='_', format='t')

    #             else:
    #                 params = [symbols, exchanges,
    #                           row['currencies'], row['resolutions']]
    #                 for symbol, exchange, currency, resolution in itertools.product(*params):
    #                     dir_path_exchange = os.path.join(
    #                         dir_path, exchange, resolution)
    #                     if not os.path.exists(dir_path_exchange):
    #                         os.makedirs(dir_path_exchange)
    #                     if not os.path.exists(os.path.join(dir_path_exchange, f'{symbol}_{currency}.h5')):
    #                         try:
    #                             df = make_request(
    #                                 row['path'], {'api_key': API_KEY, 'e': exchange, 'a': symbol, 'i': resolution, 'c': currency})
    #                             df.set_index('t').to_hdf(os.path.join(
    #                                 dir_path_exchange, f'{symbol}_{currency}.h5'), key='_', format='t')
    #                         except:
    #                             print(symbol, exchange, currency, resolution)
    #                             df.to_hdf(os.path.join(
    #                                 dir_path_exchange, f'{symbol}_{currency}.h5'), key='_', format='t')

# %%
