import pandas as pd

def loads(_path, **kwargs):
    if _path.suffix == '.csv':
        return pd.read_csv(_path, **kwargs)
    elif _path.suffix == '.h5':
        return pd.read_hdf(_path, **kwargs)
    elif _path.suffix == '.pkl':
        return pd.read_pickle(_path, **kwargs)


def resample_ohlcv(df, rule):
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades_num': 'sum',
        'taker_buy_volume': 'sum',
        'taker_buy_quote_volume': 'sum'
    }
    df = df.resample(rule).agg(agg_dict)
    return df
