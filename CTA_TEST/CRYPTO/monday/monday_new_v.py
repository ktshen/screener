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
        if row['weekday'] == 6 or row['weekday'] == 7:
            i += 1
            ret += abs(row['return'])
        elif row['weekday'] != (6 or 7):
            try:
                df['weekend_vol'].loc[idx] = ret / i
                if row['weekday'] == 1 and row['hour'] == 1:
                    ret = 0
                    i = 0
            except:
                pass
            
    df['weekend_ret'] = 0
    for idx, row in df.iterrows():
        if (row['weekday'] == 6) and (row['hour'] == 0):
            o = row['open']
        elif (row['weekday'] == 7) and (row['hour'] == 23):
            c = row['close']
        elif row['weekday'] == (1):
            try:
                df['weekend_ret'].loc[idx] = c / o - 1
            except:
                pass
            
    long_entry = (df['weekend_vol'] > vol_threshold) & (5*ret_threshold > df['weekend_ret']) & (df['weekend_ret'] > ret_threshold) #& \
                #  (df['close'] > ma)
    long_exit = (df['weekday'] == 5) & (df['hour'] == 22)
    
    short_entry = (df['weekend_vol'] > vol_threshold) & (-5*ret_threshold < df['weekend_ret']) & (df['weekend_ret'] < -ret_threshold) #& \
                #   (df['close'] < ma)
    short_exit = (df['weekday'] == 5) & (df['hour'] == 22)