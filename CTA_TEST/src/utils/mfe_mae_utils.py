import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 以下的funciton都是給定trades和eth價格資料後，在return的trades後面欄位，最下面的是分佈圖

# 計算mfe, mae, mfe/mae
def mae_mfe_cal(trades, eth): 
    trades['mae'] = 0
    trades['mfe'] = 0
    for index, row in trades.iterrows():
        entry = row['Entry Index']
        exit = row['Exit Index']
        price = row['Avg Entry Price']
        min_price = eth['close'].astype(float)[(eth.index <= exit) & (eth.index > entry)].min()
        max_price = eth['close'].astype(float)[(eth.index <= exit) & (eth.index > entry)].max()
        if row['Direction']  == 'Long':
            if min_price > price:
                trades['mae'].loc[index] = 0 
            else:
                trades['mae'].loc[index] = abs(min_price / price - 1)
            if max_price < price:
                trades['mfe'].loc[index] = 0
            else:
                trades['mfe'].loc[index] = max_price / price - 1  
        else: #short
            if max_price < price:
                trades['mae'].loc[index] = 0
            else:
                trades['mae'].loc[index] = max_price / price - 1
            if min_price > price:
                trades['mfe'].loc[index] = 0
            else:
                trades['mfe'].loc[index] = abs(min_price / price - 1)

    trades['mfe/mae'] = 20 #因為可能沒有mae，所以就當作最大值，下面有用MDD來代替mae計算
    trades.loc[trades['mae'] != 0, 'mfe/mae'] = trades.loc[trades['mae'] != 0, 'mfe'] / trades.loc[trades['mae'] != 0, 'mae'] 
    trades.loc[trades['mfe/mae'] > 20, 'mfe/mae'] = 20 #因為有可能mae超小，比如只有一跳或是因為均價的關係，所以我把大於20都訂20，之後可以改
    return trades

#計算mae前的最大fe,越大代表可在遇到mae前獲利越大，要先執行mae_mfe_cal
def bmfe_cal(trades, eth): 
    trades['bmfe'] = 0
    for index, row in trades.iterrows():
        entry = row['Entry Index']
        exit = row['Exit Index']
        price = row['Avg Entry Price']
        mae = trades['mae'].loc[index]
        min_price = eth['close'].astype(float)[(eth.index <= exit) & (eth.index > entry)].min()
        max_price = eth['close'].astype(float)[(eth.index <= exit) & (eth.index > entry)].max()
        if row['Direction']  == 'Long':
            bmfe = 0
            while entry < exit:
                if eth['close'].loc[entry] != min_price and eth['close'].loc[entry] / price - 1 > bmfe:
                    bmfe = eth['close'].loc[entry] / price - 1
                elif eth['close'].loc[entry] == min_price:
                    break
                entry += pd.Timedelta(minutes=60)
        else:
            bmfe = 0
            while entry < exit:
                if eth['close'].loc[entry] != max_price and -(eth['close'].loc[entry] / price - 1) > bmfe:
                    bmfe = -(eth['close'].loc[entry] / price - 1)
                elif eth['close'].loc[entry] == max_price:
                    break
                entry += pd.Timedelta(minutes=60)
        trades['bmfe'].loc[index] = bmfe
    return trades

# 計算每一單的mdd
def mdd_cal(trades, eth):
    trades['mdd'] = 0
    for index, row in trades.iterrows():
        entry = row['Entry Index']
        exit = row['Exit Index']
        close = eth['close'].astype(float)[(eth.index <= exit) & (eth.index > entry)]
        if row['Direction']  == 'Long':
            mdd = 0
            max_price = close.cummax()
            drawdown = (close - max_price) / max_price
            mdd = -(drawdown.min())
        else:
            mdd = 0
            min_price = close.cummin()
            drawup = (close - min_price) / min_price
            mdd = drawup.max()
        trades['mdd'].loc[index] = mdd
    return trades

# 將mae換成mdd，計算在mdd出現前的mfe，要先執行mdd_cal
def mfe_before_mdd(trades, eth):
    trades['mfe_before_mdd'] = 0
    for index, row in trades.iterrows():
        entry = row['Entry Index']
        exit = row['Exit Index']
        price = row['Avg Entry Price']
        close = eth['close'].astype(float)[(eth.index <= exit) & (eth.index >= entry)]
        mdd = trades['mdd'].loc[index]
        if row['Direction']  == 'Long':
            bmdd = 0
            max_price = close.cummax()
            drawdown = (close - max_price) / max_price
            mdd = drawdown.min()
            while entry < exit:
                if eth['close'].loc[entry] / price -1 > bmdd:
                    bmdd = eth['close'].loc[entry] / price - 1
                elif drawdown.loc[entry] == -mdd:
                    break
                entry += pd.Timedelta(minutes=60)
        else :
            bmdd = 0
            min_price = close.cummin()
            drawup = (close - min_price) / min_price
            mdd = drawup.max()
            while entry < exit:
                if -(eth['close'].loc[entry] / price - 1) > bmdd:
                    bmdd = -(eth['close'].loc[entry] / price - 1)
                elif drawup.loc[entry] == mdd:
                    break
                entry += pd.Timedelta(minutes=60)
        trades['mfe_before_mdd'].loc[index] = bmdd
    trades['mfe/mdd'] = 0
    trades.loc[trades['mdd'] != 0, 'mfe/mdd'] = trades.loc[trades['mdd'] != 0, 'mfe_before_mdd'] / trades.loc[trades['mdd'] != 0,'mdd']
    return trades

#可以把trades換成只有虧損的單，就可以看到他的分佈
def plot_mae_mfe_dist(trades): 
    # sns.distplot(trades['mfe/mae'])
    # plt.title('MFE/MAE Distribution')
    # plt.xlabel('MFE')
    # plt.ylabel('Density')
    # plt.show()
    sns.distplot(trades['bmfe'])
    plt.title('MFE bfore MAE Distribution')
    plt.xlabel('MFE bfore MAE')
    plt.ylabel('Density')
    plt.show()
    sns.distplot(trades['mfe/mdd']) # 這個我覺得可以深入研究，再遇到mdd之前的最大獲利除以mdd
    plt.title('MFE before MDD Distribution')
    plt.xlabel('MFE before MDD')
    plt.ylabel('Density')
    plt.show()

