# 1. screener cal: 1h, 4h. duration=5days.
# 5days / 4h = 30 kbars.

# 2. 1h + bollinger band, select top-N pairs.

# 3. stop loss: ATR


# another
# - -> +

import pandas as pd
import matplotlib.pyplot as plt
import os.path

from strategy_long import long_atr_tp, long_bband_tp
from strategy_short import short_atr_tp, short_bband_tp

def run_backtest(symbol, dates):
    if symbol == '1INCHUSDT' or symbol == 'GMXUSDT' or symbol == 'USTCUSDT':
        return 0
    
    filename = f'./data/UPERP/1h/{symbol}_UPERP_1h.csv'
    if not os.path.isfile(filename):
        print('{symbol} csv not found')
        return 0
    df = pd.read_csv(filename) # datetime,open,high,low,close,volume
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    date_list = pd.to_datetime(dates).date
    df.loc[df['date'].isin(date_list), 'can_entry'] = 1
    df.reset_index(drop=True, inplace=True)
    # print(symbol, df.to_string())

    # df = short_atr_tp(df)
    # df = short_bband_tp(df)
    # df = long_bband_tp(df) # 40.48%, 2.8
    df = long_atr_tp(df) # 42.86%, 6.3 因為都沒出場

    # 计算每日回报
    df['daily_return'] = df['close'].pct_change()
    df['strategy_return'] = df['daily_return'] * df['position'].shift(1)

    # 计算累计回报
    df['cumulative_market_return'] = (1 + df['daily_return']).cumprod()
    df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod()

    # 绘制回报曲线
    # print(df[['datetime','close','signal','position', 'take_profit', 'stop_loss', 'upper_band','daily_return','strategy_return']].to_string())
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['cumulative_market_return'], label='Market Return')
    # plt.plot(df['cumulative_strategy_return'], label='Strategy Return')
    # plt.legend()
    # plt.title('Cumulative Returns')
    # plt.show()

    # 分析回测结果
    total_return = df['cumulative_strategy_return'].iloc[-1] - 1
    # annualized_return = df['strategy_return'].mean() * 252
    # annualized_volatility = df['strategy_return'].std() * (252 ** 0.5)
    # sharpe_ratio = annualized_return / annualized_volatility

    # print(f'{symbol}: {total_return:.2%}')
    # print(f'Annualized Return: {annualized_return:.2%}')
    # print(f'Annualized Volatility: {annualized_volatility:.2%}')
    # print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    return total_return

def get_top_n(n):
    df = pd.read_csv('abc1.csv', index_col=0)

    def extract_symbol_quantity(cell_value):
        symbol, rs_value = cell_value.split('_')
        return symbol, float(rs_value)

    top_n_dict = {}

    len_col = len(df.columns)
    for _, row in df.iterrows():
        date = row['date']
        # for column in df.columns[1:n+2]: # get weakest top-n
        for column in df.columns[len_col - n:len_col]: # get strongest top-n
            cell = row[column]
            symbol, rs_value = extract_symbol_quantity(cell)
            if symbol not in top_n_dict:
                top_n_dict[symbol] = []
            top_n_dict[symbol].append(date)
    return top_n_dict

if __name__ == '__main__':
    n = 5
    top_n_dict = get_top_n(n)
    total_profit = 0
    total_win_money = 0
    total_loss_money = 0
    win_times = 0
    loss_times = 0
    for symbol, dates in top_n_dict.items():
        profit = run_backtest(symbol, dates)
        if profit > 0:
            total_win_money += profit
            win_times += 1
        else:
            total_loss_money += profit
            loss_times += 1
        total_profit += profit
        print(f'{symbol} {profit:.2%} {total_profit:.2%}')
    print(f"Win Rate: {(win_times / (win_times + loss_times)):.2%}")
    print(f"Profit Factor: {abs(total_win_money / total_loss_money):.2}")