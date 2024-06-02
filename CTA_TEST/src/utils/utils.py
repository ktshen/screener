import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_return_mdd(total_return, tag='', axv_index=[],txt=''):
    fig, ax = plt.subplots(figsize=(16, 5))
    MDD_series = total_return.cummax()-total_return
    high_index = total_return[total_return.cummax() == total_return].index
    (total_return).plot(label='Total Return', ax=ax, c='r')
    mdd = round(max(MDD_series), 2)
    plt.fill_between(MDD_series.index, -MDD_series,
                        0, facecolor='r', label='DD')
    plt.scatter(
        high_index, total_return.loc[high_index], c='#02ff0f', label='High')
    plt.legend()
    plt.ylabel('Return%')
    plt.xlabel('Date')
    for index in axv_index:
        plt.axvline(index, color='black', zorder=0)

    if tag == '':
        plt.title(f'Return & MDD ({mdd})', fontsize=16)
    else:
        plt.title(f'Return & MDD ({mdd}) - {tag}', fontsize=16)
    plt.text(0.9, 0.2, txt, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)
    plt.show();


def show_value_analyze(value, tags='',axv_index=[]):
        years = (value.index[-1] - value.index[0]).days/365
        apy = value.iloc[-1]/years
        mdd = max(value.cummax() - value)
        calmar = apy/mdd
        ret_series = value.resample('1d').last().diff().fillna(0)
        n = 365
        sharpe = ret_series.mean()*n/(ret_series.std() * np.sqrt(n))
        sortino = ret_series.mean()*n/(ret_series[ret_series<0].std() * np.sqrt(n))
        print(f'APY: {apy:.2f} %')
        print(f'MDD: {mdd:.2f} %')
        print('-------------------------')
        print(f'Sharpe: {sharpe:.2f}')
        print(f'Calmar: {calmar:.2f}')
        print(f'Sortino: {sortino:.2f}')
        print('-------------------------')
        plot_return_mdd(value ,tags,axv_index=axv_index)



def twinx_plot(df, col1, col2):
    fig, ax1 = plt.subplots()
    ax1.set_title(f"{col1} & {col2}")
    ax1.plot(df[col1])
    ax1.set_ylabel(col1)
    ax2 = ax1.twinx()
    ax2.plot(df[col2], color='teal')
    ax2.set_ylabel(col2)
