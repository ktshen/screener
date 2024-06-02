import itertools
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import hiplot as hip
from vectorbt.portfolio.enums import SizeType

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


class Analyzer():

    def __init__(self, strategy):
        self.strategy = strategy
        self.df = self.strategy.df # merged raw data
        self.df_freq_sec = (self.df.index[1] - self.df.index[0]).total_seconds()
        self.price = self.df['open'].shift(-1)
        self.strategy_name = type(strategy).__name__
    
    def get_value_attribute(self,value, attribute='Sharpe Ratio'):
        if attribute == 'APY':
            years = (value.index[-1] - value.index[0]).days/365
            apy = value.iloc[-1]/years
            return apy
        elif attribute == 'MDD':
            MDD_series = value.cummax()-value
            return max(MDD_series)
        elif attribute == 'Calmar Ratio':
            years = (value.index[-1] - value.index[0]).days/365
            apy = value.iloc[-1]/years
            mdd = max(value.cummax() - value)
            calmar = apy/mdd
            return calmar
        elif attribute == 'Sharpe Ratio':
            n = 365
            ret_series = value.resample('1d').last().diff().fillna(0)
            sharpe = ret_series.mean()*n/(ret_series.std() * np.sqrt(n))
            return sharpe
        elif attribute == 'Sortino Ratio':
            n = 365
            ret_series = value.resample('1d').last().diff().fillna(0)
            sortino = ret_series.mean()*n/(ret_series[ret_series<0].std() * np.sqrt(n))
            return sortino
    
    # value analysis
    def plot_return_mdd(self, total_return, tag='', axv_index=[], txt =''):
        fig, ax = plt.subplots(figsize=(16, 5))
        
        if (total_return.index[1]-total_return.index[0]).total_seconds() < 60:
            total_return = total_return.resample('1min').last().ffill()

        MDD_series = total_return.cummax()-total_return
        high_index = total_return[total_return.cummax() == total_return].index
        (total_return).plot(label='Total Return', ax=ax, c='r')
        mdd = round(max(MDD_series), 2)
        ax.fill_between(MDD_series.index, -MDD_series, 0, facecolor='r', label='DD')
        ax.scatter(high_index, total_return.loc[high_index], c='#02ff0f', label='High')
        ax.legend()
        plt.ylabel('Return%')
        plt.xlabel('Date')
        for index in axv_index:
            ax.axvline(index, color='black', zorder=0)

        if tag == '':
            plt.title(f'Return & MDD ({mdd})', fontsize=16)
        else:
            plt.title(f'Return & MDD ({mdd}) - {tag}', fontsize=16)
        if txt!='':
            plt.text(0.85, 0.2, txt, verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)

        plt.show()

    # value analysis
    def show_value_analyze(self, value, tags='',axv_index=[]):
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
        self.plot_return_mdd(value ,tags,axv_index=axv_index)

    # pf + strategy.df analysis => strategy + params + side
    def show_pf_analysis(self, params={}, side='both', tag='', axv_index=[]):
        pf = self.strategy.strategy(side = side,params=params)
        years = (self.df.index[-1] - self.df.index[0]).days/365
        stat = pf.stats()
        # print(stat)
        start_value = stat['Start Value']
        _trades = pf.trades.records_readable
        _trades['holding period'] = _trades['Exit Index'] - _trades['Entry Index']
        _trades['holding days'] = _trades['holding period'].dt.total_seconds() / 86400
        avg_holding_days = _trades['holding days'].mean()
        worst_entry = _trades.iloc[_trades['PnL'].argmin()]['Entry Index']
        worst_exit = _trades.iloc[_trades['PnL'].argmin()]['Exit Index']
        apy = stat['Total Return [%]']/years
        mdd = stat['Max Drawdown [%]']
        worst_trade = stat['Worst Trade [%]']
        print(f'APY: {apy:.2f} %')
        print(f'MDD: {mdd:.2f} %')
        print('MDD Duration',stat['Max Drawdown Duration'])
        print(f'worst trade: {worst_trade:.2f} %')
        print(worst_entry,'~',worst_exit)
        print(f'avg holding days: {avg_holding_days:.2f} days')
        print('--------------------------------------------')
        profit_f = stat['Profit Factor']
        sharpe = stat['Sharpe Ratio']
        calmar = stat['Calmar Ratio']
        sortino = stat['Sortino Ratio']
        omega = stat['Omega Ratio']
        print(f'PF: {profit_f:.2f}')
        try:
            print(f'Sharpe Ratio: {sharpe:.2f}')
            print(f'Calmar Ratio: {calmar:.2f}')
            print(f'Sortino Ratio: {sortino:.2f}')
            print(f'Omega Ratio: {omega:.2f}')
        except:
            pass
        print('--------------------------------------------')
        time_exposure = stat['Total Time Exposure [%]']
        win_rate = stat['Win Rate [%]']
        exp = (stat['Expectancy']/pf.init_cash)*100
        print(f'Expectancy : {exp:.2f} %')
        print(f'Total Time Exposure : {time_exposure:.2f} %')
        print('Total Trades:', stat['Total Trades'])
        total_trades = stat['Total Trades']
        print(f'Win Rate: {win_rate:.2f} %')
        print('--------------------------------------------')
        wrpf = win_rate*profit_f/100
        print(f'PF * Win Rate: {wrpf:.3f}')
        value_series = (pf.value/start_value)*100
        value_series = value_series - 100
        self.plot_return_mdd(value_series.ffill(), tag=tag, axv_index=axv_index, \
                                   txt=f'APY: {apy:.2f} %\nMDD: {mdd:.2f} %\nSharpe Ratio: {sharpe:.2f}\nWin Rate: {win_rate:.2f} %\nTotal Trades: {total_trades:.2f}\nPF: {profit_f:.2f}')
        return value_series

    # trades analysis, df
    def plot_signal_response(self,trades):
        trades['holding period'] = trades['Exit Index'] - trades['Entry Index']
        trades['holding days'] = trades['holding period'].dt.total_seconds() / 86400

        long_record = trades[trades['Direction'] == 'Long'][['Entry Index']]
        long_holdings = trades[trades['Direction'] == 'Long']['holding days']
        short_record = trades[trades['Direction'] == 'Short'][['Entry Index']]
        short_holdings = trades[trades['Direction'] == 'Short']['holding days']
        if len(short_record) == 0:
            hours = int(long_holdings.quantile(0.99)*24+1)+1
        elif len(long_record) == 0:
            hours = int(short_holdings.quantile(0.99)*24+1)+1
        else:
            hours = int(max(short_holdings.quantile(0.99), long_holdings.quantile(0.99))*24+1)+1 # type: ignore
        
        agg_1h_Kbar = int(3600/self.df_freq_sec)
        for i in range(1, hours):
            i = i*agg_1h_Kbar
            ret = (self.df['close'].pct_change(i).shift(-i))*100
            long_record[i] = long_record['Entry Index'].apply(lambda x: ret.loc[x])
            short_record[i] = short_record['Entry Index'].apply(lambda x: ret.loc[x])

        long_record = long_record.set_index('Entry Index')
        long_record.columns = long_record.columns*self.df_freq_sec/86400
        short_record = short_record.set_index('Entry Index')
        short_record.columns = short_record.columns*self.df_freq_sec/86400

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(long_record.mean())
        ax1.axhline(y=0, color='teal', ls='--')
        ax1.set_title('Long Signal Response')
        ax1.set_xlabel('days')
        ax1.set_ylabel('mean return %')
        ax2 = ax1.twinx()
        ax2.hist(long_holdings, bins=100, color='silver')
        ax2.set_ylabel('counts')

        ax3.plot(short_record.mean())
        ax3.axhline(y=0, color='teal', ls='--')
        ax3.set_title('Short Signal Response')
        ax3.set_xlabel('days')
        ax3.set_ylabel('mean return %')
        ax4 = ax3.twinx()
        ax4.hist(short_holdings, bins=100, color='silver')
        ax4.set_ylabel('counts')

        plt.show();
    
    # trades analysis
    def plot_ret_dist(self,trades):
        ret_distribution = (trades['Return']*100).rename('Return Rate%')
        q1 = ret_distribution.quantile(0.25)
        q3 = ret_distribution.quantile(0.75)
        mean = ret_distribution.mean()
        median = ret_distribution.median()
        plt.title('Transaction PnL Distribution')
        sns.histplot(ret_distribution, bins=100) # type: ignore
        plt.axvline(x=q1, color='teal', ls='--',
                    label="Q1 = {:.2f}%".format(q1))
        plt.axvline(x=q3, color='orange', ls='--',
                    label="Q3 = {:.2f}%".format(q3))
        plt.axvline(x=mean, color='red', ls='--',
                    label="Mean = {:.2f}%".format(mean))
        plt.axvline(x=median, color='black', ls='--',
                    label="Median = {:.2f}%".format(median))
        plt.legend(loc="best")
        plt.show();

    # trades analysis
    def plot_holding_period_dist(self,trades):
        holding_period = trades['Exit Index']-trades['Entry Index']
        holding_days = holding_period.dt.total_seconds()/86400
        q0 = holding_days.quantile(0)
        q05 = holding_days.quantile(0.05)
        q1 = holding_days.quantile(0.25)
        q2 = holding_days.quantile(0.5)
        q3 = holding_days.quantile(0.75)
        q95 = holding_days.quantile(0.95)
        plt.title('Holding Period Distribution')
        sns.histplot(holding_days.rename('Holding Days'), bins=100) # type: ignore
        plt.axvline(x=q0, color='gray', ls='--')
        plt.axvline(x=q05, ls='--', label="5%   = {:.2f} days".format(q05))
        plt.axvline(x=q1, color='#0343DF', ls='--',
                    label="25% = {:.2f} days".format(q1))
        plt.axvline(x=q2, color='teal', ls='--',
                    label="50% = {:.2f} days".format(q2))
        plt.axvline(x=q3, color='orange', ls='--',
                    label="75% = {:.2f} days".format(q3))
        plt.axvline(x=q95, color='red', ls='--',
                    label="95% = {:.2f} days".format(q95))
        plt.legend(loc="best")
        plt.show();
        
    # trades analysis
    def show_period_analysis(self,trades,period='Q'):
        trades['Return%'] = trades['Return']*100
        trades = trades[['Entry Index', 'PnL', 'Return%']].fillna(0)
        trades['Profit'] = trades['PnL'].apply(lambda x: x if x > 0 else 0)
        trades['Loss'] = trades['PnL'].apply(lambda x: x if x < 0 else 0)
        trades = trades.set_index('Entry Index')
        trades.index.name = 'datetime'
        temp = trades.resample(period).sum()
        temp['Profit Factor'] = temp['Profit']/temp['Loss'].abs()
        win = trades['PnL'].apply(lambda x: 1 if x > 0 else 0)
        temp['Win Rate%'] = 100 * win.resample(period).sum()/win.resample(period).count()
        temp['Trades'] = win.resample(period).count()
        if period == 'M' or period == 'Q':
            temp.index = temp.index.strftime("%Y-%m")
        elif period == 'Y':
            temp.index = temp.index.strftime("%Y")
        temp = temp.applymap(lambda x: round(x, 2))[['Return%', 'Profit', 'Loss', 'Profit Factor', 'Win Rate%', 'Trades']]
        temp.columns = ['收益率%','毛利','毛損','獲利因子','勝率%','交易次數']
        return temp
    
    def get_trades_attribute(self, trades, attribute='Profit Factor'):
        pass

    def plot_time_filter_analysis(self, trades): # weekday和hourly收益分析
        trades['weekday'] = trades['Entry Index'].dt.day_name()
        trades['hour'] = trades['Entry Index'].dt.hour
        sns.boxplot(x='weekday', y='Return', data=trades, 
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday'])
        plt.title("Boxplot of Returns by Weekday")
        plt.show();
        sns.boxplot(x='hour', y='Return', data=trades, order=range(24))
        plt.title("Boxplot of Returns by Hours")
        plt.show();

    # trades analysis
    def show_trades_analysis(self,trades):
        self.plot_ret_dist(trades)
        self.plot_holding_period_dist(trades)
        self.plot_signal_response(trades)
        period_df = self.show_period_analysis(trades,period='Q')
        return period_df

    # optimize 
    def show_optimize_result(self, record_df, show=3):
        record_df = record_df.iloc[:show]
        for index, row in record_df.iterrows():
            txt = self.strategy_name + '-' + str(row['params'])
            params = row['params']
            side = row['side']
            _pf = self.strategy.strategy(side = side,params=params)
            record = _pf.trades.records_readable.set_index('Exit Index').sort_index()
            plt.title(txt)
            long = record[record['Direction'] == 'Long']['PnL'].cumsum()
            short = record[record['Direction'] == 'Short']['PnL'].cumsum()
            long.plot()
            short.plot()
            (100*(_pf.value/_pf.stats()['Start Value'])).plot()
            plt.show();
    
    # optimize 
    def _plot_3d_pivot_df(self, pivot_df, metrics):
        x = pivot_df.columns
        y = pivot_df.index
        X, Y = np.meshgrid(x, y)
        Z = pivot_df
        fig = plt.figure(figsize=(8, 8))
        plt.title(metrics)
        ax = fig.add_subplot(111, projection='3d',)
        ax.set_xlabel(pivot_df.columns.name)
        ax.set_ylabel(pivot_df.index.name)
        # ax.set_zlabel(metrics)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, # type: ignore
                               linewidth=0, antialiased=True)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        fig.colorbar(surf, shrink=0.5, aspect=3)
        plt.show();
    
    # optimize 
    def plot_3d_params(self, record_df, metrics):
        param_list = list(record_df['params'].iloc[0].keys())
        params_combination = list(itertools.combinations(param_list, 2))
        for p in param_list:
            record_df[p] = record_df['params'].apply(lambda x: x[p])
        for combination in params_combination:
            x = combination[0]
            y = combination[1]
            temp_df = record_df[[x, y, metrics]].groupby([x, y]).mean().reset_index()
            pivot_df = pd.pivot_table(temp_df, values=metrics, index=[x], columns=[y], aggfunc=np.mean)
            self._plot_3d_pivot_df(pivot_df, metrics)
    
    # optimize 
    def plot_2d_params(self, record_df, metrics=[]):
        params_list = list(record_df.params.iloc[0].keys())
        if len(metrics) == 0:
            metrics = [
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Omega Ratio',
                    'Calmar Ratio',
                    'Total Return [%]',
                    'Expectancy',
                    'Win Rate [%]',
                    'Max Drawdown [%]',
                    'Total Trades'
                    ]
        cols = params_list+metrics
        hip.Experiment.from_dataframe(record_df[cols]).display();

    def outsample_result_gen(self, params={}, side='both', tag='', axv_index=[]):
        pf = self.strategy.strategy(side = side,params=params)
        years = (self.df.index[-1] - self.df.index[0]).days/365
        stat = pf.stats()
        start_value = stat['Start Value']
        value_series = (pf.value/start_value)*100
        value_series = value_series - 100
        total_return = value_series.ffill()
        MDD_series = total_return.cummax()-total_return
        MDD = round(max(MDD_series), 2)
        return MDD, stat
