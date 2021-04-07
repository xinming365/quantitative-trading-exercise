# *mpl_finance*

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as dates
import mplfinance as mpf

filename = 'F:\GitHub\My_data_set\I/20131018.csv'



plt.rcParams['font.family']=['SimHei']
# df_stock = pd.read_csv(filename)

"""
df_stock.index:
 RangeIndex(start=0, stop=15831, step=1)

df_stock.columns:
['instrument_id', 'trading_day', 'action_day', 'update_time',
'last_price', 'volume', 'open_interest', 'turnover', 'ask_price1',
 'ask_volume1', 'bid_price1', 'bid_volume1']
 
 与之相应的中文名称：
['期货合约ID', '交易日', '业务日期', '更新时间',
 '最新价格', '交易量', '持仓', '交易额', '买一价',
 '买一量', '卖一价', '卖一量']
"""

def convert_time(file_name, time_range):
    # 500ms per tick range.
    # return a class of pandas.core.frame.DataFrame
    df_stock = pd.read_csv(file_name)
    # convert the format of the date.
    df_stock['update_time'] = pd.to_datetime(df_stock['update_time'])
    n = df_stock.shape[0]
    range_dict = dict(one_s=2, thirty_s=2*30, one=2*60, five=120*5, thirty=120*30)
    if time_range in range_dict.keys():
        # tr: time range.
        tr = range_dict[time_range]
    else:
        print('Wrong time range. Only support one/five/thirty.')
        raise KeyError
    index = list(range(0, n, tr))
    return df_stock.iloc[index,:]


# df_stock.set_index('日期', inplace=True)
#
# # selecting on a multi-axis by label
# new_df = df_stock.loc[:, ['开盘', '最高', '最低', '收盘']]
# sns.lineplot(data=new_df.iloc[:150])
#
# zip_data = zip(dates.date2num(new_df, new_df.index.to_pydatetime()))
# ax = plt.gca()
# mpf.candlestick_ohlc(ax, zip_data, width=1, colorup='r', colordown='g')
# ax.axis_date()
# plt.xticks(rotation=45)

if __name__=='__main__':
    # convert_time(filename, 'sdf')
    df_stock = convert_time(filename, 'thirty_s')
    new_df = df_stock.loc[:, ['update_time', 'ask_price1', 'bid_price1']]
    print( new_df.shape)
    # sns.lineplot(x=new_df.index.to_list(), y='ask_price1', data=new_df.iloc[:150])
    sns.lineplot(x='update_time', y='ask_price1', data=new_df[:100])
    plt.show()