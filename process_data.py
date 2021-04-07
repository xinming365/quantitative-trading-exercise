# *mpl_finance*

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.dates as dates
import mplfinance as mpf

filename = 'F:\GitHub\My_data_set\I/20131018.csv'
test_ratio = 0.2
filepath = 'F:\GitHub\My_data_set\I'
# df_stock = pd.read_csv(filename)
plt.rcParams['font.family']=['Times New Roman']
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
    # df_stock['update_time'] = pd.to_datetime(df_stock['update_time'])
    # df_stock = df_stock.set_index('update_time')
    n = df_stock.shape[0]
    range_dict = dict(one_s=2, thirty_s=2 * 30, one=2 * 60, five=120 * 5, thirty=120 * 30)
    if time_range in range_dict.keys():
        # tr: time range.
        tr = range_dict[time_range]
    else:
        print('Wrong time range. Only support one/five/thirty.')
        raise KeyError
    index = list(range(0, n, tr))
    return df_stock.iloc[index, :]


def split_data(test_ratio=0.2):
    fn_list = os.listdir(filepath)
    nums = len(fn_list)
    print(nums)
    train_nums = int(nums - np.ceil(nums * test_ratio))
    train_files = fn_list[:train_nums]
    test_files = fn_list[train_nums:]
    return train_files, test_files


def read_all_data():
    fn_list = os.listdir(filepath)
    nums = len(fn_list)
    df_stock = []
    for f in fn_list:
        filename = os.path.join(filepath, f)
        df_stock_one_day = convert_time(filename, 'one')
        # df_stock_one_day = df_stock_one_day.loc[:, ['update_time', 'ask_price1', 'bid_price1']]
        df_stock_one_day = df_stock_one_day.loc[:, ['last_price']]
        df_stock.append(df_stock_one_day)
    df_all_data = pd.concat(df_stock, axis=0)
    return df_all_data


def save_train_test_data(save_all_data=True):
    def save(fn_list, data_name):
        saved_path = 'F:\GitHub\My_data_set\data'
        df_stock = []
        for f in fn_list:
            filename = os.path.join(filepath, f)
            df_stock_one_day = convert_time(filename, 'one')
            df_stock_one_day = df_stock_one_day.loc[:,
                               ['last_price', 'volume', 'open_interest', 'turnover', 'ask_price1', 'ask_volume1',
                                'bid_price1', 'bid_volume1']]
            df_stock.append(df_stock_one_day)
        df_all_data = pd.concat(df_stock, axis=0)
        print('Saving to {}'.format(data_name))
        df_all_data.to_csv(path_or_buf=os.path.join(saved_path, data_name))
        print('saved!')

    if save_all_data:
        fn_list = os.listdir(filepath)
        save(fn_list, 'dataset.csv')
    else:
        train_files, test_files = split_data(test_ratio=0.2)
        save(train_files, 'train.csv')
        save(test_files, 'test.csv')


def save_labels():
    saved_path = 'F:\GitHub\My_data_set\data'
    filename = os.path.join(saved_path, 'dataset.csv')
    dataset = pd.read_csv(filepath_or_buffer=filename)
    pass


def plot_last_price():
    fn_list = os.listdir(filepath)
    time_range = 'one'
    df_stock = []
    for f in fn_list:
        filename = os.path.join(filepath, f)
        df_stock_one_day = convert_time(filename, time_range)
        # df_stock_one_day = df_stock_one_day.loc[:, ['update_time', 'ask_price1', 'bid_price1']]
        df_stock_one_day = df_stock_one_day.loc[:, ['last_price']]
        df_stock.append(df_stock_one_day)
    df = pd.concat(df_stock, axis=0)
    new_df = df['last_price'].to_numpy()
    total_nums = len(new_df)
    x = np.arange(1, total_nums + 1)
    train_nums = int(total_nums - np.ceil(total_nums * test_ratio))
    plt.plot(x[:train_nums], new_df[:train_nums], 'b', label='Training data')
    plt.plot(x[train_nums-1:], new_df[train_nums-1:], 'r', label='Testing data')
    plt.ylabel('Last Price')
    plt.xlabel('Numbers of Data')
    plt.xlim(left=0)
    plt.legend()
    pic_name = time_range+'_m_' + str(train_nums) +'_' + str(total_nums-train_nums) + '.png'
    out_dir = './pictures'
    plt.savefig(os.path.join(out_dir, pic_name), dpi=500)
    # plt.xticks([])
    plt.show()


def plot_one_day_k():
    df_stock = convert_time(filename, 'one')
    new_df = df_stock['last_price'].to_numpy()
    x= np.arange(1,len(new_df)+1)
    nums = 50
    print(new_df.shape)
    # sns.lineplot(x=new_df.index.to_list(), y='ask_price1', data=new_df.iloc[:150])
    # sns.lineplot(x='update_time', y='ask_price1',  size='bid_price1', data=new_df)

    # new_df.plot(x='update_time', y='ask_price1', style='g')
    # new_df[50:].plot(x='update_time', y='ask_price1', style='r')
    plt.plot(x[:nums], new_df[:nums], 'k', label='Training data')
    plt.plot(x[nums-1:], new_df[nums-1:], 'r', label='Testing data')
    plt.ylabel('Last Price')
    plt.xlabel('Numbers of Data')
    plt.xlim(left=0)
    # plt.savefig('./one_minute.png', dpi=500)
    # plt.xticks([])
    plt.show()



def plot_k():
    train_files, test_files = split_data(test_ratio=0.2)
    nums_train, nums_test = len(train_files), len(test_files)
    df_stock = []
    for f in train_files:
        filename = os.path.join(filepath, f)
        df_stock_one_day = convert_time(filename, 'one')
        df_stock_one_day = df_stock_one_day.loc[:, ['update_time', 'ask_price1', 'bid_price1']]
        df_stock.append(df_stock_one_day)
    df_tran = pd.concat(df_stock, axis=0)

    df_total = df_tran.copy(deep=True)
    df_stock = []
    for f in test_files:
        filename = os.path.join(filepath, f)
        df_stock_one_day = convert_time(filename, 'one')
        df_stock_one_day = df_stock_one_day.loc[:, ['update_time', 'ask_price1', 'bid_price1']]
        df_stock.append(df_stock_one_day)
    df_test = pd.concat(df_stock, axis=0)
    df_total = pd.concat([df_total, df_test], axis=0)
    sns.lineplot(x='update_time', y='ask_price1', data=df_total[:nums_train])
    plt.show()



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

if __name__ == '__main__':
    # convert_time(filename, 'sdf')
    # plot_k()
    # plot_one_day_k()
    # plot_last_price()
    save_train_test_data(save_all_data=False)