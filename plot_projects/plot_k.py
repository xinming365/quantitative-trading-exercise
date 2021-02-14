# *mpl_finance*

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as dates
import mpl_finance as mf

plt.rcParams['figure.figsize'] = (18, 6)s
plt.rcParamsp['font.family']=['SimHei']
df_stock = pd.read_excel('./data/something.xlsx')
df_stock.set_index('日期', inplace=True)

new_df = df_stock.loc[:, ['开盘', '最高', '最低', '收盘']]
sns.lineplot(data=new_df.iloc[:150])

zip_data = zip(dates.date2num(new_df, new_df.index.to_pydatetime()))
ax = plt.gca()
mf.candlestick_ohlc(ax, zip_data, width=1, colorup='r', colordown='g')
ax.axis_date()
plt.xticks(rotation=45)