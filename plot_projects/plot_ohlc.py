import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(18,6)
sns.set_style('darkgrid', {'font.sans-serif':['SimHei', 'Arial']})
filename = 'F:\GitHub\My_data_set\I/20131018.csv'
df_stock = pd.read_csv(filename)

new_df = df_stock.loc[:, ['开盘', '最高', '最低', '收盘']]
sns.lineplot(data=new_df.iloc[:150])