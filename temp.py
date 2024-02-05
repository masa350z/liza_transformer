# %%
from glob import glob
import pandas as pd
import numpy as np
# %%
price_diff_list = glob('datas/price_diff_list/*.csv')
# %%
df = pd.DataFrame()
for i in price_diff_list:
    df = pd.concat([df, pd.read_csv(i)])

# %%
df_eurusd = df[df['0'] == 'EURUSD'].reset_index(drop=True)
df_usdjpy = df[df['0'] == 'USDJPY'].reset_index(drop=True)
# %%
df_eurusd[df_eurusd['1'] == 'buy']['3'].hist(bins=100)
# %%
df_eurusd[df_eurusd['1'] == 'sell']['3'].hist(bins=100)
# %%
df_eurusd['3'].plot()
# %%
df_usdjpy['3'].plot()
# %%
df_eurusd['3'].hist(bins=30)
# %%
df_usdjpy['3'].hist(bins=30)
# %%
df.columns = ['pair', 'side', 'time_diff', 'price_diff']
# %%
df.to_csv('datas/price_diff_list_combined.csv', index=False)
# %%
df_usdjpy['3'].mean()
# %%
df_eurusd['3'].mean()
