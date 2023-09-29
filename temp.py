# %%
import liza_module
import numpy as np
import pandas as pd
import tensorflow as tf


# %%
k = 90
pr_k = 30

btc_hist_path = 'D:/documents/hist_data/symbol/BTCJPY/1m.csv'
df = pd.read_csv(btc_hist_path)
hist = np.array(df['price'], dtype='int32')
hist = np.arange(len(df['price']), dtype='int32')


m_lis = [30, 60, 90]
base_m = m_lis[0]

data_xy = liza_module.ret_data_xy(hist, m_lis, m_lis[0], k, pr_k)
# %%
data_xy[0]

# %%
