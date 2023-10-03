# %%
import tensorflow as tf
import pandas as pd
import numpy as np
import liza_models
import liza_module
import trainer
# %%
symbol = 'BTCJPY'

k = 72
pr_k = 12

hist_path = 'D:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
df = pd.read_csv(hist_path)
hist = np.array(df['price'], dtype='float32')
# hist = np.array(np.arange(len(df['price'])), dtype='int32')


m_lis = [15, 90]
base_m = m_lis[0]

batch_size = 120*50
# lizadataset = trainer.LizaDataSet(hist, m_lis, k, pr_k, batch_size, base_m)
# %%
y_diff, y_one_hot, y_updn = liza_module.ret_data_y(
    hist, m_lis, base_m, k, pr_k)
# %%
a = int(len(y_one_hot)*0.2)
np.sum(y_one_hot[:, 0][-a:])/len(y_one_hot[-a:])
# %%
for x, y in lizadataset.train_dataset:
    break
# %%
x
# %%
mx = tf.reduce_max(x, axis=1, keepdims=True)
mn = tf.reduce_min(x, axis=1, keepdims=True)

normed = (x-mn)/(mx-mn)
normed = tf.where(tf.math.is_finite(normed), normed, 0.0)
normed
# %%
mx = tf.reduce_max(tf.reduce_max(x, axis=1, keepdims=True),
                   axis=2, keepdims=True)
mn = tf.reduce_min(tf.reduce_min(x, axis=1, keepdims=True),
                   axis=2, keepdims=True)
# %%
