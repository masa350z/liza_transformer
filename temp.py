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


m_lis = [5, 10]
base_m = m_lis[0]

batch_size = 120*50
lizadataset = trainer.LizaDataSet(hist, m_lis, k, pr_k, batch_size, base_m)

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
std = tf.math.reduce_std(x, axis=1)
# %%
x[:, -1]
# %%
roll01 = int(x.shape[1]/5)
roll02 = int(x.shape[1]/2)
# %%
diff01 = (x - tf.roll(x, roll01, axis=1))[:, roll01:]
diff02 = (x - tf.roll(x, roll02, axis=1))[:, roll02:]
