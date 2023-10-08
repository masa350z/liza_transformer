# %%

import pandas as pd
import numpy as np
import tensorflow as tf
from modules import modules


def calc_cos_sim(x, y):
    """
    x: 入力側
    y: 比較側（過去データ側）
    """

    dot = tf.tensordot(x, y, axes=[-1, -1])

    normx = tf.norm(x, axis=1)
    normy = tf.norm(y, axis=1)
    norm = tf.expand_dims(normx, 1)*tf.expand_dims(normy, 0)

    cos_sim = dot/norm
    cos_sim = cos_sim.numpy()

    return cos_sim


# %%
symbol = 'USDJPY'

hist, timestamp = modules.ret_hist(symbol)
# %%
k = 12
pr_k = 12

base_m = 15
m_lis = [base_m, base_m*2, base_m*3]

data_x, data_y = modules.ret_data_xy(
    hist, m_lis, base_m, k+pr_k, pr_k,
    norm=False, y_mode='binary')
# %%
data_x_past = data_x[:, :k]
data_x_future = data_x[:, k-1:]
# %%
mx = np.max(data_x_past, axis=1, keepdims=True)
mn = np.min(data_x_past, axis=1, keepdims=True)

data_x_past = (data_x_past - mn)/(mx - mn)
data_x_future = (data_x_future - mn)/(mx - mn)

data_x_past = np.nan_to_num(data_x_past).astype('float32')
data_x_future = np.nan_to_num(data_x_future).astype('float32')
# %%
# data_x_past = modules.normalize(data_x_past)
# data_x_future = modules.normalize(data_x_future)
# %%
data_x_past = data_x_past.astype('float16')

data_x_past = tf.constant(data_x_past)
# %%
batch_size = 1200
i = 50000
# %%
data_x_past[i:i+batch_size].shape
# %%
temp_x_past = data_x_past[i:i+batch_size]
temp_x_future = data_x_future[i:i+batch_size]
# %%
data_array_past = data_x_past[:i-pr_k]
data_array_future = data_x_future[:i-pr_k]

cos_sim_lis = []
for j in range(temp_x_past.shape[2]):
    cos_sim = calc_cos_sim(temp_x_past[:, :, j], data_array_past[:, :, j])
    cos_sim_lis.append(cos_sim)
cos_sim_lis = np.stack(cos_sim_lis, axis=2)

cos_sim_lis = np.prod(cos_sim_lis, axis=2)


# %%
sorted = tf.argsort(cos_sim_lis, axis=1,).numpy()
sorted = sorted[:, ::-1][:, :100]

sorted_lis = []
for s in sorted:
    sorted_lis.append(np.take(data_array_future, s, axis=0))

sorted_lis = np.array(sorted_lis)

# %%
future_binary = temp_x_future[:, :, 0][:, -1] - temp_x_future[:, :, 0][:, 0]
future_binary = future_binary > 0
future_binary = np.stack([future_binary, (future_binary-1)*-1], axis=1)
# %%
future_binary
# %%
sorted_lis
# %%
temp_x_past.shape
