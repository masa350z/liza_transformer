# %%

import pickle
import numpy as np
import tensorflow as tf
from modules import modules
from tqdm import tqdm
import sys


def ret_datax_past_future(data_x_past, data_x_future):
    mx = np.max(data_x_past, axis=1, keepdims=True)
    mn = np.min(data_x_past, axis=1, keepdims=True)

    data_x_past = (data_x_past - mn)/(mx - mn)
    data_x_future = (data_x_future - mn)/(mx - mn)

    data_x_past = np.nan_to_num(data_x_past).astype('float32')
    data_x_future = np.nan_to_num(data_x_future).astype('float32')

    return data_x_past, data_x_future


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


def ret_future_past_dataset(temp_x_past, temp_x_future, data_array_past, data_array_future, num_similar=10):
    cos_sim_lis = []
    for j in range(temp_x_past.shape[2]):
        cos_sim = calc_cos_sim(temp_x_past[:, :, j], data_array_past[:, :, j])
        cos_sim_lis.append(cos_sim)
    cos_sim = np.stack(cos_sim_lis, axis=2)

    cos_sim = np.prod(cos_sim, axis=2)

    sorted = tf.argsort(cos_sim, axis=1).numpy()
    sorted = sorted[:, ::-1][:, :num_similar]

    similar_future = []
    cos_sim_lis = []
    for j, s in enumerate(sorted):
        similar_future.append(data_array_future[s][:, :, 0])
        cos_sim_lis.append(cos_sim[j][s])

    similar_future = np.array(similar_future)
    cos_sim_lis = np.array(cos_sim_lis)

    future_binary = \
        temp_x_future[:, :, 0][:, -1] - temp_x_future[:, :, 0][:, 0]
    future_binary = future_binary > 0
    future_binary = np.stack([future_binary, (future_binary-1)*-1], axis=1)

    return cos_sim_lis, similar_future, future_binary


# %%
symbol = sys.argv[1]
k = int(sys.argv[2])
pr_k = int(sys.argv[3])
base_m = int(sys.argv[4])

m_lis = [base_m, base_m*2, base_m*3]

hist, timestamp = modules.ret_hist(symbol)

data_x, data_y = modules.ret_data_xy(
    hist, m_lis, base_m, k+pr_k, pr_k,
    norm=False, y_mode='binary')
# %%
data_x_past = data_x[:, :k]
data_x_future = data_x[:, k-1:]

data_x_past, data_x_future = ret_datax_past_future(data_x_past, data_x_future)
# %%
data_x_past = data_x_past.astype('float16')
data_x_past = tf.constant(data_x_past)
# %%
num_similar = 10

data_length = len(data_x_future)
batch_size = int(500*base_m/15)
num_batches = int(data_length/batch_size)

cos_sim_lis, future_lis, binary_lis, rawinput_lis = [], [], [], []

for h in tqdm(range(num_batches)):
    i = batch_size*(h+1)
    if h+1 == num_batches:
        temp_x_past = data_x_past[i:]
        temp_x_future = data_x_future[i:]
    else:
        temp_x_past = data_x_past[i:i+batch_size]
        temp_x_future = data_x_future[i:i+batch_size]

    data_array_past = data_x_past[:i-pr_k]
    data_array_future = data_x_future[:i-pr_k]

    cos_sim, similar_future, future_binary = ret_future_past_dataset(
        temp_x_past, temp_x_future, data_array_past, data_array_future)

    cos_sim_lis.append(cos_sim)
    future_lis.append(similar_future)
    binary_lis.append(future_binary)
    rawinput_lis.append(temp_x_future)
# %%
cos_sim_lis = np.concatenate(cos_sim_lis)
future_lis = np.concatenate(future_lis)
binary_lis = np.concatenate(binary_lis)
rawinput_lis = np.concatenate(rawinput_lis)
# %%
dataset = [rawinput_lis, future_lis, cos_sim_lis, binary_lis]
# %%
data_path = 'datas/dataset/k{}_pr{}_m{}.pickle'.format(k, pr_k, base_m)
with open(data_path, 'wb') as f:
    pickle.dump(dataset, f)

# %%
