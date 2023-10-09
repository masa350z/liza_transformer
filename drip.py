# %%
import numpy as np
import pickle
import tensorflow as tf
from modules import modules
# %%
symbol = 'USDJPY'
k = 12
pr_k = 12
base_m = 15

data_path = 'datas/dataset/{}_k{}_pr{}_m{}.pickle'.format(
    symbol, k, pr_k, base_m)

with open(data_path, 'rb') as f:
    rawinput, future, cos_sim, binary = pickle.load(f)

tr_raw, vl_raw, te_raw = modules.split_data(rawinput)
tr_future, vl_future, te_future = modules.split_data(future)
tr_sim, vl_sim, te_sim = modules.split_data(cos_sim)
tr_y, vl_y, te_y = modules.split_data(binary)

tr_raw = tf.data.Dataset.from_tensor_slices(tr_raw)
vl_raw = tf.data.Dataset.from_tensor_slices(vl_raw)
te_raw = tf.data.Dataset.from_tensor_slices(te_raw)

tr_future = tf.data.Dataset.from_tensor_slices(tr_future)
vl_future = tf.data.Dataset.from_tensor_slices(vl_future)
te_future = tf.data.Dataset.from_tensor_slices(te_future)

tr_sim = tf.data.Dataset.from_tensor_slices(tr_sim)
vl_sim = tf.data.Dataset.from_tensor_slices(vl_sim)
te_sim = tf.data.Dataset.from_tensor_slices(te_sim)

tr_y = tf.data.Dataset.from_tensor_slices(tr_y)
vl_y = tf.data.Dataset.from_tensor_slices(vl_y)
te_y = tf.data.Dataset.from_tensor_slices(te_y)


tr_x = tf.data.Dataset.zip((tr_raw, tr_future, tr_sim))
vl_x = tf.data.Dataset.zip((vl_raw, vl_future, vl_sim))
te_x = tf.data.Dataset.zip((te_raw, te_future, te_sim))

train_x = tf.data.Dataset.zip((tr_x, tr_y))
valid_x = tf.data.Dataset.zip((vl_x, vl_y))
test_x = tf.data.Dataset.zip((te_x, te_y))
