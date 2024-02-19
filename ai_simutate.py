# %%
from modules import models, modules
import pandas as pd
import numpy as np
import tensorflow as tf


def simulate(hist_data, pr_k, rik, son, pred=None, spread=0):
    kane = 0
    asset = []
    profit_list = []
    get_price = 0
    position = 0
    position_count = 0

    if pred is None:
        pred = np.random.random(hist_data.shape)

    for h, i in enumerate(hist_data):
        position_count = position_count - 1 if position_count > 0 else 0
        profit = (i - get_price)*position - spread
        if get_price != 0:
            if (i - get_price)*position > rik*get_price:
                kane += profit
                profit_list.append(profit/get_price)
                get_price = 0

            elif (i - get_price)*position < -son*get_price or \
                    position_count == 0:
                kane += profit
                profit_list.append(profit/get_price)
                get_price = 0

        elif get_price == 0:
            get_price = i
            up_ = pred[h] > 0.5

            position = 1 if up_ else -1
            position_count = pr_k

        asset.append(kane)

    return kane, np.array(asset), np.array(profit_list)


# %%
symbol = 'USDJPY'
hist, timestamp = modules.ret_hist(symbol)

k = 12
pr_k = 12

hist_2d = modules.hist_conv2d(hist, k+pr_k)
# %%
model = models.LizaAffine()
# %%
data_x = hist_2d[:, :k]
data_y = hist_2d[:, k:]
# %%
data_x - tf.reduce_min(data_x, axis=1, keepdims=True)
