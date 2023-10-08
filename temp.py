# %%

import pandas as pd
import numpy as np
from modules import modules
# %%
symbol = 'USDJPY'

hist, timestamp = modules.ret_hist(symbol)
# %%
k = 12
pr_k = 12

base_m = 15
m_lis = [base_m, base_m*2, base_m*3]

data_x, data_y = modules.ret_data_xy(
    hist, m_lis, base_m, k, pr_k, y_mode='binary')

# %%
data_x

# %%
train_y, valid_y, test_y = modules.split_data(data_y)
train_x, valid_x, test_x = modules.split_data(data_x)
# %%
np.sum(valid_y[:, 0])/len(valid_y)
# %%
np.sum(test_y[:, 0])/len(test_y)
# %%
np.sum(valid_y[:, 0]) - np.sum(valid_y[:, 1])
# %%
len(valid_y)
# %%
np.sum(test_y[:, 0]) - np.sum(test_y[:, 1])
# %%
len(test_y)
# %%
inp_y = valid_y
inp_x = valid_x
# %%


def half_updn(inp_x, inp_y):
    up_y = inp_y[inp_y[:, 0] == 1]
    dn_y = inp_y[inp_y[:, 1] == 1]

    up_x = inp_x[inp_y[:, 0] == 1]
    dn_x = inp_x[inp_y[:, 1] == 1]

    diff = len(up_y) - len(dn_y)

    if diff > 0:
        index = np.arange(len(up_y))
        np.random.shuffle(index)

        up_y = up_y[index][:-diff]
        up_x = up_x[index][:-diff]

        y_ = np.concatenate([up_y, dn_y])
        x_ = np.concatenate([up_x, dn_x])
    elif diff < 0:
        index = np.arange(len(dn_y))
        np.random.shuffle(index)

        dn_y = dn_y[index][:-diff]
        dn_x = dn_x[index][:-diff]

        y_ = np.concatenate([up_y, dn_y])
        x_ = np.concatenate([up_x, dn_x])

    else:
        y_ = inp_y
        x_ = inp_x

    return x_, y_


updn = half_updn(inp_x, inp_y)
# %%
updn[1]
# %%
