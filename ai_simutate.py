# %%
from modules import models, modules
import pandas as pd
import numpy as np


def simulate(pred, hist_data, rik, son, spread=0):
    kane = 0
    asset = []
    position = 0
    pos = 0
    for h, i in enumerate(hist_data):
        if position != 0:
            if (i - position)*pos > rik:
                # kane += (i - position)*pos
                kane += rik
                position = 0

            elif (i - position)*pos < -son:
                # kane += (i - position)*pos
                kane += -son
                position = 0

        if position == 0:
            if pred[h] > 0.5:
                position = i + spread
                pos = 1
            else:
                position = i - spread
                pos = -1

        asset.append(kane)

    return kane, asset


def simulate_random(hist_data, rik, son, spread=0):
    kane = 0
    asset = []
    position = 0
    pos = 0
    for h, i in enumerate(hist_data):
        if position != 0:
            if (i - position)*pos > rik:
                # kane += (i - position)*pos
                kane += rik
                position = 0

            elif (i - position)*pos < -son:
                # kane += (i - position)*pos
                kane += -son
                position = 0

        if position == 0:
            if np.random.random() > 0.5:
                position = i + spread
                pos = 1
            else:
                position = i - spread
                pos = -1

        asset.append(kane)

    return kane, asset


# %%
y_mode = 'binary'

symbol = 'USDJPY'
hist_path = 'D:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
hist, timestamp = modules.ret_hist(symbol)

k = 3
pr_k = 3

base_m = 1
m_lis = [base_m, base_m*2, base_m*3]
# %%
weight_name = modules.ret_weight_name(symbol=symbol,
                                      k=k,
                                      pr_k=pr_k,
                                      m_lis=m_lis,
                                      y_mode=y_mode)

model = models.LizaTransformer(k, out_dim=2)
model.load_weights(weight_name + '/best_weights')
# %%
data_x, data_y = modules.ret_data_xy(
    hist, m_lis, base_m, k, pr_k, y_mode=y_mode)
pred = model.predict(data_x, batch_size=12000)
pred = pred[:, 0]

hist_data = data_x[:, -1, 0]
hist_data = hist_data[::base_m]
# %%
rik = 0.05/1
son = 0.005/1

# kane, asset = simulate(pred, hist_data, rik, son)
kane, asset = simulate_random(hist_data, rik, son, spread=0./100)
kane
# %%
pd.DataFrame(asset).plot()
# %%
pd.DataFrame(hist_data).plot()
# %%
a = 6000000
b = a+60*24*25
pd.DataFrame(asset[a:b]).plot()
# %%
len(asset)
# %%
data_x[:, -1, 0][::base_m].shape
# %%
