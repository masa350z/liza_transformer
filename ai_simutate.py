# %%
from modules import models, modules
import pandas as pd
import numpy as np


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
y_mode = 'binary'

symbol = 'USDJPY'
hist_path = 'D:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
hist, timestamp = modules.ret_hist(symbol)

k = 12
pr_k = 12

base_m = 1
m_lis = [base_m, base_m*2, base_m*3]

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
pred = model.predict(data_x, batch_size=120000)
pred = pred[:, 0]

hist_data = data_x[:, -1, 0]
hist_data = hist_data[::base_m]
# %%
hist_data.shape
# %%
np.random.random(hist_data.shape)
# %%
rik = 0.01/100
son = 0.05/100

kane, asset, profit_list = simulate(hist_data, pr_k, rik, son, pred=None)
kane
# %%
pd.DataFrame(asset[int(len(asset)*0.8):]).plot()
# %%
pd.DataFrame(hist_data).plot()
# %%
pd.DataFrame(asset).plot()
# %%
win_ave = np.average(profit_list[profit_list > 0])
lose_ave = np.average(profit_list[profit_list < 0])

win_std = np.std(profit_list[profit_list > 0])
lose_std = np.std(profit_list[profit_list < 0])

# %%
win_ave, lose_ave, win_std, lose_std
# %%
pd.DataFrame(asset).plot()
