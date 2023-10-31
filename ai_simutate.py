# %%
from modules import models, modules
import pandas as pd
import numpy as np


def simulate(hist_data, rik, son, spread=0, pred=None):
    kane = 0
    asset = []
    profit_list = []
    get_price = 0
    position = 0
    for h, i in enumerate(hist_data):
        if get_price != 0:
            if (i - get_price)*position > rik:
                profit = (i - get_price)*position - spread
                kane += profit
                profit_list.append(profit)
                get_price = 0

            elif (i - get_price)*position < -son:
                profit = (i - get_price)*position - spread
                kane += profit
                profit_list.append(profit)
                get_price = 0

        if get_price == 0:
            get_price = i
            if pred is None:
                up_ = np.random.random() > 0.5
            else:
                up_ = pred[h] > 0.5

            position = 1 if up_ else -1

        asset.append(kane)

    return kane, np.array(asset), np.array(profit_list)


# %%
y_mode = 'binary'

symbol = 'USDJPY'
hist_path = 'D:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
hist, timestamp = modules.ret_hist(symbol)

k = 3
pr_k = 3

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
rik = 0.005
son = 0.1

kane, asset, profit_list = simulate(hist_data, rik, son, pred=pred)
kane
# %%
pd.DataFrame(asset).plot()
# %%
a = 7000000
b = a+60*24*1
pd.DataFrame(asset[a:b]).plot()
# %%
win_ave = np.average(profit_list[profit_list > 0])
lose_ave = np.average(profit_list[profit_list < 0])

win_std = np.std(profit_list[profit_list > 0])
lose_std = np.std(profit_list[profit_list < 0])

# %%
win_ave, lose_ave, win_std, lose_std

# %%
