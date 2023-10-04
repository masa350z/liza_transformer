# %%
from modules import modules
from modules import models
import pandas as pd
import numpy as np


def ret_updn(pred, threshold):
    up = pred[:, 0] > threshold
    dn = pred[:, 1] > threshold

    updn = 1*up + -1*dn

    return updn


def ret_future_updn_ratio(pr_k, updn, future):
    future_updn = np.expand_dims(updn, 1)*future

    ratio_lis = []
    for i in range(pr_k):
        temp = future_updn[:, i+1]/future_updn[:, 0] - 1
        ratio_lis.append(temp)

    ratio_lis = np.stack(ratio_lis).T

    return ratio_lis


def simulate_rikson(updn, future_price, updn_ratio, spread_ratio, rik, son):
    rik_son_status = np.zeros(len(updn_ratio))

    for i in range(updn_ratio.shape[1]):
        r = updn_ratio[:, i] > rik
        s = updn_ratio[:, i] < -son

        rs = r+s

        rik_son_status += updn_ratio[:, i]*rs*(rik_son_status == 0)

    rik_son_array = (updn_ratio - np.expand_dims(rik_son_status, 1)) == 0

    rik_son_price = np.sum(future_price[:, 1:]*rik_son_array, axis=1)
    rik_son_price = np.where(
        rik_son_price == 0, future_price[:, -1], rik_son_price)

    asset = updn * \
        (rik_son_price - future_price[:, 0] - future_price[:, 0]*spread_ratio)

    return asset


# %%
y_mode = 'binary'

symbol = 'BTCJPY'
hist_path = 'D:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
df = pd.read_csv(hist_path)
hist = np.array(df['price'], dtype='float32')

k = 12
pr_k = 12

base_m = 10
m_lis = [base_m, base_m*2, base_m*3]

weight_name = modules.ret_weight_name(symbol=symbol,
                                      k=k,
                                      pr_k=pr_k,
                                      m_lis=m_lis,
                                      y_mode=y_mode)

model = models.LizaTransformer(k, out_dim=2)
model.load_weights(weight_name + '/best_weights')
# %%
dataset = modules.LizaDataSet(hist, m_lis, k, pr_k, batch_size=120)

pred_train = model.predict(dataset.train_dataset)
pred_valid = model.predict(dataset.valid_dataset)
pred_test = model.predict(dataset.test_dataset)
# %%
spread_ratio = 0.05/100
threshold = 0.5

train_updn = ret_updn(pred_train, threshold)
valid_updn = ret_updn(pred_valid, threshold)
test_updn = ret_updn(pred_test, threshold)

future_y = modules.ret_future_y(hist, m_lis, base_m, k, pr_k)
train_future, valid_future, test_future = modules.split_data(future_y)

train_updn_ratio = ret_future_updn_ratio(pr_k, train_updn, train_future)
valid_updn_ratio = ret_future_updn_ratio(pr_k, valid_updn, valid_future)
test_updn_ratio = ret_future_updn_ratio(pr_k, test_updn, test_future)

# %%
rik, son = 0.01, 0.01

train_asset = simulate_rikson(train_updn, train_future,
                              train_updn_ratio, spread_ratio, rik, son)
valid_asset = simulate_rikson(valid_updn, valid_future,
                              valid_updn_ratio, spread_ratio, rik, son)
test_asset = simulate_rikson(test_updn, test_future,
                             test_updn_ratio, spread_ratio, rik, son)

print(np.sum(train_asset))
print(np.sum(valid_asset))
print(np.sum(test_asset))
# %%
