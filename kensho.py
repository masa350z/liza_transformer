# %%
import pandas as pd
import numpy as np
import liza_models
import liza_module


def ret_updn(pred, threshold):
    up = pred[:, 0] > threshold
    dn = pred[:, 1] > threshold

    return up, dn


def ret_kane_ratio(pred, y, threshold, sp, sp_mode='ratio'):
    up, dn = ret_updn(pred, threshold)

    kane_up = y*up
    kane_dn = -y*dn

    kane = kane_up + kane_dn

    if sp_mode == 'ratio':
        kane_sum = np.sum(kane - abs(kane)*sp)
    else:
        kane_sum = np.sum(kane - sp)

    ratio = np.sum(kane > 0)/np.sum(kane != 0)

    return kane_sum, ratio


# %%
symbol = 'BTCJPY'

k = 12
pr_k = 12

hist_path = 'D:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
df = pd.read_csv(hist_path)
hist = np.array(df['price'], dtype='float32')


base_m = 5
m_lis = [base_m, base_m*2, base_m*3]

st = ''
for i in m_lis:
    st += str(i) + '_'
st = st[:-1]

weight_name = 'weights/{}/k{}_prk{}_basem{}_mlis{}'.format(
    symbol, k, pr_k, base_m, st)

# %%
model = liza_models.LizaTransformer(k)
model.load_weights(weight_name + '/best_weights')
# %%
data_x, data_y = liza_module.ret_data_xy(
    hist, m_lis, base_m, k, pr_k, y_mode='diff')

train_x, valid_x, test_x = liza_module.split_data(data_x)
train_y, valid_y, test_y = liza_module.split_data(data_y)
# %%
pred_train = model.predict(train_x)
pred_valid = model.predict(valid_x)
pred_test = model.predict(test_x)
# %%
sp = 0.05/100
# sp = 0.5/100*0
threshold = 0.5
sp_mode = 'ratio'

kane, ratio = ret_kane_ratio(
    pred_train, train_y, threshold, sp, sp_mode=sp_mode)
print(kane)
print(ratio)

kane, ratio = ret_kane_ratio(
    pred_valid, valid_y, threshold, sp, sp_mode=sp_mode)
print(kane)
print(ratio)

kane, ratio = ret_kane_ratio(
    pred_test, test_y, threshold, sp, sp_mode=sp_mode)
print(kane)
print(ratio)
# %%
