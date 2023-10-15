# %%
from modules import models, modules
from tqdm import tqdm
import numpy as np
import pandas as pd
# %%
y_mode = 'binary'

symbol = 'EURUSD'
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
pred = model.predict(data_x)
pred = pred[:, 0]
# %%
data_x = data_x[:, -1, 0]
# %%
rik = 0.005/100
son = 0.1/100

kane = 0
count = 0
count_list = []
asset = []
position = 0
pos = 0
for h, i in tqdm(enumerate(data_x)):
    if position == 0:
        position = i
        # if pred[h] > 0.5:
        if np.random.random() > 0.5:
            pos = 1
        else:
            pos = -1
        count = 0
    else:
        if (i - position)*pos > rik:
            kane += (i - position)*pos
            position = 0
            count_list.append(count)
        elif (i - position)*pos < -son:
            kane += (i - position)*pos
            position = 0
            count_list.append(count)
        count += 1
    asset.append(kane)

# %%
kane
# %%
pd.DataFrame(asset).plot()
# %%
tr_len = int(len(asset)*0.6)
vl_len = int(len(asset)*0.2)
# %%
tr_asset = asset[:tr_len]
vl_asset = asset[tr_len:tr_len+vl_len]
te_asset = asset[tr_len+vl_len:]
# %%
tr_asset[-1] - tr_asset[0]
# %%
vl_asset[-1] - vl_asset[0]
# %%
te_asset[-1] - te_asset[0]
# %%
pd.DataFrame(tr_asset).plot()
# %%
pd.DataFrame(vl_asset).plot()
# %%
pd.DataFrame(te_asset).plot()
# %%
np.average(np.array(count_list))

# %%
