# %%
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os


def ret_hist(symbol):
    hist_path = 'E:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
    df = pd.read_csv(hist_path)
    hist = np.array(df['price'], dtype='float32')
    timestamp = np.array(df['timestamp'], dtype='int32')

    return hist, timestamp


def ret_kane_asset(hist, rik, son):
    kane = 0
    asset = []
    position = 0
    pos = 0
    for i in tqdm(hist):
        if position == 0:
            position = i
            if np.random.random() > 0.5:
                pos = 1
            else:
                pos = -1
        else:
            if (i - position)*pos > rik*position:
                kane += (i - position)*pos
                position = 0
            elif (i - position)*pos < -son*position:
                kane += (i - position)*pos
                position = 0
        asset.append(kane)

    return kane, asset


# %%
symbol = sys.argv[1]
m = int(sys.argv[2])
rik = round(float(sys.argv[3]), 3+2)
son = round(float(sys.argv[4]), 2+2)
# %%
"""
symbol = 'USDJPY'
m = 1
rik = 0.005
son = 0.1
"""

base_dir = 'datas/simulation/{}'.format(symbol)
os.makedirs(base_dir, exist_ok=True)
save_dir = base_dir + '/m{}_rik{}_son{}.npy'.format(m, rik, son)

if os.path.exists(save_dir.format(m, rik, son)):
    print('already exists')
    # asset_std = np.load(save_dir)
else:

    hist, timestamp = ret_hist(symbol)
    hist = hist[::m]

    asset_lis = []
    for _ in range(10):
        kane, asset = ret_kane_asset(hist, rik, son)

        asset_lis.append(asset)

    asset = np.stack(asset_lis, axis=1)
    std = np.std(asset, axis=1)
    asset = np.average(asset, axis=1)

    asset_std = np.stack([asset + std*2, asset, asset - std*2], axis=1)

    np.save(save_dir, asset_std)

# %%
hist, timestamp = ret_hist(symbol)
hist = hist[::m]

asset_lis = []
for _ in range(10):
    kane, asset = ret_kane_asset(hist, rik, son)

    asset_lis.append(asset)

asset = np.stack(asset_lis, axis=1)
std = np.std(asset, axis=1)
asset = np.average(asset, axis=1)

asset_std = np.stack([asset + std*2, asset, asset - std*2], axis=1)
# %%
pd.DataFrame(asset_std).plot()
