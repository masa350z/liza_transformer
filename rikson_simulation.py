# %%
from modules import modules
from tqdm import tqdm
import numpy as np
import sys
import os


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
            if (i - position)*pos > rik:
                kane += (i - position)*pos
                position = 0
            elif (i - position)*pos < -son:
                kane += (i - position)*pos
                position = 0
        asset.append(kane)

    return kane, asset


# %%

symbol = sys.argv[1]
m = int(sys.argv[2])
rik, son = float(sys.argv[2]), float(sys.argv[3])

base_dir = 'datas/simulation/{}'.format(symbol)
os.makedirs(base_dir, exist_ok=True)

hist, timestamp = modules.ret_hist(symbol)
hist = hist[::m]


asset_lis = []
for _ in range(10):
    kane, asset = ret_kane_asset(hist, rik, son)

    asset_lis.append(asset)

asset = np.stack(asset_lis, axis=1)
std = np.std(asset, axis=1)
asset = np.average(asset, axis=1)

asset_std = np.stack([asset + std*2, asset, asset - std*2], axis=1)

save_dir = base_dir + '/m{}_rik{}_son{}.npy'
np.save(save_dir.format(m, rik, son), asset_std)
# %%
