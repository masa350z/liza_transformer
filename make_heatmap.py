# %%
import os
import numpy as np
import seaborn as sns
# %%
symbol = 'EURUSD'
m = 1
heatmap = []

for i in range(20):
    # rik = round(0.001*(i+1), 3)
    rik = round(0.001*(i+1), 4)
    temp = []
    for j in range(20):
        # son = round(0.01*(j+1), 2)
        son = round(0.001*(j+1), 4)

        base_dir = 'datas/simulation/{}'.format(symbol)
        save_dir = base_dir + '/m{}_rik{}_son{}.npy'.format(m, rik, son)

        if os.path.exists(save_dir):
            asset_std = np.load(save_dir)

            kane = asset_std[-1:, 1][0]
        else:
            kane = 0
        temp.append(kane)

    heatmap.append(temp)

# %%
sns.heatmap(heatmap,
            yticklabels=(np.arange(20)+1)/1000,
            xticklabels=(np.arange(20)+1)/1000)
# %%
