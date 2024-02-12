# %%
import os
import numpy as np
import seaborn as sns
# %%
symbol = 'EURUSD'
symbol = 'USDJPY'
m = 1
heatmap = []

for i in range(10):
    rik = 0.00001*(i+1)
    temp = []
    for j in range(10, 20):
        son = 0.00005*(j+1)

        base_dir = 'datas/simulation/{}'.format(symbol)
        save_dir = base_dir + \
            '/m{}_rik{}_son{}.npy'.format(1,
                                          "{:.5f}".format(rik),
                                          "{:.5f}".format(son))

        if os.path.exists(save_dir):
            asset_std = np.load(save_dir)

            kane = asset_std[-1]
        else:
            kane = 0
        temp.append(kane)

    heatmap.append(temp)

# %%
sns.heatmap(heatmap)
# %%
0.00005*10
# %%
list(range(10, 20))[7]
# %%
list(range(10))[2]
# %%
0.00001*(2+1)
# %%
0.00005*(17+1)
# %%
