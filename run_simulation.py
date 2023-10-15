# %%
import subprocess
# %%
symbol = 'EURUSD'
m = 15

for i in range(10):
    for j in range(0, 20):
        rik = 0.001*(i+1)/10
        son = 0.01*(j+1)/10
        subprocess.Popen(['python', 'rikson_simulation.py', symbol,
                         str(m), str(rik), str(son)])

# %%
"""
symbol = 'EURUSD'
m = 3
rik = 0.005
son = 0.1

base_dir = 'datas/simulation/{}'.format(symbol)

list01 = []
for i in range(10):
    list02 = []
    for j in range(20):
        rik = round(0.001*(i+1)/100, 3+2)
        son = round(0.01*(j+1)/100, 2+2)

        save_dir = base_dir + '/m{}_rik{}_son{}.npy'.format(m, rik, son)

        temp_array = np.load(save_dir)
        kane = temp_array[:, 1][-1]

        list02.append(kane)
    list01.append(list02)
# %%
list01 = np.array(list01)
# %%
sns.heatmap(list01)
"""
