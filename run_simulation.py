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
