# %%
import subprocess
# %%
m = 1
for symbol in ['EURUSD', 'USDJPY',]:
    for i in range(20, 30):
        for j in range(20, 30):
            rik = 0.001*(i+1)
            son = 0.001*(j+1)
            subprocess.Popen(['python', 'rikson_simulation.py', symbol,
                              str(m), str(rik), str(son)])

# %%
