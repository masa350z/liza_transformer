# %%
import subprocess
# %%
symbol = 'USDJPY'
m = 1

for i in range(10):
    for j in range(10):
        rik = 0.001*(i+1)
        son = 0.001*(j+1)
        subprocess.Popen(['python', 'main.py', symbol,
                         str(m), str(rik), str(son)])

# %%
