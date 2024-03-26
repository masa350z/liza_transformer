# %%
import subprocess

for jj in range(10):
    subprocess.Popen(['python', 'simulator.py', str(jj)])

# %%
