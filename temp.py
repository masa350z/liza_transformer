# %%

import concurrent.futures
import pandas as pd
import numpy as np
from modules import modules
from tqdm import tqdm
# %%
symbol = 'USDJPY'

hist, timestamp = modules.ret_hist(symbol)
# %%


def ret_kane_asset(hist, rik, son):
    kane = 0
    asset = []
    position = 0
    pos = 0
    for i in hist:
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
"""
list01 = []
for j in range(100):
    list02 = []
    for k in range(100):
        rik, son = j/1000, k/1000
        kane, asset = ret_kane_asset(hist, rik, son)

        list02.append(kane)
    list01.append(list02)
"""
# %%


def compute_kane_for_jk(jk, hist):
    j, k = jk
    rik, son = j/100, k/100
    kane, asset = ret_kane_asset(hist, rik, son)
    return j, k, kane


list01 = [[None]*10 for _ in range(10)]

jk_combinations = [(j, k) for j in range(10) for k in range(10)]

with concurrent.futures.ThreadPoolExecutor() as executor:
    for j, k, result in executor.map(compute_kane_for_jk, jk_combinations, [hist]*10000):
        list01[j][k] = result

# %%
