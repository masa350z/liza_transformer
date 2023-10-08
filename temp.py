# %%

import pandas as pd
import numpy as np
from modules import modules
# %%
symbol = 'USDJPY'

hist, timestamp = modules.ret_hist(symbol)
# %%
k = 12
pr_k = 12

base_m = 15
m_lis = [base_m, base_m*2, base_m*3]

data_x, data_y = modules.ret_data_xy(
    hist, m_lis, base_m, k, pr_k, y_mode='binary')

# %%
