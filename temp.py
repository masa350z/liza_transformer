# %%

import pandas as pd
import numpy as np
from modules import modules
# %%
symbol = 'EURUSD'

hist, timestamp = modules.ret_hist(symbol)
# %%
pd.DataFrame(hist).plot()
# %%
symbol_list = ['USDJPY', 'EURUSD', 'EURJPY']
hist, timestamp = modules.ret_multi_symbol_hist(symbol_list)
# %%
hist.shape
# %%
