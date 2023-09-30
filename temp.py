# %%
import liza_transformer
import trainer
import numpy as np
import pandas as pd
# %%
k = 90
pr_k = 30

btc_hist_path = 'D:/documents/hist_data/symbol/BTCJPY/1m.csv'
df = pd.read_csv(btc_hist_path)
hist = np.array(df['price'], dtype='int32')


m_lis = [5, 10, 15, 30]
base_m = m_lis[0]
# %%
"""
model = liza_transformer.BTC_Transformer(seq_len=k,
                                         num_layers=1,
                                         d_model=128,
                                         num_heads=4,
                                         dff=128,
                                         output_size=128)
"""
# %%
model = liza_transformer.TimeSeriesModel()
# %%
batch_size = 120*50
liza_trainer = trainer.LizaTrainer(model, 'weight_name', batch_size,
                                   hist, m_lis, k, pr_k,)

# %%
liza_trainer.repeat_train(liza_trainer, repeats=5)
# %%
