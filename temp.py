# %%
import liza_transformer
import trainer
import numpy as np
import pandas as pd
import os
# %%
k = 30
pr_k = 30

btc_hist_path = 'D:/documents/hist_data/symbol/BTCJPY/1m.csv'
df = pd.read_csv(btc_hist_path)
hist = np.array(df['price'], dtype='int32')


m_lis = [5, 10, 15, 30]
base_m = m_lis[0]
# %%


class ModelTrainer:
    def __init__(self, hist, m_lis, k, pr_k):
        self.hist = hist
        self.m_lis = m_lis
        self.k = k
        self.pr_k = pr_k

    def run_train(self, weights_save_name, batch_size,
                  per_batch=1, repeats=1, break_epochs=5):

        os.makedirs(weights_save_name, exist_ok=True)

        best_val_loss = float('inf')
        best_val_acc = 0

        # 指定した反復回数でモデルのトレーニングを実行
        for repeat in range(repeats):
            model = liza_transformer.TimeSeriesModel(self.k)

            # データとモデルを用いてトレーニングのセッションを初期化
            liza_trainer = trainer.LizaTrainer(model, 'weight_name', batch_size,
                                               self.hist, self.m_lis, self.k, self.pr_k)

            liza_trainer.repeats = repeat
            liza_trainer.best_val_loss = best_val_loss
            liza_trainer.best_val_acc = best_val_acc

            # トレーニングの実行
            test_acc, test_loss = liza_trainer.run_train(
                per_batch, break_epochs=break_epochs)

            # 最も良いtest_dataの損失を更新
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                best_val_acc = test_acc

                # トレーニング後のモデルの重みを保存
                liza_trainer.model.save_weights(liza_trainer.weight_name)


# %%
model = liza_transformer.TimeSeriesModel(k)
# %%
batch_size = 120*50
liza_trainer = ModelTrainer(hist, m_lis, k, pr_k,)

# %%
liza_trainer.run_train('weight_name', batch_size, repeats=100)
# %%
