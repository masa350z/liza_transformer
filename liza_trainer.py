# %%
import liza_models
import trainer
import numpy as np
import pandas as pd
import os


class ModelTrainer:
    def __init__(self, hist, m_lis, k, pr_k):
        self.hist = hist
        self.m_lis = m_lis
        self.k = k
        self.pr_k = pr_k

    def run_train(self, seq_len, weights_save_name, batch_size,
                  per_batch=1, repeats=1, break_epochs=5):

        os.makedirs(weights_save_name, exist_ok=True)

        best_val_loss = float('inf')
        best_val_acc = 0

        # 指定した反復回数でモデルのトレーニングを実行
        for repeat in range(repeats):
            model = liza_models.LizaTransformer(seq_len)

            # データとモデルを用いてトレーニングのセッションを初期化
            liza_trainer = trainer.LizaTrainer(model, weight_name, batch_size,
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
symbol = 'BTCJPY'

k = 24
pr_k = 12

hist_path = 'D:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
df = pd.read_csv(hist_path)
hist = np.array(df['price'], dtype='float32')


m_lis = [15, 30, 45]
base_m = m_lis[0]

st = ''
for i in m_lis:
    st += str(i) + '_'
st = st[:-1]

weight_name = 'weights/{}/k{}_prk{}_basem{}_mlis{}'.format(
    symbol, k, pr_k, base_m, st)
os.makedirs(weight_name, exist_ok=True)


# %%
batch_size = 120*300
liza_trainer = ModelTrainer(hist, m_lis, k, pr_k,)
liza_trainer.run_train(k, weight_name, batch_size, repeats=1000)
# %%
