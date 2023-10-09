# %%
from modules import modules
from modules import models
import pandas as pd
import os


def save_dataframe(weight_name, best_test_acc, best_test_loss):
    weights_df = pd.read_csv('weights/model_dic.csv', index_col=0)

    if weight_name in weights_df.index:
        weights_df.loc[weight_name].iloc[0] = best_test_loss
        weights_df.loc[weight_name].iloc[1] = best_test_acc
    else:
        model_dic = {}
        model_dic[weight_name] = [best_test_loss, best_test_acc]
        df = pd.DataFrame(model_dic).T
        df.columns = ['loss', 'acc']
        weights_df = pd.concat([weights_df, df])

    weights_df.to_csv('weights/model_dic.csv')


class ModelTrainer:
    def __init__(self, symbol, base_m, k, pr_k, y_mode='binary'):
        self.symbol = symbol
        self.base_m = base_m
        self.k = k
        self.pr_k = pr_k
        self.y_mode = y_mode

    def run_train(self, seq_len, weights_save_name, batch_size,
                  per_batch=1, repeats=1000,
                  break_epochs=5, break_repeats=10):

        os.makedirs(weights_save_name, exist_ok=True)

        best_test_loss = float('inf')
        # best_test_value = float('inf')
        best_test_acc = 0
        last_repeat = 0
        repeats_count = 0

        # 指定した反復回数でモデルのトレーニングを実行
        for repeat in range(repeats):
            model = models.LizaSimilarTransformer(seq_len, out_dim=2)
            # データとモデルを用いてトレーニングのセッションを初期化
            liza_trainer = modules.LizaTrainerSimilar(model, weight_name, batch_size,
                                                      self.symbol, self.k, self.pr_k, self.base_m)

            liza_trainer.repeats = repeat
            liza_trainer.best_test_loss = best_test_loss
            liza_trainer.best_test_acc = best_test_acc

            # トレーニングの実行
            test_acc, test_loss = liza_trainer.run_train(
                per_batch, break_epochs=break_epochs)

            # test_value = test_loss*abs(test_loss-liza_trainer.temp_val_loss)

            if test_acc != 0:
                # 最も良いtest_dataの損失を更新
                # if test_value < best_test_value:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_acc = test_acc
                    # best_test_value = test_value

                    last_repeat = repeats_count

                    # トレーニング後のモデルの重みを保存
                    liza_trainer.model.save_weights(liza_trainer.weight_name)

                repeats_count += 1

            if repeats_count - last_repeat >= break_repeats:
                break

        return best_test_acc, best_test_loss


# %%
y_mode = 'binary'
symbol = 'USDJPY'
for symbol in ['USDJPY']:
    for pr_k in [12]:
        for k, batch_size in [[12, 120*1000]]:
            for base_m in [15]:
                weight_name = modules.ret_weight_name(symbol=symbol,
                                                      k=k,
                                                      pr_k=pr_k,
                                                      m_lis='similar',
                                                      y_mode=y_mode)

                os.makedirs(weight_name, exist_ok=True)

                liza_trainer = ModelTrainer(symbol, base_m, k, pr_k)
                best_test_acc, best_test_loss = liza_trainer.run_train(
                    k, weight_name, batch_size, break_repeats=10)

                save_dataframe(weight_name, best_test_acc, best_test_loss)
