# %%
import liza_module
import numpy as np
import pandas as pd
import tensorflow as tf


# %%
m = 1
k = 90
pr_k = 30

btc_hist_path = 'D:/documents/hist_data/symbol/BTCJPY/1m.csv'
df = pd.read_csv(btc_hist_path)
hist = np.array(df['price'], dtype='int32')


pr_m = 256
m_lis = [128, 64, 32, 16, 8, 4, 2, 1]
multi_hist = [liza_module.ret_long_hist(hist, k, pr_m)]
for mm in m_lis:
    long_hist = liza_module.ret_long_hist(hist, k, mm)
    long_hist = long_hist[k*(pr_m-mm)-(pr_m-mm):]
    multi_hist.append(long_hist)

mn_len = len(multi_hist[-1])
multi_hist = [ml[:mn_len] for ml in multi_hist]
multi_hist = np.array(multi_hist)
multi_hist = multi_hist.transpose(1, 2, 0)

y_2d = liza_module.hist_conv2d(hist[(pr_m)*(k-1):], pr_k)
data_y = 1*(y_2d[:, -1] - y_2d[:, 0] > 0)
data_y = np.concatenate(
    [data_y.reshape(-1, 1), ((data_y-1)*-1).reshape(-1, 1)], axis=1)

multi_hist = multi_hist[:len(data_y)]
# %%
mx = np.max(np.max(multi_hist, axis=1), axis=1)
mn = np.min(np.min(multi_hist, axis=1), axis=1)
# %%
a = (multi_hist - mn.reshape(-1, 1, 1))
b = (mx.reshape(-1, 1, 1) - mn.reshape(-1, 1, 1))
normed = (a/b).astype('float32')
# %%
train_x, valid_x, test_x = liza_module.split_data(normed)
train_y, valid_y, test_y = liza_module.split_data(data_y)
# %%
model = liza_module.BTC_Transformer(seq_len=k,
                                    num_layers=1,
                                    d_model=126,
                                    num_heads=3,
                                    dff=128,
                                    output_size=128)


learning_rate = 2e-5
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
# %%
best_val_loss = float('inf')
k_freeze = 3
epochs = 100


def model_weights_random_init(init_ratio=0.0001):
    """
    モデルの重みをランダムに初期化する関数
    """
    # モデルの重みを取得する
    weights = model.get_weights()

    # 重みをランダムに初期化する
    for i, weight in enumerate(weights):
        if len(weight.shape) == 2:
            # 重み行列の場合、init_ratioの割合でランダム初期化する
            rand_mask = np.random.binomial(1, init_ratio, size=weight.shape)
            rand_weights = np.random.randn(*weight.shape) * rand_mask
            weights[i] = weight * (1 - rand_mask) + rand_weights

    # モデルの重みをセットする
    model.set_weights(weights)


freeze = k_freeze
val_loss, val_acc = model.evaluate(valid_x, valid_y)
print(f"Initial valid loss: {val_loss}")

# 学習を開始する
for epoch in range(epochs):
    model.fit(train_x, train_y)
    val_loss, val_acc = model.evaluate(valid_x, valid_y)

    # valid lossが減少した場合、重みを保存
    if val_loss < best_val_loss:
        freeze = 0
        best_val_loss = val_loss
        model.save_weights('weights/best_weights')
        print(
            f"Epoch {epoch + 1}: Valid loss decreased to {val_loss}, saving weights.")

    # valid lossが減少しなかった場合、保存しておいた最良の重みをロード
    else:
        if freeze == 0:
            model.load_weights('weights/best_weights')
            model_weights_random_init()
            freeze = k_freeze
            print(
                f"Epoch {epoch + 1}: Valid loss did not decrease, loading weights.")
        else:
            print(f"Epoch {epoch + 1}: Valid loss did not decrease.")

    freeze = freeze - 1 if freeze > 0 else freeze

    print('')

# %%
