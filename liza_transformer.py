# %%
import liza_module
import pandas as pd
import numpy as np
import tensorflow as tf


btc_hist_path = 'D:/documents/hist_data/symbol/BTCJPY/1m.csv'
# %%
m = 1
df = pd.read_csv(btc_hist_path)
hist_data = np.array(df['price'], dtype='int32')[::m]
hist_data_2d = liza_module.hist_conv2d(hist_data, 120)
# %%
data_x = hist_data_2d[:, :90]
data_y = hist_data_2d[:, 90:]
data_y = 1*(data_y[:, -1] - data_x[:, -1] > 0)
data_y = np.concatenate(
    [data_y.reshape(-1, 1), ((data_y-1)*-1).reshape(-1, 1)], axis=1)

data_x = liza_module.normalize(data_x)

train_x, valid_x, test_x = liza_module.split_data(data_x)
train_y, valid_y, test_y = liza_module.split_data(data_y)

indx = np.arange(len(train_x))
np.random.shuffle(indx)
train_x = train_x[indx]
train_y = train_y[indx]
# %%

model = liza_module.BTC_Transformer(seq_len=88,
                                    num_layers=3,
                                    d_model=32,
                                    num_heads=8,
                                    dff=512,
                                    output_size=256)


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
model.load_weights('weights/best_weights')
# %%
model.evaluate(test_x, test_y)
# %%
tr_pred = model.predict(train_x)
vl_pred = model.predict(valid_x)
te_pred = model.predict(test_x)
# %%
k = 0.5
win = np.sum((tr_pred > k)*train_y, axis=1)
bet = np.sum(tr_pred > k)
ratio = np.sum(win)/bet
print(bet/len(win))
print(ratio)
# %%
win = np.sum((vl_pred > k)*valid_y, axis=1)
bet = np.sum(vl_pred > k)
ratio = np.sum(win)/bet
print(bet/len(win))
print(ratio)
# %%
win = np.sum((te_pred > k)*test_y, axis=1)
bet = np.sum(te_pred > k)
ratio = np.sum(win)/bet
print(bet/len(win))
print(ratio)
# %%
