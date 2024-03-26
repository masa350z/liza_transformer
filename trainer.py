# %%
import os
import numpy as np
from modules import modules
from modules import models


def ret_sampled_indices(data_y):
    # ラベルのカウント
    up_count = np.sum(data_y[:, 0])
    down_count = np.sum(data_y[:, 1])

    # 少ない方のクラスの数を取得
    min_count = min(up_count, down_count)

    # 上昇と下降のインデックスを取得
    up_indices = np.where(data_y[:, 0])[0]
    down_indices = np.where(data_y[:, 1])[0]

    # それぞれからランダムにサンプリング
    np.random.seed(0)  # 再現性のためのシード設定
    up_sampled_indices = np.random.choice(up_indices, min_count, replace=False)
    down_sampled_indices = np.random.choice(
        down_indices, min_count, replace=False)

    # サンプリングしたインデックスを統合
    sampled_indices = np.concatenate(
        [up_sampled_indices, down_sampled_indices])
    np.random.shuffle(sampled_indices)  # データの順番をランダムにする

    return sampled_indices


def ret_data_xy(k, hist_data2d):
    data_x = hist_data2d[:, :k]
    data_y = hist_data2d[:, k-1:]

    data_y = data_y[:, -1] > data_y[:, 0]
    data_y = np.expand_dims(data_y, 1)
    data_y = np.concatenate([data_y, np.logical_not(data_y)], axis=1)

    return data_x, data_y


# %%
k = 18
p = 6
# %%
for symbol in ['USDJPY', 'EURUSD']:
    hist_data, _ = modules.ret_hist(symbol)

    hist_data2d = modules.hist_conv2d(hist_data, k+p)

    data_x = hist_data2d[:, :k]
    data_y = hist_data2d[:, k-1:]

    data_y = data_y[:, -1] > data_y[:, 0]
    data_y = np.expand_dims(data_y, 1)
    data_y = np.concatenate([data_y, np.logical_not(data_y)], axis=1)

    # 新しいdata_xとdata_yをサンプリングしたインデックスで構築
    sampled_indices = ret_sampled_indices(data_y)
    data_x_sampled = data_x[sampled_indices]
    data_y_sampled = data_y[sampled_indices]

    os.makedirs('weights/affine/{}/{}_{}'.format(symbol, k, p), exist_ok=True)
    weights_name = 'weights/affine/{}/{}_{}/best_weights'.format(symbol, k, p)
    trainer = modules.SimpleTrainer(models.LizaAffine(),
                                    data_x_sampled, data_y_sampled, batch_size=2500000,
                                    opt1=1e-4, opt2=1e-5, switch_epoch=1000,
                                    model_name=weights_name)

    trainer.train(10000)
# %%

symbol = 'EURUSD'
weights_name = 'weights/affine/{}/{}_{}/best_weights'.format(symbol, k, p)
hist_data, _ = modules.ret_hist(symbol)

hist_data2d = modules.hist_conv2d(hist_data, k+p)


data_x, data_y = ret_data_xy(k, hist_data2d)

model = models.LizaAffine()
model.load_weights(weights_name)
# %%
prediction = model.predict(data_x, batch_size=2500000)
# %%
np.average(prediction[:, 0])
# %%
np.sum(data_y[:, 0])/np.sum(data_y)

# %%
prediction
# %%
win = np.sum(data_y*prediction, axis=1) > 0.5
lose = np.sum(data_y*prediction, axis=1) <= 0.5
# %%
np.sum(win)/(len(win))
# %%
np.sum(win[-int(len(win)*0.2):])/int(len(win)*0.2)

# %%
