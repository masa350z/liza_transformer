from scipy.spatial import cKDTree
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def ret_hist(symbol):
    hist_path = 'E:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
    df = pd.read_csv(hist_path)
    hist = np.array(df['price'], dtype='float32')
    timestamp = np.array(df['timestamp'], dtype='int32')

    return hist, timestamp


def ret_multi_symbol_hist(symbol_list):
    hist_lis, time_lis = [], []
    for symbol in symbol_list:
        hist, timestamp = ret_hist(symbol)

        hist_lis.append(hist)
        time_lis.append(timestamp)

    # Convert lists to numpy arrays
    time_lis0 = np.array(time_lis[0])
    time_lis1 = np.array(time_lis[1])
    time_lis2 = np.array(time_lis[2])

    # Create KDTree for time_lis[1] and time_lis[2]
    tree1 = cKDTree(time_lis1[:, np.newaxis])
    tree2 = cKDTree(time_lis2[:, np.newaxis])

    # Query the nearest point for each value in time_lis0
    dist1, idx1 = tree1.query(time_lis0[:, np.newaxis], k=1)
    dist2, idx2 = tree2.query(time_lis0[:, np.newaxis], k=1)

    # Fetch the prices and time differences
    price01_values = np.array(hist_lis[1])[idx1]
    price02_values = np.array(hist_lis[2])[idx2]

    hist_data = np.column_stack(
        (time_lis0, hist_lis[0], price01_values, price02_values))
    # time_diff_lis = np.column_stack((dist1, dist2))

    cut_pos = 0
    for i in range(3):
        temp_data = hist_data[:, i+1]
        for j in range(len(temp_data)-1):
            if temp_data[j+1] == temp_data[j]:
                if j > cut_pos:
                    cut_pos = j
            else:
                break

    hist_data = hist_data[cut_pos:]
    timestamp = hist_data[:, 0]
    hist_data = hist_data[:, 1:]

    return hist_data, timestamp


def hist_conv2d(hist, k, m=1):
    """
    1次元のヒストリカルデータを2次元の形に変換する関数。

    引数:
    - hist (np.array): (n,)の形を持つヒストリカルデータ。
    - k (int): 各エントリーに対して考慮する前のデータポイントの数。
    - m (int, optional): ローリング時のストライド。デフォルトは1。

    戻り値:
    - np.array: (n-k+1, k)の形を持つ2次元配列。各行はkのヒストリカルデータポイントのシーケンスに対応する。
    """

    # kまでの各インデックスに対してデータをロールする。
    hist_2d = [np.roll(hist[::m], -i) for i in range(k)]
    hist_2d = np.array(hist_2d)[:, :-(k-1)].T  # 各シーケンスを別々の行にするために転置する。

    return hist_2d


def ret_long_hist2d(hist, k, m):
    """
    2次元のヒストリカルデータの拡張バージョンを作成する関数。

    引数:
    - hist (np.array): (n,)の形を持つヒストリカルデータ。
    - k (int): 各エントリーに対して考慮する前のデータポイントの数。
    - m (int): ローリング時のストライド。

    戻り値:
    - np.array: オリジナルの2次元配列から各シーケンスがm回繰り返される拡張2次元配列。
    """
    hist_2d = hist_conv2d(hist, k, m)  # 1Dのヒストリカルデータを2Dに変換する。

    # 新しい第3次元に沿って各シーケンスをm回繰り返す。
    hist_2d = np.tile(np.expand_dims(hist_2d, 2), m)
    hist_2d = hist_2d.transpose(0, 2, 1)  # 次元の順序を変更する。
    hist_2d = np.concatenate(hist_2d)  # 最終的な2Dの形を取得するために第1軸に沿って結合する。

    return hist_2d


def ret_multi_length_hist(hist, k, m_lis):
    """
    複数のストライド値に基づいてヒストリカルデータの拡張版を作成する関数。

    引数:
    - hist (np.array): (n,)の形を持つヒストリカルデータ。
    - k (int): 各エントリーに対して考慮する前のデータポイントの数。
    - m_lis (list of int): ローリング時のストライド値のリスト。

    戻り値:
    - np.array: m_lisの各ストライド値に対する拡張2次元配列を結合した結果。
    """
    hist_lis = []
    for m in m_lis:
        # mごとのストライドでの拡張2次元ヒストリカルデータを取得し、リストに追加する。
        hist_lis.append(ret_long_hist2d(hist, k, m)[(k-1)*(max(m_lis)-m):])

    # すべての拡張データで共通の最小の長さを取得する。
    mn_len = min([i.shape[0] for i in hist_lis])
    # 各拡張データを共通の長さに切り詰める。
    hist_lis = np.array([i[:mn_len] for i in hist_lis])
    # 最終的な配列を形成するために次元を転置する。
    hist_lis = hist_lis.transpose(1, 2, 0)

    return hist_lis[::min(m_lis)]


def ret_data_y(hist, m_lis, base_m, k, pr_k, threshold=0.3):
    """
    ヒストリカルデータから目的のラベルデータを取得する関数。

    引数:
    - hist (np.array): (n,)の形を持つヒストリカルデータ。
    - m_lis (list of int): ローリング時のストライド値のリスト。
    - base_m (int): 基準となるストライド値。
    - k (int): ヒストリカルデータを考慮するデータポイントの数。
    - pr_k (int): 予測する先のデータポイント数。

    戻り値:
    - y_diff (np.array): 予測期間内の差分値。
    - y_one_hot (np.array): 予測の差分が正かどうかの真偽値。
    """
    y_2d = ret_long_hist2d(hist, k+pr_k, base_m)  # 拡張ヒストリカルデータを取得する。
    y_2d = y_2d[(k-1)*(max(m_lis)-base_m):]  # 必要な部分だけスライスする。
    y_2d = y_2d[::base_m]  # 基準のストライドでダウンサンプリングする。
    y_2d = y_2d[:, k-1:]  # k以降の部分だけを取得する。
    y_diff = y_2d[:, -1] - y_2d[:, 0]  # 予測期間の差分を計算する。
    y_one_hot = y_diff > 0  # 差分が正かどうかを判断する。

    norm_y2d = normalize(y_2d)
    up10 = norm_y2d[:, 0] > (1-threshold)
    mid = (norm_y2d[:, 0] <= (1-threshold))*(norm_y2d[:, 0] >= threshold)
    dn10 = norm_y2d[:, 0] < threshold

    y_updn = np.stack([up10, mid, dn10], axis=1)

    d1 = np.expand_dims(y_one_hot*1, 1)
    d2 = (d1-1)*-1
    y_one_hot = np.concatenate([d1, d2], axis=1)

    return y_diff, y_one_hot, y_updn


def ret_future_y(hist, m_lis, base_m, k, pr_k):
    y_2d = ret_long_hist2d(hist, k+pr_k, base_m)  # 拡張ヒストリカルデータを取得する。
    y_2d = y_2d[(k-1)*(max(m_lis)-base_m):]  # 必要な部分だけスライスする。
    y_2d = y_2d[::base_m]  # 基準のストライドでダウンサンプリングする。
    y_2d = y_2d[:, k-1:]  # k以降の部分だけを取得する。

    return y_2d


def ret_data_xy(hist, m_lis, base_m, k, pr_k,
                norm=False, y_mode='binary'):
    """
    ヒストリカルデータから入力Xと目的のYデータを取得する関数。

    引数:
    - hist (np.array): (n,)の形を持つヒストリカルデータ。
    - m_lis (list of int): ローリング時のストライド値のリスト。
    - base_m (int): 基準となるストライド値。
    - k (int): ヒストリカルデータを考慮するデータポイントの数。
    - pr_k (int): 予測する先のデータポイント数。
    - y_mode (str, optional): 返されるYデータの形式 ('binary'または他の任意の値)。

    戻り値:
    - multi_hist (np.array): 学習データ。
    - y_one_hot (np.array) or y_diff (np.array): ラベルデータ。
    """
    multi_length_hist = ret_multi_length_hist(hist, k, m_lis)
    y_diff, y_one_hot, y_updn = ret_data_y(hist, m_lis, base_m, k, pr_k)

    multi_length_hist = multi_length_hist[:len(y_one_hot)]

    if norm:
        multi_length_hist = normalize(multi_length_hist)

    if y_mode == 'binary':
        return multi_length_hist, y_one_hot
    if y_mode == 'contrarian':
        return multi_length_hist, y_updn
    else:
        return multi_length_hist, np.stack([y_diff, -1*y_diff], axis=1)


def normalize(hist_data_2d):
    mx = np.max(hist_data_2d, axis=1, keepdims=True)
    mn = np.min(hist_data_2d, axis=1, keepdims=True)

    normed = (hist_data_2d - mn)/(mx - mn)

    return np.nan_to_num(normed).astype('float32')


def split_data(inp, tr_rate=0.6, val_rate=0.2):
    train_len = int(len(inp)*tr_rate)
    valid_len = int(len(inp)*val_rate)

    train = inp[:train_len]
    valid = inp[train_len:train_len+valid_len]
    test = inp[train_len+valid_len:]

    return train, valid, test


def ret_weight_name(symbol, k, pr_k, m_lis, y_mode='binary'):
    st = ''
    for i in m_lis:
        st += str(i) + '_'
    st = st[:-1]

    weight_name = 'weights/{}/{}/k{}_prk{}_mlis{}'.format(y_mode,
                                                          symbol,
                                                          str(k).zfill(3),
                                                          str(pr_k).zfill(3),
                                                          str(st).zfill(3))

    return weight_name


def half_updn(inp_x, inp_y):
    up_y = inp_y[inp_y[:, 0] == 1]
    dn_y = inp_y[inp_y[:, 1] == 1]

    up_x = inp_x[inp_y[:, 0] == 1]
    dn_x = inp_x[inp_y[:, 1] == 1]

    diff = len(up_y) - len(dn_y)

    if diff > 0:
        index = np.arange(len(up_y))
        np.random.shuffle(index)

        up_y = up_y[index][:-diff]
        up_x = up_x[index][:-diff]

        y_ = np.concatenate([up_y, dn_y])
        x_ = np.concatenate([up_x, dn_x])
    elif diff < 0:
        index = np.arange(len(dn_y))
        np.random.shuffle(index)

        dn_y = dn_y[index][:diff]
        dn_x = dn_x[index][:diff]

        y_ = np.concatenate([up_y, dn_y])
        x_ = np.concatenate([up_x, dn_x])

    else:
        y_ = inp_y
        x_ = inp_x

    return x_, y_


class GradualDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    徐々に学習率を減少させるカスタムスケジュールクラス。
    初期学習率から最終学習率に向かって指定したステップ数で減少する。
    """

    def __init__(self, initial_learning_rate, final_learning_rate, decay_steps):
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate

        # 多項式減衰を使用して学習率を調整
        self.decay_schedule_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate, decay_steps, final_learning_rate, power=1.0)

    def __call__(self, step):
        return self.decay_schedule_fn(step)


class LizaDataSet:
    def __init__(self, hist, m_lis, k, pr_k,
                 batch_size, base_m=None,
                 train_rate=0.6, valid_rate=0.2,
                 y_mode='binary'):

        self.train_rate = train_rate
        self.valid_rate = valid_rate
        self.y_mode = y_mode

        self.batch_size = batch_size

        train_dataset, valid_dataset, test_dataset = \
            self.ret_dataset(hist, m_lis, k, pr_k, base_m=base_m)

        self.train_dataset = train_dataset.batch(batch_size)
        self.valid_dataset = valid_dataset.batch(batch_size)
        self.test_dataset = test_dataset.batch(batch_size)

    def ret_data_array(self, hist, m_lis, k, pr_k, base_m=None):
        if base_m is None:
            base_m = m_lis[0]

        data_x, data_y = ret_data_xy(
            hist, m_lis, base_m, k, pr_k, y_mode=self.y_mode)
        dataset_size = len(data_x)
        self.train_size = int(self.train_rate * dataset_size)
        self.val_size = int(self.valid_rate * dataset_size)

        train_datax = data_x[:self.train_size]
        valid_datax = data_x[self.train_size + self.val_size:]
        test_datax = data_x[self.train_size + self.val_size:]

        train_datay = data_y[:self.train_size]
        valid_datay = data_y[self.train_size + self.val_size:]
        test_datay = data_y[self.train_size + self.val_size:]

        return [train_datax, valid_datax, test_datax], [train_datay, valid_datay, test_datay]

    def ret_dataset(self, hist, m_lis, k, pr_k, base_m=None, half=True):
        if base_m is None:
            base_m = m_lis[0]

        data_x, data_y = ret_data_xy(
            hist, m_lis, base_m, k, pr_k, y_mode=self.y_mode)

        dataset_size = len(data_x)
        self.train_size = int(self.train_rate * dataset_size)
        self.val_size = int(self.valid_rate * dataset_size)

        train_x, valid_x, test_x = split_data(
            data_x, self.train_rate, self.valid_rate)
        train_y, valid_y, test_y = split_data(
            data_y, self.train_rate, self.valid_rate)

        if half:
            valid_x, valid_y = half_updn(valid_x, valid_y)
            test_x, test_y = half_updn(test_x, test_y)

        train_x, valid_x, test_x = self.convert_to_tensor_slices(
            train_x, valid_x, test_x)
        train_y, valid_y, test_y = self.convert_to_tensor_slices(
            train_y, valid_y, test_y)

        train_dataset = tf.data.Dataset.zip((train_x, train_y))
        valid_dataset = tf.data.Dataset.zip((valid_x, valid_y))
        test_dataset = tf.data.Dataset.zip((test_x, test_y))

        return train_dataset, valid_dataset, test_dataset

    def convert_to_tensor_slices(self, train_d, valid_d, test_d):

        train_data = tf.data.Dataset.from_tensor_slices(train_d)
        valid_data = tf.data.Dataset.from_tensor_slices(valid_d)
        test_data = tf.data.Dataset.from_tensor_slices(test_d)

        return train_data, valid_data, test_data


class Trainer(LizaDataSet):
    def __init__(self, model, weight_name,
                 hist, m_lis, k, pr_k, batch_size,
                 base_m=None,
                 train_rate=0.6, valid_rate=0.2,
                 k_freeze=3, init_ratio=1e-4,
                 y_mode='binary'):

        super().__init__(hist, m_lis, k, pr_k,
                         batch_size, base_m,
                         train_rate, valid_rate,
                         y_mode=y_mode)

        os.makedirs(weight_name, exist_ok=True)
        self.weight_name = weight_name + '/best_weights'

        self.model = model

        self.k_freeze = k_freeze
        self.temp_weights = None
        self.freeze, self.last_epoch = 0, 0
        self.init_ratio = init_ratio

        self.temp_val_loss = float('inf')
        self.best_test_loss = float('inf')
        self.temp_val_acc = 0
        self.best_test_acc = 0

        self.repeats = 0

    def model_weights_random_init(self, init_ratio=1e-4):
        weights = self.model.get_weights()

        for i, weight in enumerate(weights):
            if len(weight.shape) == 2:
                # ランダムなマスクを作成して重みを初期化
                rand_mask = np.random.binomial(
                    1, init_ratio, size=weight.shape)
                rand_weights = np.random.randn(*weight.shape) * rand_mask
                weights[i] = weight * (1 - rand_mask) + rand_weights

        self.model.set_weights(weights)

    def ret_prediction_onehot(self, dataset):
        prediction, one_hot_label = [], []
        # データセットを反復処理して予測を行う
        for x, y in tqdm(dataset):
            prediction.append(self.model(x))
            one_hot_label.append(y)

        prediction = tf.concat(prediction, axis=0)
        one_hot_label = tf.concat(one_hot_label, axis=0)

        return prediction, one_hot_label

    def run_mono_train(self, epoch, per_batch):
        """
        単一のトレーニングエポックを実行します。

        Args:
            epoch (int): 現在のエポック番号。
            per_batch (int): メトリック計算間に処理するバッチ数。
            acc_func (function): 検証精度を計算する関数。
            loss_func (function): 検証損失を計算する関数。
            init_ratio (float): モデルの重みのランダム初期化の割合。

        Returns:
            None
        """
        for (batch, (data_x, data_y)) in enumerate(self.train_dataset):
            # 現在のバッチに対してトレーニンングステップを実行する
            self.train_step(data_x, data_y)
            steps = math.ceil(self.train_size/self.batch_size)
            condition1 = (
                batch+1) % int(steps/per_batch) == 0
            if condition1 and (steps == 1 or (batch != 0)):
                # バリデーションデータセットで予測を行う
                prediction, one_hot_label = self.ret_prediction_onehot(
                    self.valid_dataset)

                # バリデーションの精度を計算する
                val_acc = self.calc_acurracy(prediction, one_hot_label).numpy()
                # バリデーションの損失を計算する
                val_loss = self.calc_loss(prediction, one_hot_label).numpy()

                if val_loss < self.temp_val_loss:
                    # 現在の損失がより低い場合、最良のバリデーション損失と精度を更新する
                    self.last_epoch = epoch
                    self.freeze = 0
                    self.temp_val_loss = val_loss
                    self.temp_val_acc = val_acc
                    self.temp_weights = self.model.get_weights()
                else:
                    if self.freeze == 0:
                        # モデルの重みを以前に保存した最良の重みにリセットする
                        self.model.set_weights(self.temp_weights)
                        # モデルの重みを与えられた割合でランダムに初期化する
                        self.model_weights_random_init(
                            init_ratio=self.init_ratio)
                        self.freeze = self.k_freeze

                print('=================')
                print(self.weight_name)
                # トレーニング情報を表示する
                print(f"Repeat : {self.repeats + 1}")
                print(f"Epoch : {epoch + 1}")
                print(f"Temp valid loss : {self.temp_val_loss:.8f}")
                print(f"Temp valid acc  : {self.temp_val_acc:.8f}")
                print(f"Best test  loss : {self.best_test_loss:.8f}")
                print(f"Best test  acc  : {self.best_test_acc:.8f}")

    def run_train(self, per_batch,
                  epochs=100000000,
                  break_epochs=5):

        self.temp_val_loss = float('inf')
        self.temp_val_acc = 0
        self.last_epoch = 0

        break_repeats = 5

        for epoch in range(epochs):
            self.run_mono_train(epoch, per_batch)

            if epoch - self.last_epoch >= break_epochs or self.temp_val_loss == 0:
                break

            if epoch == 0:
                first_acc = self.temp_val_acc

            elif epoch == break_repeats:
                if first_acc == self.temp_val_acc:
                    break

        if epoch != break_repeats:
            self.model.set_weights(self.temp_weights)
            test_acc, test_loss = self.ret_acc_loss()

            return test_acc, test_loss

        else:
            return 0, float('inf')

    def ret_acc_loss(self):
        prediction, one_hot_label = self.ret_prediction_onehot(
            self.test_dataset)

        # テストデータの精度を計算する
        test_acc = self.calc_acurracy(prediction, one_hot_label).numpy()
        # テストデータの損失を計算する
        test_loss = self.calc_loss(prediction, one_hot_label).numpy()

        return test_acc, test_loss

    def repeat_train(self, trainer,
                     per_batch=1, repeats=1, break_epochs=5):
        """
        モデルのトレーニングを実行する関数。

        Args:
        - weights_name: 保存されるモデルの重みの名前
        - per_batch, batch_size: トレーニングのバッチに関するパラメータ
        - repeats: トレーニングの反復回数
        - opt1, opt2, switch_epoch: オプティマイザの学習率に関するパラメータ

        Returns:
        - None
        """

        best_val_loss = float('inf')
        best_val_acc = 0

        # 指定した反復回数でモデルのトレーニングを実行
        for repeat in range(repeats):
            trainer.repeats = repeat
            trainer.best_val_loss = best_val_loss
            trainer.best_val_acc = best_val_acc

            # トレーニングの実行
            test_acc, test_loss = trainer.run_train(
                per_batch, break_epochs=break_epochs)

            # 最も良いtest_dataの損失を更新
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                best_val_acc = test_acc

                # トレーニング後のモデルの重みを保存
                trainer.model.save_weights(trainer.weight_name)

    def train_step(self, data_x, data_y):
        with tf.GradientTape() as tape:
            # 損失の計算
            # loss = tf.keras.losses.CategoricalCrossentropy()(data_y[0], self.model(data_x))
            loss = self.calc_loss(self.model(data_x), data_y)

        # 勾配の計算と重みの更新
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        return loss


class LizaTrainerBinary(Trainer):
    def __init__(self, model, weight_name, batch_size,
                 hist, m_lis, k, pr_k, base_m=None,
                 k_freeze=3, train_rate=0.6, valid_rate=0.2,
                 init_ratio=1e-4, opt1=1e-5, opt2=1e-6, switch_epoch=100):

        super().__init__(model, weight_name,
                         hist, m_lis, k, pr_k, batch_size,
                         base_m,
                         train_rate, valid_rate,
                         k_freeze, init_ratio)

        self.optimizer = self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=GradualDecaySchedule(opt1, opt2, switch_epoch))

    def calc_acurracy(self, prediction, label):
        predicted_indices = tf.argmax(prediction, axis=1)
        true_indices = tf.argmax(label, axis=1)

        correct_predictions = tf.equal(predicted_indices, true_indices)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        return accuracy

    def calc_loss(self, prediction, label):
        # loss = tf.keras.losses.CategoricalCrossentropy()(label, prediction)
        loss = tf.keras.losses.BinaryCrossentropy()(label, prediction)

        return loss


class LizaTrainerContrarian(Trainer):
    def __init__(self, model, weight_name, batch_size,
                 hist, m_lis, k, pr_k, base_m=None,
                 k_freeze=3, train_rate=0.6, valid_rate=0.2,
                 init_ratio=1e-4, opt1=1e-5, opt2=1e-6, switch_epoch=30):

        super().__init__(model, weight_name,
                         hist, m_lis, k, pr_k, batch_size,
                         base_m,
                         train_rate, valid_rate,
                         k_freeze, init_ratio,
                         y_mode='contrarian')

        self.optimizer = self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=GradualDecaySchedule(opt1, opt2, switch_epoch))

    def calc_acurracy(self, prediction, label):
        predicted_indices = tf.argmax(prediction, axis=1)
        true_indices = tf.argmax(label, axis=1)

        correct_predictions = tf.equal(predicted_indices, true_indices)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        return accuracy

    def calc_loss(self, prediction, label):
        loss = tf.keras.losses.CategoricalCrossentropy()(label, prediction)
        # loss = tf.keras.losses.BinaryCrossentropy()(label, prediction)

        return loss


class LizaTrainerDiffer(Trainer):
    def __init__(self, model, weight_name, batch_size,
                 hist, m_lis, k, pr_k, base_m=None,
                 k_freeze=3, train_rate=0.6, valid_rate=0.2,
                 init_ratio=1e-4, opt1=1e-5, opt2=1e-6, switch_epoch=30):

        super().__init__(model, weight_name,
                         hist, m_lis, k, pr_k, batch_size,
                         base_m,
                         train_rate, valid_rate,
                         k_freeze, init_ratio,
                         y_mode='differ')

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=GradualDecaySchedule(opt1, opt2, switch_epoch))

    def calc_acurracy(self, prediction, label):
        predicted_indices = tf.argmax(prediction, axis=1)
        true_indices = tf.argmax(label, axis=1)

        correct_predictions = tf.equal(predicted_indices, true_indices)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        return accuracy

    def calc_loss(self, prediction, label, spread=0):
        loss = prediction*label - spread
        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss)

        return -1*loss


class SimpleTrainer:
    def __init__(self, model, data_x, data_y,
                 train_rate=0.6, valid_rate=0.2, test_rate=0.2,
                 opt1=1e-4, opt2=1e-6, switch_epoch=30,
                 batch_size=32, init_ratio=1e-4,
                 model_name='best_model'):

        self.model = model
        self.model_name = model_name
        self.data_x = data_x
        self.data_y = data_y
        self.train_rate = train_rate
        self.valid_rate = valid_rate
        self.test_rate = test_rate
        self.batch_size = batch_size
        self.init_ratio = init_ratio

        self.model.optimizer = tf.keras.optimizers.Adam(
            learning_rate=GradualDecaySchedule(opt1, opt2, switch_epoch))

        # データを訓練データ、検証データ、テストデータに分割
        self.train_dataset, self.valid_dataset, self.test_dataset = self.split_dataset()

    def split_dataset(self):
        # データセット全体のサイズ
        dataset_size = len(self.data_x)
        # 訓練データセットのサイズ
        train_size = int(self.train_rate * dataset_size)
        # 検証データセットのサイズ
        valid_size = int(self.valid_rate * dataset_size)

        # データを分割
        train_x = self.data_x[:train_size]
        train_y = self.data_y[:train_size]
        valid_x = self.data_x[train_size:train_size+valid_size]
        valid_y = self.data_y[train_size:train_size+valid_size]
        test_x = self.data_x[train_size+valid_size:]
        test_y = self.data_y[train_size+valid_size:]

        # TensorFlowのデータセットに変換
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_x, train_y)).batch(self.batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices(
            (valid_x, valid_y)).batch(self.batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_x, test_y)).batch(self.batch_size)

        return train_dataset, valid_dataset, test_dataset

    def weights_random_init(self):
        weights = self.model.get_weights()
        for i, weight in enumerate(weights):
            if len(weight.shape) > 1:  # 重みが行列の場合に限定
                rand_mask = np.random.binomial(
                    1, self.init_ratio, size=weight.shape)
                rand_weights = np.random.randn(*weight.shape) * rand_mask
                weights[i] = weight * (1 - rand_mask) + rand_weights
        self.model.set_weights(weights)

    def train(self, epochs=100, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 訓練ループ
            print(f'\nEpoch {epoch+1}/{epochs}')
            train_progress = tqdm(self.train_dataset, desc='Training')
            for x, y in train_progress:
                self.train_step(x, y)

            # 検証ループ
            val_loss = self.evaluate(self.valid_dataset)
            train_progress.set_postfix(val_loss=val_loss)
            print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

            # 検証損失が改善されたかチェック
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 最良モデルを保存
                self.model.save_weights(self.model_name)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping...")
                    break

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.compute_loss(y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

    def compute_loss(self, y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    def evaluate(self, dataset):
        total_loss = 0
        num_batches = 0
        for x, y in dataset:
            predictions = self.model(x, training=False)
            loss = self.compute_loss(y, predictions)
            total_loss += tf.reduce_mean(loss)
            num_batches += 1
        avg_loss = total_loss / num_batches
        return avg_loss.numpy()

    def test_evaluate(self):
        self.model.load_weights('best_model.h5')
        test_loss = self.evaluate(self.test_dataset)
        print(f'Test Loss: {test_loss}')
        return test_loss

    def repeat_train(self, repeats=5):
        best_test_loss = float('inf')
        for repeat in range(repeats):
            print(f"Training repeat {repeat+1}/{repeats}")
            # 重みをランダムに初期化
            self.weights_random_init()
            # 訓練と検証
            self.train()
            # テストデータで評価
            test_loss = self.test_evaluate()
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                # 最良モデルを再保存
                self.model.save_weights('best_overall_model.h5')
                print(
                    f'>>> Best overall model updated with test loss: {best_test_loss:.4f}')
