import tensorflow as tf
from tqdm import tqdm
import numpy as np
import liza_module
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
                 train_rate=0.6, valid_rate=0.2):

        self.train_rate = train_rate
        self.valid_rate = valid_rate

        self.batch_size = batch_size

        train_dataset, valid_dataset, test_dataset = \
            self.ret_dataset(hist, m_lis, k, pr_k, base_m=base_m)

        self.train_dataset = train_dataset.batch(batch_size)
        self.valid_dataset = valid_dataset.batch(batch_size)
        self.test_dataset = test_dataset.batch(batch_size)

    def ret_dataset(self, hist, m_lis, k, pr_k, base_m=None):
        if base_m is None:
            base_m = m_lis[0]

        data_x, data_y = liza_module.ret_data_xy(hist, m_lis, base_m, k, pr_k)
        dataset_size = len(data_x)
        self.train_size = int(self.train_rate * dataset_size)
        self.val_size = int(self.valid_rate * dataset_size)

        train_x, valid_x, test_x = self.ret_data_xy(data_x)
        train_y, valid_y, test_y = self.ret_data_xy(data_y)

        train_dataset = tf.data.Dataset.zip((train_x, train_y))
        valid_dataset = tf.data.Dataset.zip((valid_x, valid_y))
        test_dataset = tf.data.Dataset.zip((test_x, test_y))

        return train_dataset, valid_dataset, test_dataset

    def ret_data_xy(self, inp_data):
        train_data = tf.data.Dataset.from_tensor_slices(
            inp_data[:self.train_size])

        valid_data = tf.data.Dataset.from_tensor_slices(
            inp_data[self.train_size:self.train_size + self.val_size])

        test_data = tf.data.Dataset.from_tensor_slices(
            inp_data[self.train_size + self.val_size:])

        return train_data, valid_data, test_data


class Trainer(LizaDataSet):
    def __init__(self, model, weight_name,
                 hist, m_lis, k, pr_k, batch_size,
                 base_m=None,
                 train_rate=0.6, valid_rate=0.2,
                 k_freeze=3, init_ratio=1e-4):

        super().__init__(hist, m_lis, k, pr_k,
                         batch_size, base_m,
                         train_rate, valid_rate)

        os.makedirs(weight_name, exist_ok=True)
        self.weight_name = weight_name + '/best_weights'

        self.model = model

        self.k_freeze = k_freeze
        self.temp_weights = None
        self.freeze, self.last_epoch = 0, 0
        self.init_ratio = init_ratio

        self.temp_val_loss = float('inf')
        self.best_val_loss = float('inf')
        self.temp_val_acc = 0
        self.best_val_acc = 0

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
                print(f"Best test  loss : {self.best_val_loss:.8f}")
                print(f"Best test  acc  : {self.best_val_acc:.8f}")

    def run_train(self, per_batch,
                  epochs=100000000, break_epochs=5):
        self.temp_val_loss = float('inf')
        self.temp_val_acc = 0
        self.last_epoch = 0

        for epoch in range(epochs):
            self.run_mono_train(epoch, per_batch)

            if epoch - self.last_epoch >= break_epochs or self.temp_val_loss == 0:
                break

        self.model.set_weights(self.temp_weights)
        test_acc, test_loss = self.ret_acc_loss()

        return test_acc, test_loss

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


class LizaTrainer(Trainer):
    def __init__(self, model, weight_name, batch_size,
                 hist, m_lis, k, pr_k, base_m=None,
                 k_freeze=3, train_rate=0.6, valid_rate=0.2,
                 init_ratio=1e-4, opt1=1e-5, opt2=5e-7, switch_epoch=30):

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