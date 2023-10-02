# %%
from keras import layers
import tensorflow as tf
import numpy as np

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

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


def normalize(x, mode=0):
    if mode == 0:
        mx = tf.reduce_max(x, axis=1, keepdims=True)
        mn = tf.reduce_min(x, axis=1, keepdims=True)
    else:
        mx = tf.reduce_max(tf.reduce_max(
            x, axis=1, keepdims=True), axis=2, keepdims=True)
        mn = tf.reduce_min(tf.reduce_min(
            x, axis=1, keepdims=True), axis=2, keepdims=True)

    normed = (x-mn)/(mx-mn)
    normed = tf.where(tf.math.is_finite(normed), normed, 0.0)

    return normed


class FX_Transformer(layers.Layer):
    def __init__(self, seq_len, num_layer_loops, vector_dims, num_heads, inner_dims):
        super(FX_Transformer, self).__init__()

        self.num_layer_loops = num_layer_loops
        self.vector_dims = vector_dims
        self.num_heads = num_heads
        self.inner_dims = inner_dims

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=vector_dims // num_heads,
            dropout=0.0,
            use_bias=True
        )
        self.ffn = self.point_wise_feed_forward_network(inner_dims)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.pos_encoding = self.positional_encoding(seq_len)

    def point_wise_feed_forward_network(self, dff):
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            layers.Dense(self.vector_dims)  # (batch_size, seq_len, d_model)
        ])

    def get_angles(self, pos, i):
        angle_rates = 1 / \
            np.power(10000, (2 * (i//2)) / np.float32(self.vector_dims))

        return pos * angle_rates

    def positional_encoding(self, position):
        """
        poisition: シーケンスの最大長
        d_model: 1シーケンスの次元数(単語ベクトルの次元数)
        """
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(self.vector_dims)[np.newaxis, :])

        # 配列中の偶数インデックスにはsinを適用; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 配列中の奇数インデックスにはcosを適用; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        x += self.pos_encoding

        for _ in range(self.num_layer_loops):
            attn_output = self.mha(x, x, x)  # MultiHeadAttention
            out1 = self.layernorm1(x + attn_output)

            ffn_output = self.ffn(out1)  # Feed-forward network
            x = self.layernorm2(out1 + ffn_output)

        return x


class OutputLayers(layers.Layer):
    def __init__(self):
        super(OutputLayers, self).__init__()

        self.dense01 = layers.Dense(250, activation='relu')
        self.dense02 = layers.Dense(250, activation='relu')

        self.dense03 = layers.Dense(150, activation='relu')
        self.dense04 = layers.Dense(150, activation='relu')

        self.dense05 = layers.Dense(50, activation='relu')
        self.dense06 = layers.Dense(50, activation='relu')

        self.dense07 = layers.Dense(25, activation='relu')
        self.dense08 = layers.Dense(25, activation='relu')

    def call(self, x):
        x = self.dense01(x)
        x_ = self.dense02(x)

        x = self.dense03(x + x_)
        x_ = self.dense04(x)

        x = self.dense05(x + x_)
        x_ = self.dense06(x)

        x = self.dense06(x + x_)
        x_ = self.dense07(x)

        return x_


class LizaTransformer(tf.keras.Model):
    def __init__(self, seq_len):
        super(LizaTransformer, self).__init__()
        feature_dim = 32

        self.conv01 = layers.Conv1D(
            filters=feature_dim, kernel_size=3, activation='relu')
        self.conv02 = layers.Conv1D(
            filters=feature_dim, kernel_size=3, activation='relu')
        self.conv03 = layers.Conv1D(
            filters=feature_dim, kernel_size=3, activation='relu')

        self.fx_transfomer = FX_Transformer(
            seq_len-2, 1, feature_dim, 4, feature_dim)

        self.dence_layer = OutputLayers()
        self.output_layer = layers.Dense(2)

    def call(self, x):
        norm01 = normalize(x, mode=0)
        x = normalize(x, mode=1)
        x1 = self.conv01(x)
        x2 = self.conv02(tf.expand_dims(norm01[:, :, 0], axis=2))
        x3 = self.conv03(tf.expand_dims(norm01[:, :, -1], axis=2))

        x = self.fx_transfomer(x1 + x2 + x3)

        x = x[:, -1]

        x = self.dence_layer(tf.cast(x, tf.float32))
        x = self.output_layer(x)

        return layers.Activation('softmax', dtype='float32')(x)
