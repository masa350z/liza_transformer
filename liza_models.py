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


def scaled_dot_product_attention(q, k, v, mask=None):
    """アテンションの重みの計算
    q, k, vは最初の次元が一致していること
    k, vは最後から2番めの次元が一致していること
    マスクは型(パディングかルックアヘッドか)によって異なるshapeを持つが、
    加算の際にブロードキャスト可能であること
    引数：
     q: query shape == (..., seq_len_q, depth)
     k: key shape == (..., seq_len_k, depth)
     v: value shape == (..., seq_len_v, depth_v)
     mask: (..., seq_len_q, seq_len_k) にブロードキャスト可能な
      shapeを持つ浮動小数点テンソル。既定値はNone
    戻り値：
     出力、アテンションの重み
     """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # matmul_qkをスケール
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # マスクをスケール済みテンソルに加算
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax は最後の軸(seq_len_k)について
    # 合計が1となるように正規化
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    poisition: シーケンスの最大長
    d_model: 1シーケンスの次元数(単語ベクトルの次元数)
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 配列中の偶数インデックスにはsinを適用; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 配列中の奇数インデックスにはcosを適用; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """最後の次元を(num_heads, depth)に分割。
        結果をshapeが(batch_size, num_heads, seq_len, depth)となるようにリシェイプする。
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


class OutputLayer(layers.Layer):
    def __init__(self, vector_dims, num_heads, inner_dims):
        super(OutputLayer, self).__init__()

        self.mha = MultiHeadAttention(vector_dims, num_heads)
        self.ffn = point_wise_feed_forward_network(vector_dims, inner_dims)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):

        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class TransformerBase(layers.Layer):
    def __init__(self, num_layer_loops, vector_dims,
                 num_heads, inner_dims):
        super(TransformerBase, self).__init__()

        self.num_layer_loops = num_layer_loops
        self.vector_dims = vector_dims
        self.num_heads = num_heads
        self.inner_dims = inner_dims

        self.cls_embedding = layers.Embedding(1, vector_dims)

        self.enc_layers = [OutputLayer(vector_dims, num_heads, inner_dims)
                           for _ in range(num_layer_loops)]

    def add_cls(self, x, batch_size):
        cls = self.cls_embedding(0)
        cls = tf.reshape(cls, (1, 1, -1))
        cls = tf.tile(cls, [batch_size, 1, 1])
        x = tf.concat([cls, x], axis=1)

        return x


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):

        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 max_sequence_len, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_sequence_len,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # 埋め込みと位置エンコーディングを合算する
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class NoEmbeddingEncoder(layers.Layer):
    def __init__(self, seq_len, num_layers, d_model, num_heads, dff, rate=0.1):
        super(NoEmbeddingEncoder, self).__init__()
        self.seq_len = seq_len

        self.d_model = d_model
        self.num_layers = num_layers

        self.encoder = layers.Dense(d_model)
        self.pos_encoding = positional_encoding(self.seq_len, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training=False, mask=False):
        x = self.encoder(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :self.seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class BTC_Transformer(tf.keras.Model):
    def __init__(self, seq_len, num_layers, d_model, num_heads, dff, output_size):
        super(BTC_Transformer, self).__init__()

        self.encoder = NoEmbeddingEncoder(
            seq_len, num_layers, d_model, num_heads, dff)
        self.dense01 = layers.Dense(output_size, activation='relu')
        self.dense02 = layers.Dense(output_size, activation='relu')
        self.output_layer = layers.Dense(2, activation='softmax')

        self.flatten = layers.Flatten()

    def call(self, inp):
        x = self.encoder(inp)
        x = self.flatten(x)

        x = self.dense01(x)
        x = self.dense02(x)

        return self.output_layer(x)


class FX_Transformer(TransformerBase):
    def __init__(self, seq_len, num_layer_loops, vector_dims, num_heads, inner_dims):
        super(FX_Transformer, self).__init__(num_layer_loops,
                                             vector_dims,
                                             num_heads,
                                             inner_dims)

        self.num_layer_loops = num_layer_loops
        self.pos_encoding = positional_encoding(seq_len, vector_dims)

    def call(self, x):
        for i in range(self.num_layer_loops):
            x += self.enc_layers[i](x)

        return x


class FX_Transformer_v2(layers.Layer):
    def __init__(self, seq_len, num_layer_loops, vector_dims, num_heads, inner_dims):
        super(FX_Transformer_v2, self).__init__()

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
        self.ffn = point_wise_feed_forward_network(vector_dims, inner_dims)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.pos_encoding = positional_encoding(seq_len, vector_dims)

    def call(self, x):
        x += self.pos_encoding

        for _ in range(self.num_layer_loops):
            attn_output = self.mha(x, x, x)  # MultiHeadAttention
            out1 = self.layernorm1(x + attn_output)

            ffn_output = self.ffn(out1)  # Feed-forward network
            x = self.layernorm2(out1 + ffn_output)

        return x


class SelfAttention(layers.Layer):
    def __init__(self, output_shape, act='relu'):
        super(SelfAttention, self).__init__()

        self.dense01 = layers.Dense(output_shape*2, activation='relu')
        self.dense02 = layers.Dense(output_shape, activation=act)

    def call(self, x):
        x = self.dense01(x)
        x = self.dense02(x)

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

        self.fx_transfomer = FX_Transformer_v2(
            seq_len-2, 1, feature_dim, 4, feature_dim)

        self.dence_layer = OutputLayers()
        self.output_layer = layers.Dense(2)

    def normalize(self, x, mode=0):
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

    def call(self, x):
        norm01 = self.normalize(x, mode=0)
        x = self.normalize(x, mode=1)
        x1 = self.conv01(x)
        x2 = self.conv02(tf.expand_dims(norm01[:, :, 0], axis=2))
        x3 = self.conv03(tf.expand_dims(norm01[:, :, -1], axis=2))

        x = self.fx_transfomer(x1 + x2 + x3)

        x = x[:, -1]

        x = self.dence_layer(tf.cast(x, tf.float32))
        x = self.output_layer(x)

        return layers.Activation('softmax', dtype='float32')(x)
