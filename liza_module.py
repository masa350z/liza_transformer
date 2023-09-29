from keras import layers
import tensorflow as tf
import numpy as np


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


def ret_multi_hist(hist, k, m_lis):
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


def ret_data_y(hist, m_lis, base_m, k, pr_k):
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

    return y_diff, y_one_hot


def ret_data_xy(hist, m_lis, base_m, k, pr_k,
                norm=True, y_mode='onehot'):
    """
    ヒストリカルデータから入力Xと目的のYデータを取得する関数。

    引数:
    - hist (np.array): (n,)の形を持つヒストリカルデータ。
    - m_lis (list of int): ローリング時のストライド値のリスト。
    - base_m (int): 基準となるストライド値。
    - k (int): ヒストリカルデータを考慮するデータポイントの数。
    - pr_k (int): 予測する先のデータポイント数。
    - y_mode (str, optional): 返されるYデータの形式 ('onehot'または他の任意の値)。

    戻り値:
    - multi_hist (np.array): 学習データ。
    - y_one_hot (np.array) or y_diff (np.array): ラベルデータ。
    """
    multi_hist = ret_multi_hist(hist, k, m_lis)
    y_diff, y_one_hot = ret_data_y(hist, m_lis, base_m, k, pr_k)

    multi_hist = multi_hist[:len(y_one_hot)]

    if norm:
        multi_hist = normalize(multi_hist)

    if y_mode == 'onehot':
        return multi_hist, y_one_hot
    else:
        return multi_hist, y_diff


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


def scaled_dot_product_attention(q, k, v, mask=None):
    """アテンションの重みの計算
    q, k, vは最初の次元が一致していること
    k, vは最後から2番めの次元が一致していること
    マスクは型（パディングかルックアヘッドか）によって異なるshapeを持つが、
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
    d_model: 1シーケンスの次元数（単語ベクトルの次元数）
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

    def call(self, v, k, q, mask):
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
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


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

        self.pos_encoding = positional_encoding(self.seq_len, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training=False, mask=False):
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
