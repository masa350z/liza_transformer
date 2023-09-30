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

    d1 = np.expand_dims(y_one_hot*1, 1)
    d2 = (d1-1)*-1
    y_one_hot = np.concatenate([d1, d2], axis=1)

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
