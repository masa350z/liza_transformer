# %%
import os
import sys
from modules import models, modules
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm


def ret_data_xy(k, hist_data2d):
    data_x = hist_data2d[:, :k]
    data_y = hist_data2d[:, k-1:]

    data_y = data_y[:, -1] > data_y[:, 0]
    data_y = np.expand_dims(data_y, 1)
    data_y = np.concatenate([data_y, np.logical_not(data_y)], axis=1)

    return data_x, data_y


def ret_combined_pricediff():
    price_diff_list = glob('datas/price_diff_list/*.csv')

    df = pd.DataFrame()
    for i in price_diff_list:
        df = pd.concat([df, pd.read_csv(i)])

    df_eurusd = df[df['0'] == 'EURUSD'].reset_index(drop=True)
    df_usdjpy = df[df['0'] == 'USDJPY'].reset_index(drop=True)

    return df_eurusd, df_usdjpy


class Simulator:
    def __init__(self, input_data, rik, son,
                 price_diff=None):

        self.input_data = input_data
        self.pricediff_df = price_diff

        self.kane = 0
        self.asset = []
        self.get_price = 0
        self.posi = 0

        self.rik = rik
        self.son = son

    def ret_random_position(self):
        if np.random.random() > 0.5:
            posi = 1
        else:
            posi = -1

        return posi

    def ret_rikson_bool(self, price):
        rikaku_bool = (price - self.get_price) * \
            self.posi > self.rik*self.get_price
        sonkiri_bool = (price - self.get_price) * \
            self.posi < -self.son*self.get_price

        return rikaku_bool, sonkiri_bool

    def ret_current_old_price(self, price):
        if self.pricediff_df is not None:
            current_price = price + self.pricediff_df.sample().iloc[0]
            old_price = self.get_price + self.pricediff_df.sample().iloc[0]
        else:
            current_price = price
            old_price = self.get_price

        return current_price, old_price

    def update_simlation(self, price, pred=None):
        rikaku_bool, sonkiri_bool = self.ret_rikson_bool(price)

        if rikaku_bool or sonkiri_bool:
            current_price, old_price = self.ret_current_old_price(price)

            self.kane += (current_price - old_price)*self.posi
            self.get_price = 0

        if self.get_price == 0:
            self.get_price = price
            if pred is None:
                self.posi = self.ret_random_position()
            else:
                self.posi = pred

        self.asset.append(self.kane)

    def run_simulation(self):
        if len(self.input_data.shape) == 1:
            for pred in tqdm(self.input_data):
                self.update_simlation(pred)

        else:
            for price, pred in tqdm(self.input_data):
                self.update_simlation(price, pred)

        return self.kane, self.asset


# %%
k, pr_k = 18, 6
df_eurusd, df_usdjpy = ret_combined_pricediff()
df_dic = {'EURUSD': df_eurusd['3'],
          'USDJPY': df_usdjpy['3']}

args = sys.argv

rik = (int(args[1])+1)*0.00001

for symbol in ['EURUSD', 'USDJPY']:
    for son_ in range(10):
        son = (son_+1)*0.0001

        weights_name = 'weights/affine/{}/{}_{}/best_weights'.format(
            symbol, k, pr_k)
        hist_data, _ = modules.ret_hist(symbol)

        hist_data2d = modules.hist_conv2d(hist_data, k+pr_k)
        data_x, data_y = ret_data_xy(k, hist_data2d)

        model = models.LizaAffine()
        model.load_weights(weights_name)

        prediction = model.predict(data_x, batch_size=50000)
        prediction = (prediction[:, 0] > 0.5)*2-1

        input_data = np.stack([hist_data2d[:, -1],
                               prediction], axis=1)

        simulator = Simulator(input_data,
                              rik=rik,
                              son=son,
                              price_diff=df_dic[symbol])

        kane, asset = simulator.run_simulation()

        data_name = 'datas/simulation/{}/rik{}_son{}.txt'.format(
            symbol, round(rik, 5), round(son, 5))

        with open(data_name, 'w') as f:
            f.write(str(kane))
# %%
