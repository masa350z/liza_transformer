# %%
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os


class Simulator:
    def __init__(self, symbol, rik, son):
        hist_path = 'E:/documents/hist_data/symbol/{}/1m.csv'.format(symbol)
        df = pd.read_csv(hist_path)
        self.hist = np.array(df['price'], dtype='float32')
        self.timestamp = np.array(df['timestamp'], dtype='int32')

        df_eurusd, df_usdjpy = self.ret_combined_pricediff()
        if symbol == 'EURUSD':
            self.pricediff_df = df_eurusd['3']
        else:
            self.pricediff_df = df_usdjpy['3']

        self.kane = 0
        self.asset = []
        self.get_price = 0
        self.posi = 0

        self.rik = rik
        self.son = son

    def ret_combined_pricediff(self):
        price_diff_list = glob('datas/price_diff_list/*.csv')

        df = pd.DataFrame()
        for i in price_diff_list:
            df = pd.concat([df, pd.read_csv(i)])

        df_eurusd = df[df['0'] == 'EURUSD'].reset_index(drop=True)
        df_usdjpy = df[df['0'] == 'USDJPY'].reset_index(drop=True)

        return df_eurusd, df_usdjpy

    def ret_random_position(self):
        if np.random.random() > 0.5:
            posi = 1
        else:
            posi = -1

        return posi

    def update_simlation(self, price):
        rikaku_bool = (price - self.get_price) * \
            self.posi > self.rik*self.get_price
        sonkiri_bool = (price - self.get_price) * \
            self.posi < -self.son*self.get_price

        if rikaku_bool or sonkiri_bool:
            price_diff01 = price + self.pricediff_df.sample().iloc[0]
            price_diff02 = self.get_price + self.pricediff_df.sample().iloc[0]
            self.kane += (price_diff01 - price_diff02)*self.posi
            self.get_price = 0

        if self.get_price == 0:
            self.get_price = price
            self.posi = self.ret_random_position()

        self.asset.append(self.kane)

    def run_simulation(self):
        for i in tqdm(self.hist):
            self.update_simlation(i)

        return self.kane, self.asset


# %%

jj = int(float(sys.argv[1]))

for symbol in ['USDJPY', 'EURUSD']:
    for i in range(20):
        rik = 0.00001*(i+1)
        for j in range(4):
            son = 0.00005*(j*10+jj)

            base_dir = 'datas/simulation/{}'.format(symbol)
            os.makedirs(base_dir, exist_ok=True)
            save_dir = base_dir + \
                '/m{}_rik{}_son{}.npy'.format(1,
                                              "{:.5f}".format(rik),
                                              "{:.5f}".format(son))

            if not os.path.exists(save_dir):
                simulator = Simulator(symbol, rik, son)
                kane, asset = simulator.run_simulation()

                print(symbol, rik, son, kane)

                np.save(save_dir, np.array(asset, dtype='float32'))
# %%
