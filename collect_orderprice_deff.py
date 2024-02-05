# %%
from fixar import FIXAR
import time
import pandas as pd
from glob import glob
import random


class TestFIXAR(FIXAR):
    def __init__(self, amount, k, pr_k,
                 sashine_eurusd, gyaku_sashine_eurusd,
                 sashine_usdjpy, gyaku_sashine_usdjpy,
                 dynamic_rik, dynamic_son):
        super().__init__(amount, k, pr_k,
                         sashine_eurusd, gyaku_sashine_eurusd,
                         sashine_usdjpy, gyaku_sashine_usdjpy,
                         dynamic_rik, dynamic_son)

    def test_order(self, side, symbol):
        rate_dic = self.ret_pricedic()

        rate = rate_dic[symbol]

        self.fixa[symbol].symbol_get_price = None

        self.fixa[symbol].ret_prediction()
        pred = True if side == 'buy' else False
        sashine, gyaku_sashine = self.ret_sashine_gyakusashine(
            symbol, rate, pred)

        self.make_order(symbol, side,
                        self.amount,
                        sashine, gyaku_sashine)

    def test_run(self, symbol, side):
        symbol_index = 0 if symbol == 'EURUSD' else 1

        start_time = time.time()
        usdjpy, eurusd = self.get_price()
        self.test_order(side, symbol)

        bool_eurusd = False
        while bool_eurusd is False:
            bool_eurusd, _, \
                get_price, _ \
                = self.position_bool()[symbol_index]

        end_time = time.time()

        time_diff = end_time - start_time
        if symbol == 'EURUSD':
            price_diff = get_price - eurusd
        else:
            price_diff = get_price - usdjpy

        self.settle_all_position()
        bool_eurusd = True
        while bool_eurusd is True:
            bool_eurusd, _, \
                get_price, _\
                = self.position_bool()[symbol_index]

        return time_diff, price_diff


# %%
demo = False
amount = 1000

sashine_eurusd, gyaku_sashine_eurusd = round(0.1/150, 5), round(0.1/150, 5)
sashine_usdjpy, gyaku_sashine_usdjpy = round(0.1, 3), round(0.1, 3)

dynamic_rik = {'EURUSD': 0.008/100, 'USDJPY': 0.008/100}
dynamic_son = {'EURUSD': 0.015/100, 'USDJPY': 0.015/100}

k, pr_k = 12, 12

while True:
    fixar = TestFIXAR(amount, k, pr_k,
                      sashine_eurusd, gyaku_sashine_eurusd,
                      sashine_usdjpy, gyaku_sashine_usdjpy,
                      dynamic_rik, dynamic_son)

    for i in range(100):
        fixar.fixa['EURUSD'].refresh_pricelist(1.08234)
        fixar.fixa['USDJPY'].refresh_pricelist(108.234)

    while True:
        try:
            fixar.login(demo=demo)
            time.sleep(60)

            data_path_base = 'datas/price_diff_list/'
            num_csv = len(glob(data_path_base + '*'))
            file_name = data_path_base + 'pricedifflist' + \
                str(num_csv).zfill(4) + '.csv'

            price_diff_list = []
            error = False
            while not error:
                try:
                    symbol = 'USDJPY' if random.random() > 0.5 else 'EURUSD'
                    side = 'buy' if random.random() > 0.5 else 'sell'
                    fixar.settle_all_position()
                    time.sleep(30)

                    time_diff, price_diff = fixar.test_run(symbol, side)
                    price_diff_list.append(
                        [symbol, side, time_diff, price_diff])
                except Exception as e:
                    print(e)
                    error = True

                if len(price_diff_list) % 10 == 0:
                    pd.DataFrame(price_diff_list).to_csv(
                        file_name, index=False)

                if len(price_diff_list) == 100:
                    fixar.driver_refresh()
                    fixar.login(demo=demo)
                    time.sleep(60)
                elif len(price_diff_list) == 300:
                    break

            pd.DataFrame(price_diff_list).to_csv(file_name, index=False)

        except Exception as e:
            print(e)
            time.sleep(60)
            continue

# %%
