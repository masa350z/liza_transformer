# %%
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver

from modules import models, modules

import numpy as np
import random
import time


def ret_inpdata(hist):
    data_x01 = hist[-3:]
    data_x02 = hist[::-1][::2][::-1][-3:]
    data_x03 = hist[::-1][::3][::-1][-3:]

    inp_data = np.stack([data_x01, data_x02, data_x03]).T
    inp_data = np.expand_dims(inp_data, 0)

    return inp_data


def ret_model(symbol, base_m, k, pr_k):
    m_lis = [base_m, base_m*2, base_m*3]
    weight_name = modules.ret_weight_name(symbol=symbol,
                                          k=k,
                                          pr_k=pr_k,
                                          m_lis=m_lis,
                                          y_mode='binary')

    model = models.LizaTransformer(k, out_dim=2)
    model.load_weights(weight_name + '/best_weights')

    return model


class FX_Model:
    def __init__(self, symbol):
        self.model = ret_model(symbol, 1, 3, 3)

    def make_prediction(self, inp_data):
        prediction = self.model.predict(inp_data)[0]

        return prediction[0] > 0.5


class TraderDriver:
    def __init__(self):
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        options.add_argument("--no-sandbox")

        # ユーザープロファイルの保管場所
        PROFILE_PATH: str = (
            "C:\\Users\\User\\AppData\\Local\\Google\\Chrome\\User Data"
        )
        # プロファイルの名前
        PROFILE_DIR: str = "Default"

        options.add_argument(f"user-data-dir={PROFILE_PATH}")
        options.add_argument(f"profile-directory={PROFILE_DIR}")

        self.driver = webdriver.Chrome(
            executable_path='datas/chromedriver.exe', options=options)
        self.wait = WebDriverWait(self.driver, 60)
        self.actions = ActionChains(self.driver)

        self.button_dic = {'USDJPY': {'sell': 0,
                                      'buy': 1},
                           'EURUSD': {'sell': 2,
                                      'buy': 3}}

        self.driver.get('https://web.thinktrader.com/web-trader/watchlist')

    def login(self):
        self.driver.find_element(
            By.ID, "email").send_keys('msum4524@gmail.com')
        self.driver.find_element(By.ID, "password").send_keys('masaumi73177@T')

        spans = self.driver.find_elements(By.TAG_NAME, "span")

        for i in spans:
            if i.text == 'ログイン':
                i.click()

    def select_symbol_position(self, symbol, position):
        retry = True
        while retry:
            try:
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "Item_iconContainer__10Rq0")
                self.actions.move_to_element(elements[0]).perform()

                elements = self.driver.find_elements(
                    By.CLASS_NAME, "BidAskSpread_tickerContainer__jjJnL")
                elements[self.button_dic[symbol][position]].click()

                retry = False

            except StaleElementReferenceException:
                continue

    def make_order(self):
        elements = self.driver.find_elements(
            By.CLASS_NAME, "Button_button__CftuL")
        elements[1].click()
        time.sleep(1)
        elements = self.driver.find_elements(
            By.CLASS_NAME, "Button_button__CftuL")
        elements[1].click()

    def settle_position(self):
        elements = self.driver.find_elements(
            By.CLASS_NAME, "PositionGrid_triggerContainer__1yWG1")
        elements[1].click()
        time.sleep(1)
        elements = self.driver.find_elements(
            By.CLASS_NAME, "Button_buttonLabel__3kVe6")
        elements[1].click()

    def get_price(self):
        retry = 0
        usdjpy, eurusd = 0, 0
        while retry < 3:
            try:
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "BidAskSpread_showPrice__2ijn7")
                spans_usdjpy = elements[0].find_elements(By.TAG_NAME, "span")
                usdjpy = ''
                for i in spans_usdjpy:
                    usdjpy += i.text
                break
            except StaleElementReferenceException:
                retry += 1

        retry = 0
        while retry < 3:
            try:
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "BidAskSpread_showPrice__2ijn7")
                spans_eurusd = elements[2].find_elements(By.TAG_NAME, "span")
                eurusd = ''
                for i in spans_eurusd:
                    eurusd += i.text
                break
            except StaleElementReferenceException:
                retry += 1

        usdjpy = float(usdjpy)
        eurusd = float(eurusd)

        return usdjpy, eurusd

    def zero_spread(self):
        elements = self.driver.find_elements(
            By.CLASS_NAME, "BidAskSpread_spread__21ZFB")

        return sum([float(i.text) for i in elements]) == 0

    def close(self):
        self.driver.close()


class FIXA(FX_Model):
    def __init__(self, symbol, rik, son):
        super().__init__(symbol)
        self.count = 0
        self.position = 0
        self.get_price = 0

        self.price_list = []
        self.rik, self.son = rik, son

    def mono_run(self, price):
        self.price_list.append(price)
        self.price_list = self.price_list[-10:]

        if self.position != 0:
            diff = price - self.get_price

            if diff*self.position > self.rik:
                self.position = 0
            elif diff*self.position < -self.son:
                self.position = 0

        if self.position == 0:
            if len(self.price_list) >= 9:
                inp_data = ret_inpdata(self.price_list)
                if self.make_prediction(inp_data):
                    self.position = 1
                else:
                    self.position = -1
            else:
                if random.random() > 0.5:
                    self.position = 1
                else:
                    self.position = -1

            self.get_price = price
        self.count += 1

    def random_run(self, price):
        self.price_list.append(price)
        self.price_list = self.price_list[-10:]

        if self.position != 0:
            diff = price - self.get_price

            if diff*self.position > self.rik:
                self.position = 0
            elif diff*self.position < -self.son:
                self.position = 0

        if self.position == 0:
            if random.random() > 0.5:
                self.position = 1
            else:
                self.position = -1
            self.get_price = price

        self.count += 1


# %%
price_list = []
traderdriver = TraderDriver()
fixa_usdjpy = FIXA('USDJPY', 0.005, 0.1)
time.sleep(100)
# %%
error_count = 0

while error_count < 3:
    t = time.time()
    try:
        usdjpy, eurusd = traderdriver.get_price()

        pr_position_usdjpy = fixa_usdjpy.position
        fixa_usdjpy.mono_run(usdjpy)
        position_usdjpy = fixa_usdjpy.position

        if pr_position_usdjpy != position_usdjpy:
            if pr_position_usdjpy != 0:
                traderdriver.settle_position()
                time.sleep(1)

            if position_usdjpy == 1:
                traderdriver.select_symbol_position('USDJPY', 'buy')
            elif position_usdjpy == -1:
                traderdriver.select_symbol_position('USDJPY', 'sell')
            traderdriver.make_order()
            time.sleep(1)

        if fixa_usdjpy.count % 10 == 0:
            traderdriver.driver.refresh()

        error_count = 0

    except Exception as e:
        print(e)
        error_count += 1
        traderdriver.driver.refresh()

    sleep_time = 60 - (time.time() - t)
    time.sleep(sleep_time)
# %%
