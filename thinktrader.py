# %%
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver
import numpy as np
from modules import models, modules
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

# %%


class FX_Model:
    def __init__(self):
        self.model_usdjpy = ret_model('USDJPY', 1, 3, 3)
        self.model_eurusd = ret_model('EURUSD', 1, 3, 3)

    def make_prediction(self, symbol, inp_data):
        if symbol == 'USDJPY':
            prediction = self.model_usdjpy.predict(inp_data)[0]
        elif symbol == 'EURUSD':
            prediction = self.model_eurusd.predict(inp_data)[0]

        return prediction[0] > 0.5


class ThinkTrader(FX_Model):
    def __init__(self):
        super().__init__()

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

    def idle(self, sleep_time=0):
        elements = self.driver.find_elements(
            By.CLASS_NAME, "Item_iconContainer__10Rq0")

        elements[0].click()
        time.sleep(sleep_time)

    def zero_spread(self):
        elements = self.driver.find_elements(
            By.CLASS_NAME, "BidAskSpread_spread__21ZFB")

        return sum([float(i.text) for i in elements]) == 0


# %%
price_list = []
thinktrader = ThinkTrader()
thinktrader.driver.get('https://web.thinktrader.com/web-trader/watchlist')
time.sleep(100)
# %%
count = 0
position = 0
get_price = 0
rik, son = 0.005, 0.1
error_count = 0

usdjpy_price_list = []
eurusd_price_list = []

while error_count < 3:
    t = time.time()
    try:
        usdjpy, eurusd = thinktrader.get_price()
        usdjpy_price_list.append(usdjpy)
        eurusd_price_list.append(eurusd)

        usdjpy_price_list = usdjpy_price_list[-10:]
        eurusd_price_list = eurusd_price_list[-10:]

        if position == 0:
            if random.random() > 0.5:
                position = 1
                thinktrader.select_symbol_position('USDJPY', 'buy')
            else:
                position = -1
                thinktrader.select_symbol_position('USDJPY', 'sell')
            thinktrader.make_order()
            get_price = usdjpy

        else:
            diff = usdjpy - get_price
            if diff*position > rik:
                thinktrader.settle_position()
                position = 0
            elif diff*position < -son:
                thinktrader.settle_position()
                position = 0

        if count % 10 == 0:
            thinktrader.driver.refresh()
        count += 1

        error_count = 0

    except Exception as e:
        print(e)
        error_count += 1
        thinktrader.driver.refresh()

    sleep_time = 60 - (time.time() - t)
    time.sleep(sleep_time)
# %%
