# %%
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome import service as fs
from selenium.webdriver.common.by import By
from selenium import webdriver

from modules import models, modules

from datetime import datetime
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


class FX_Model:
    def __init__(self, symbol, base_m, k, pr_k):
        m_lis = [base_m, base_m*2, base_m*3]
        weight_name = modules.ret_weight_name(symbol=symbol,
                                              k=k,
                                              pr_k=pr_k,
                                              m_lis=m_lis,
                                              y_mode='binary')

        self.model = models.LizaTransformer(k, out_dim=2)
        self.model.load_weights(weight_name + '/best_weights')

    def make_prediction(self, inp_data):
        prediction = self.model.predict(inp_data)[0]

        return prediction[0] > 0.5


class TraderDriver:
    def __init__(self):
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # ユーザープロファイルの保管場所
        PROFILE_PATH: str = (
            "C:\\Users\\User\\AppData\\Local\\Google\\Chrome\\User Data"
        )
        # プロファイルの名前
        PROFILE_DIR: str = "Default"

        options.add_argument(f"user-data-dir={PROFILE_PATH}")
        options.add_argument(f"profile-directory={PROFILE_DIR}")

        chrome_service = fs.Service(executable_path='datas/chromedriver.exe')
        self.driver = webdriver.Chrome(service=chrome_service, options=options)
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

    def make_order(self, amount):
        elements = self.driver.find_elements(
            By.CLASS_NAME, "Button_button__CftuL")
        elements[1].click()
        time.sleep(1)

        amount_box = self.driver.find_element(
            By.CLASS_NAME, 'FormattedNumberInput_input__3uB6c')
        amount_box.clear()
        amount_box.send_keys(str(amount))

        elements = self.driver.find_elements(
            By.CLASS_NAME, "Button_button__CftuL")
        elements[1].click()

    def settle_all_position(self):
        elements = self.driver.find_elements(
            By.CLASS_NAME, "PositionGrid_triggerContainer__1yWG1")
        elements[1].click()
        time.sleep(1)
        elements = self.driver.find_elements(
            By.CLASS_NAME, "Button_buttonLabel__3kVe6")
        elements[1].click()

    def settle_position(self, symbol):
        dx_row = self.driver.find_elements(By.CLASS_NAME, 'dx-row')
        dx_row = [i.text for i in dx_row]

        bool_eurusd, bool_usdjpy = False, False

        for i in dx_row:
            if 'EURUSD' in i:
                bool_eurusd = True
            if 'USDJPY' in i:
                bool_usdjpy = True

        if bool_usdjpy and bool_eurusd:
            button_dic = {'EURUSD': 11,
                          'USDJPY': 13}

            posi = self.driver.find_elements(
                By.CLASS_NAME, "PositionGrid_closeAndMoreOptionsCell__1EeeS")
            posi[button_dic[symbol]].click()
            time.sleep(1)

            elements = self.driver.find_elements(
                By.CLASS_NAME, "Button_buttonLabel__3kVe6")
            elements[1].click()

        else:
            self.settle_all_position()

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
        super().__init__(symbol, 1, 3, 3)
        self.count = 0
        self.position = 0
        self.get_price = 0

        self.price_list = []
        self.rik, self.son = rik, son

    def mono_run(self, price, random_=False):
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
                if random_:
                    up_ = random.random() > 0.5
                else:
                    inp_data = ret_inpdata(self.price_list)
                    up_ = self.make_prediction(inp_data)

                if up_:
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


class FIXAR(TraderDriver):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount
        self.fixa_usdjpy = FIXA('USDJPY', 0.005, 0.1)
        self.fixa_eurusd = FIXA('EURUSD', 0.005/100, 0.1/100)

    def run(self):
        usdjpy, eurusd = self.get_price()

        pr_position_usdjpy = self.fixa_usdjpy.position
        pr_position_eurusd = self.fixa_eurusd.position

        self.fixa_usdjpy.mono_run(usdjpy)
        self.fixa_eurusd.mono_run(eurusd)

        position_usdjpy = self.fixa_usdjpy.position
        position_eurusd = self.fixa_eurusd.position

        if pr_position_usdjpy != position_usdjpy:
            if pr_position_usdjpy != 0:
                self.settle_position('USDJPY')
                time.sleep(5)

            if self.zero_spread():
                if position_usdjpy == 1:
                    self.select_symbol_position('USDJPY', 'buy')
                elif position_usdjpy == -1:
                    self.select_symbol_position('USDJPY', 'sell')
                self.make_order(self.amount)
                time.sleep(5)
            else:
                self.fixa_usdjpy.position = 0
                self.fixa_eurusd.position = 0

        if pr_position_eurusd != position_eurusd:
            if pr_position_eurusd != 0:
                self.settle_position('EURUSD')
                time.sleep(5)

            if self.zero_spread():
                if position_eurusd == 1:
                    self.select_symbol_position('EURUSD', 'buy')
                elif position_eurusd == -1:
                    self.select_symbol_position('EURUSD', 'sell')
                self.make_order(self.amount)
                time.sleep(5)
            else:
                self.fixa_usdjpy.position = 0
                self.fixa_eurusd.position = 0

        if self.fixa_usdjpy.count % 10 == 0:
            self.driver.refresh()


# %%
if __name__ == '__main__':
    fixar = FIXAR(amount=1000)
    error_count = 0
    time.sleep(60)

    while error_count < 3:
        t = time.time()
        try:
            fixar.run()

            print(datetime.now())
            print('USD/JPY-{} position: {}'.format(fixar.fixa_usdjpy.price_list[-1],
                                                   fixar.fixa_usdjpy.position))

            print('EUR/USD-{} position: {}'.format(fixar.fixa_eurusd.price_list[-1],
                                                   fixar.fixa_eurusd.position))

            print('==============================\n')

            error_count = 0
        except Exception as e:
            print(e)
            error_count += 1
            fixar.driver.refresh()

        sleep_time = 60 - (time.time() - t)
        time.sleep(sleep_time)

# %%
