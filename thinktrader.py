# %%
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver
import random
import time

# %%


class ThinkTrader:
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
while True:
    t = time.time()
    usdjpy, eurusd = thinktrader.get_price()

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

    sleep_time = 60 - (time.time() - t)

    time.sleep(sleep_time)
# %%