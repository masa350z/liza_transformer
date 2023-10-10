# %%
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import StaleElementReferenceException
# %%
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

driver = webdriver.Chrome(
    executable_path='datas/chromedriver.exe', options=options)
wait = WebDriverWait(driver, 60)

driver.get('https://web.thinktrader.com/web-trader/watchlist')
time.sleep(60)
# %%

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

        # options.add_argument(f"user-data-dir={PROFILE_PATH}")
        # options.add_argument(f"profile-directory={PROFILE_DIR}")

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
        retry = True
        while retry:
            try:
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "BidAskSpread_showPrice__2ijn7")
                spans_usdjpy = elements[0].find_elements(By.TAG_NAME, "span")
                spans_eurusd = elements[2].find_elements(By.TAG_NAME, "span")

                usdjpy = ''
                for i in spans_usdjpy:
                    usdjpy += i.text

                eurusd = ''
                for i in spans_eurusd:
                    eurusd += i.text

            except StaleElementReferenceException:
                continue

        usdjpy = float(usdjpy)
        eurusd = float(eurusd)

        return usdjpy, eurusd

    def idle(self, sleep_time):
        elements = self.driver.find_elements(
            By.CLASS_NAME, "Item_iconContainer__10Rq0")

        elements[0].click()
        time.sleep(sleep_time)


# %%
"""
thinktrader = ThinkTrader()

# %%
thinktrader.driver.get('https://web.thinktrader.com/web-trader/watchlist')

# %%

thinktrader.select_symbol_position('USDJPY', 'buy')
thinktrader.make_order()
# %%
thinktrader.settle_position()
# %%
"""
