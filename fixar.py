# %%
import subprocess
import pandas as pd
from selenium.common.exceptions import StaleElementReferenceException, WebDriverException, NoSuchElementException, \
    ElementClickInterceptedException, NoSuchWindowException, InvalidSessionIdException
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome import service as fs
from selenium.webdriver.common.by import By
from selenium import webdriver

import undetected_chromedriver as uc

from modules import models, modules
import line

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


def launch_chrome(chrome_port):
    subprocess.run(["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                    '-remote-debugging-port={}'.format(chrome_port),
                    '--user-data-dir=C:\\Users\\ai-so\\local_program\\liza_transformer\\fixar_data'])


class ToPageRefreshError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


class ToDriverRefreshError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


class HumanChallengeError(Exception):
    def __init__(self):
        self.message = 'human challenge needed'
        super().__init__(self.message)


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
        prediction = self.model.predict(inp_data, verbose=0)[0]

        return prediction[0] > 0.5


class FIXA(FX_Model):
    def __init__(self, symbol, sashine, gyakusashine):
        super().__init__(symbol, 1, 3, 3)
        self.count = 0
        self.position = 0
        self.get_price = 0

        self.price_list = []
        self.sahine, self.gyakusashine = sashine, gyakusashine

    def refresh_pricelist(self, price):
        self.price_list.append(price)
        self.price_list = self.price_list[-10:]

    def ret_prediction(self):
        if len(self.price_list) < 9:
            up_ = random.random() > 0.5
        else:
            inp_data = ret_inpdata(self.price_list)
            up_ = self.make_prediction(inp_data)

        return up_

    def refresh_position(self, price, random_=False):
        self.refresh_position(price)

        if self.position != 0:
            diff = price - self.get_price

            if diff*self.position > self.sahine:
                self.position = 0
            elif diff*self.position < -self.gyakusashine:
                self.position = 0

        if self.position == 0:
            if len(self.price_list) >= 9:
                if random_:
                    up_ = random.random() > 0.5
                else:
                    up_ = self.ret_prediction()

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


class TraderDriver:
    def __init__(self):
        # options = webdriver.ChromeOptions()
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        # options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])

        options.headless = False

        # ユーザープロファイルの保管場所
        PROFILE_PATH: str = (
            "C:\\Users\\User\\AppData\\Local\\Google\\Chrome\\User Data"
        )
        # プロファイルの名前
        PROFILE_DIR: str = "FIXAR"

        # options.add_argument(f"user-data-dir={PROFILE_PATH}")
        # options.add_argument(f"profile-directory={PROFILE_DIR}")
        # options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36")

        chrome_service = fs.Service(
            executable_path=ChromeDriverManager().install())
        # self.driver = webdriver.Chrome(service=chrome_service, options=options)
        self.driver = uc.Chrome(service=chrome_service, options=options)
        self.wait = WebDriverWait(self.driver, 60)
        self.actions = ActionChains(self.driver)

        self.button_dic = {'USDJPY': {'sell': 0, 'buy': 1},
                           'EURUSD': {'sell': 2, 'buy': 3}}

    def login(self):
        self.driver.get('https://web.thinktrader.com/account/login')
        time.sleep(3)
        try:
            if self.driver.find_element(By.ID, "email"):
                self.driver.find_element(By.ID, "email").send_keys('a')
                self.driver.find_element(By.ID, "email").clear()
                self.driver.find_element(
                    By.ID, "email").send_keys('msum4524@gmail.com')

                self.driver.find_element(By.ID, "password").send_keys('a')
                self.driver.find_element(By.ID, "password").clear()
                self.driver.find_element(
                    By.ID, "password").send_keys('masaumi73177@T')

                spans = self.driver.find_elements(By.TAG_NAME, "span")

                for i in spans:
                    if i.text == 'ログイン':
                        i.click()

                iframe = self.driver.find_elements(By.TAG_NAME, "iframe")
                for i in iframe:
                    if '<iframe src=' in i.get_attribute('outerHTML'):
                        iframe = i
                        break
                time.sleep(1)
                if 'relative; width:' in iframe.get_attribute('outerHTML'):
                    raise HumanChallengeError()
            else:
                pass

        except NoSuchElementException:
            pass

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

    def make_order(self, symbol, position, amount):
        try:
            self.select_symbol_position(symbol, position)

            amount_box = self.driver.find_element(
                By.CLASS_NAME, 'FormattedNumberInput_input__3uB6c')
            amount_box.clear()
            amount_box.send_keys(str(amount))

            elements = self.driver.find_elements(
                By.CLASS_NAME, "Button_button__CftuL")
            elements[1].click()
            time.sleep(1)

            elements = self.driver.find_elements(
                By.CLASS_NAME, "Button_button__CftuL")
            elements[1].click()

        except ElementClickInterceptedException as e:
            print('Error occured make_nariyuki_order \n{}'.format(e))
            raise ToPageRefreshError(e)

        except Exception as e:
            print('Error occured make_order \n{}'.format(e))

    def make_oneclick_order(self, symbol, position):
        try:
            order_buttons = self.driver.find_elements(
                By.CLASS_NAME, 'OneClickTradeButtons_tradeButtonContainer__3Z-Xe')
            sell_usdjpy, buy_usdjpy, sell_eurusd, buy_eurusd = order_buttons

            order_buttons = {'EURUSD': [buy_eurusd, sell_eurusd],
                             'USDJPY': [buy_usdjpy, sell_usdjpy]}

            buy_sell = 0 if position == 'buy' else 1

            order_buttons[symbol][buy_sell].click()

        except ElementClickInterceptedException as e:
            print('Error occured make_nariyuki_order \n{}'.format(e))
            raise ToPageRefreshError(e)

        except Exception as e:
            print('Error occured make_oneclick_order \n{}'.format(e))

    def make_sashine_order(self, symbol, position, amount,
                           rate, sashine, gyaku_sashine):
        try:
            self.select_symbol_position(symbol, position)
            time.sleep(1)

            tradeticket_container = self.driver.find_element(
                By.CLASS_NAME, 'TradeTicket_container__2S2h7')

            ticket_dropdown = tradeticket_container.find_element(
                By.CLASS_NAME, 'TradeTicket_dropdownMenu__3Zdn-')
            ticket_dropdown.click()
            tradeticket_container.find_elements(
                By.CLASS_NAME, 'item')[1].click()

            tradeticket_container.find_elements(
                By.CLASS_NAME, 'checkbox')[0].click()
            tradeticket_container.find_elements(
                By.CLASS_NAME, 'checkbox')[1].click()

            tradeticket_container = self.driver.find_element(
                By.CLASS_NAME, 'TradeTicket_container__2S2h7')

            input_ = tradeticket_container.find_elements(By.TAG_NAME, 'input')

            rate_inp = input_[2]
            amount_inp = input_[3]
            sashine_inp = input_[5]
            gyaku_sashine_inp = input_[7]

            rate_inp.clear()
            rate_inp.send_keys(rate)

            amount_inp.clear()
            amount_inp.send_keys(amount)

            sashine_inp.clear()
            sashine_inp.send_keys(sashine)

            gyaku_sashine_inp.clear()
            gyaku_sashine_inp.send_keys(gyaku_sashine)

            for _ in range(2):
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "Button_button__CftuL")
                elements[1].click()
                time.sleep(1)

        except ElementClickInterceptedException as e:
            print('Error occured make_nariyuki_order \n{}'.format(e))
            raise ToPageRefreshError(e)

        except Exception as e:
            print('Error occured make_sashine_order \n{}'.format(e))

    def make_nariyuki_order(self, symbol, position, amount,
                            sashine, gyaku_sashine):
        try:
            self.select_symbol_position(symbol, position)
            time.sleep(1)

            tradeticket_container = self.driver.find_element(
                By.CLASS_NAME, 'TradeTicket_container__2S2h7')

            tradeticket_container.find_elements(
                By.CLASS_NAME, 'checkbox')[0].click()
            tradeticket_container.find_elements(
                By.CLASS_NAME, 'checkbox')[1].click()

            tradeticket_container = self.driver.find_element(
                By.CLASS_NAME, 'TradeTicket_container__2S2h7')

            input_ = tradeticket_container.find_elements(By.TAG_NAME, 'input')

            amount_inp = input_[2]
            sashine_inp = input_[4]
            gyaku_sashine_inp = input_[6]

            amount_inp.clear()
            amount_inp.send_keys(amount)

            sashine_inp.clear()
            sashine_inp.send_keys(sashine)

            gyaku_sashine_inp.clear()
            gyaku_sashine_inp.send_keys(gyaku_sashine)

            for _ in range(2):
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "Button_button__CftuL")
                elements[1].click()
                time.sleep(1)

        except ElementClickInterceptedException as e:
            print('Error occured make_nariyuki_order \n{}'.format(e))
            raise ToPageRefreshError(e)

        except Exception as e:
            print('Error occured make_nariyuki_order \n{}'.format(e))
            raise ToPageRefreshError(e)

    def settle_all_position(self):
        elements = self.driver.find_elements(
            By.CLASS_NAME, "PositionGrid_triggerContainer__1yWG1")
        elements[1].click()
        time.sleep(1)
        elements = self.driver.find_elements(
            By.CLASS_NAME, "Button_buttonLabel__3kVe6")
        elements[1].click()

    def settle_position(self, symbol):
        try:
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

        except StaleElementReferenceException as e:
            raise ToPageRefreshError(e)

    def get_price(self):
        retry = 0
        usdjpy, eurusd = 0, 0
        while retry < 3:
            try:
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "BidAskSpread_showPrice__2ijn7")
                try:
                    spans_usdjpy = elements[0].find_elements(
                        By.TAG_NAME, "span")
                except IndexError as e:
                    raise ToPageRefreshError(e)
                usdjpy = ''
                for i in spans_usdjpy:
                    usdjpy += i.text
                break
            except (StaleElementReferenceException, NoSuchElementException):
                retry += 1

        retry = 0
        while retry < 3:
            try:
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "BidAskSpread_showPrice__2ijn7")
                try:
                    spans_eurusd = elements[2].find_elements(
                        By.TAG_NAME, "span")
                except IndexError as e:
                    raise ToPageRefreshError(e)
                eurusd = ''
                for i in spans_eurusd:
                    eurusd += i.text
                break
            except (StaleElementReferenceException, NoSuchElementException):
                retry += 1

        usdjpy = float(usdjpy)
        eurusd = float(eurusd)

        return usdjpy, eurusd

    def get_price_oneclick(self):
        retry = 0
        usdjpy, eurusd = 0, 0
        while retry < 3:
            try:
                order_buttons = self.driver.find_elements(
                    By.CLASS_NAME, 'OneClickTradeButtons_tradeButtonContainer__3Z-Xe')

                sell_usdjpy, buy_usdjpy, sell_eurusd, buy_eurusd = order_buttons

                usdjpy = float(buy_usdjpy.text.split('\n')[1])
                eurusd = float(buy_eurusd.text.split('\n')[1])
                break
            except StaleElementReferenceException:
                retry += 1

        usdjpy = float(usdjpy)
        eurusd = float(eurusd)

        return usdjpy, eurusd

    def zero_spread(self):
        try:
            elements = self.driver.find_elements(
                By.CLASS_NAME, "BidAskSpread_spread__21ZFB")

            return sum([float(i.text) for i in elements]) == 0

        except Exception as e:
            print('Error Occured zero_spread \n{}'.format(e))
            return False

    def position_bool(self):
        get_price_eurusd = 0
        now_price_eurusd = 0

        get_price_usdjpy = 0
        now_price_usdjpy = 0

        position_eurusd, position_usdjpy = 0, 0
        try:
            dx_row = self.driver.find_elements(By.CLASS_NAME, 'dx-row')
            dx_row = [i.text for i in dx_row]

            bool_eurusd = 'EURUSD' in dx_row
            bool_usdjpy = 'USDJPY' in dx_row

            if bool_eurusd and bool_usdjpy:
                get_price_eurusd = float(dx_row[2].split('\n')[3])
                now_price_eurusd = float(dx_row[2].split('\n')[5])

                get_price_usdjpy = float(dx_row[3].split('\n')[3])
                now_price_usdjpy = float(dx_row[3].split('\n')[5])

                position_eurusd = dx_row[2].split('\n')[0]
                position_usdjpy = dx_row[3].split('\n')[0]

                position_eurusd = 1 if position_eurusd == '買い' else -1
                position_usdjpy = 1 if position_usdjpy == '買い' else -1

            elif bool_eurusd:
                get_price_eurusd = float(dx_row[2].split('\n')[3])
                now_price_eurusd = float(dx_row[2].split('\n')[5])

                position_eurusd = dx_row[2].split('\n')[0]
                position_eurusd = 1 if position_eurusd == '買い' else -1

            elif bool_usdjpy:
                get_price_usdjpy = float(dx_row[2].split('\n')[3])
                now_price_usdjpy = float(dx_row[2].split('\n')[5])

                position_usdjpy = dx_row[2].split('\n')[0]
                position_usdjpy = 1 if position_usdjpy == '買い' else -1

            return [bool_eurusd, position_eurusd, get_price_eurusd, now_price_eurusd], \
                [bool_usdjpy, position_usdjpy, get_price_usdjpy, now_price_usdjpy]
        except WebDriverException as e:
            print('Error Occured position_bool \n{}'.format(e))
            raise ToDriverRefreshError(e)

    def set_sashine_p2(self, bool_eurusd, bool_usdjpy):
        tbody = self.driver.find_elements(By.TAG_NAME, 'tbody')
        td_ = tbody[-2].find_elements(By.TAG_NAME, 'td')

        get_eurusd = float(td_[4].text)
        get_usdjpy = float(td_[16].text)

        position_eurusd = td_[4-3].text
        position_usdjpy = td_[16-3].text

        status_list = [[get_eurusd, position_eurusd],
                       [get_usdjpy, position_usdjpy]]

        for i in range(2):
            bool_ = bool_eurusd if i == 0 else bool_usdjpy
            symbol = 'EURUSD' if i == 0 else 'USDJPY'
            if not bool_:
                tbody = self.driver.find_elements(By.TAG_NAME, 'tbody')
                trigger = tbody[-1].find_elements(
                    By.CLASS_NAME, 'PositionGrid_triggerContainer__1yWG1')

                if i == 0:
                    trigger[0].click()
                else:
                    trigger[2].click()
                sashine_gyaku = self.driver.find_elements(
                    By.CLASS_NAME, 'PositionGrid_PopupContainer__3AWXo')[-1]
                sashine_gyaku.click()

                rik = self.fixa[symbol].rik
                son = self.fixa[symbol].son

                if status_list[i][1] == '買い':
                    sashine = status_list[i][0] + rik
                    gyaku_sashine = status_list[i][0] - son
                else:
                    sashine = status_list[i][0] - rik
                    gyaku_sashine = status_list[i][0] + son

                self.submit_sashine(sashine, gyaku_sashine)
                time.sleep(3)

    def set_sashine_p1(self, symbol):
        tbody = self.driver.find_elements(By.TAG_NAME, 'tbody')
        td_ = tbody[-2].find_elements(By.TAG_NAME, 'td')

        get_ = float(td_[4].text)
        position_ = td_[4-3].text

        trigger = tbody[-1].find_elements(
            By.CLASS_NAME, 'PositionGrid_triggerContainer__1yWG1')
        trigger[0].click()

        sashine_gyaku = self.driver.find_elements(
            By.CLASS_NAME, 'PositionGrid_PopupContainer__3AWXo')[-1]
        sashine_gyaku.click()

        rik = self.fixa[symbol].rik
        son = self.fixa[symbol].son

        if position_ == '買い':
            sashine = get_ + rik
            gyaku_sashine = get_ - son
        else:
            sashine = get_ - rik
            gyaku_sashine = get_ + son

        self.submit_sashine(sashine, gyaku_sashine)

    def submit_sashine(self, sashine, gyaku_sashine):
        tradeticket_container = self.driver.find_element(
            By.CLASS_NAME, 'TradeTicket_container__2S2h7')
        inpts_ = tradeticket_container.find_elements(By.TAG_NAME, 'input')

        inpts_[1].clear()
        inpts_[1].send_keys(sashine)

        inpts_[3].clear()
        inpts_[3].send_keys(gyaku_sashine)

        tradeticket_container = self.driver.find_element(
            By.CLASS_NAME, 'TradeTicket_container__2S2h7')
        button_ = tradeticket_container.find_elements(
            By.CLASS_NAME, 'Button_button__CftuL')

        button_[1].click()


class NTraderDriver(TraderDriver):
    def __init__(self, chrome_port=9222):
        launch_chrome(chrome_port)

        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36")
        options.add_experimental_option(
            "debuggerAddress", "127.0.0.1:{}".format(chrome_port))

        chrome_service = fs.Service(
            executable_path=ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=chrome_service, options=options)
        self.wait = WebDriverWait(self.driver, 60)
        self.actions = ActionChains(self.driver)

        self.button_dic = {'USDJPY': {'sell': 0, 'buy': 1},
                           'EURUSD': {'sell': 2, 'buy': 3}}

        self.driver.get('https://web.thinktrader.com/web-trader/watchlist')


class FIXAR(TraderDriver):
    def __init__(self, amount,
                 sashine_eurusd, gyaku_sashine_eurusd,
                 sashine_usdjpy, gyaku_sashine_usdjpy):
        super().__init__()
        self.amount = amount
        self.fixa = {'EURUSD': FIXA('EURUSD', sashine_eurusd, gyaku_sashine_eurusd),
                     'USDJPY': FIXA('USDJPY', sashine_usdjpy, gyaku_sashine_usdjpy)}

    def run(self):
        usdjpy, eurusd = self.get_price()
        price = {'EURUSD': eurusd,
                 'USDJPY': usdjpy}

        # for symbol in ['EURUSD', 'USDJPY']:
        for symbol in ['USDJPY']:
            pr_position = self.fixa[symbol].position
            self.fixa[symbol].refresh_position(price[symbol])
            new_position = self.fixa[symbol].position

            if pr_position != new_position:
                if self.zero_spread():
                    if pr_position != 0:
                        self.settle_position(symbol)
                        time.sleep(5)

                    if new_position == 1:
                        self.make_order(symbol, 'buy', self.amount)
                    elif new_position == -1:
                        self.make_order(symbol, 'sell', self.amount)
                    time.sleep(5)
                else:
                    self.fixa[symbol].position = pr_position

        if self.fixa['USDJPY'].count % 10 == 0:
            # self.driver_refresh()
            self.driver.refresh()

    def run_oneclick(self):
        for symbol in ['EURUSD', 'USDJPY']:
            usdjpy, eurusd = self.get_price_oneclick()
            price = {'EURUSD': eurusd,
                     'USDJPY': usdjpy}

            pr_position = self.fixa[symbol].position
            self.fixa[symbol].refresh_position(price[symbol])
            new_position = self.fixa[symbol].position

            if pr_position != new_position:
                if pr_position != 0:
                    self.settle_position(symbol)

                if new_position == 1:
                    self.make_oneclick_order(symbol, 'buy')
                elif new_position == -1:
                    self.make_oneclick_order(symbol, 'sell')

        if self.fixa['USDJPY'].count % 5 == 0:
            self.driver.refresh()

    def ret_df(self):
        df = []
        position_list = ['NONE', 'BUY', 'SELL']
        for symbol in ['EURUSD', 'USDJPY']:
            price = self.fixa[symbol].price_list[-1]
            position = self.fixa[symbol].position
            position_str = position_list[position]
            get_price = self.fixa[symbol].get_price
            price_difference = (price - get_price)*position

            df.append([symbol, position_str, get_price,
                      price, price_difference])

        df = pd.DataFrame(
            df, columns=['symbol', 'side', 'root',  'price', 'diff'])

        return df

    def driver_refresh(self):
        try:
            self.driver.close()
        except (NoSuchWindowException, InvalidSessionIdException):
            pass

        super().__init__()


class FIXAR_V2(FIXAR):
    def __init__(self, amount,
                 sashine_eurusd, gyaku_sashine_eurusd,
                 sashine_usdjpy, gyaku_sashine_usdjpy,
                 dynamic_rik, dynamic_son):
        super().__init__(amount,
                         sashine_eurusd, gyaku_sashine_eurusd,
                         sashine_usdjpy, gyaku_sashine_usdjpy)

        self.dynamic_rik, self.dynamic_son = dynamic_rik, dynamic_son

    def ret_pricedic(self):
        usdjpy, eurusd = self.get_price()
        price_dic = {'EURUSD': eurusd,
                     'USDJPY': usdjpy}

        return price_dic

    def run(self):
        try:
            recconect_button = self.driver.find_elements(
                By.CLASS_NAME, 'ReconnectModal_button__1DjYD')
        except WebDriverException as e:
            raise ToDriverRefreshError(e)

        if len(recconect_button) > 0:
            raise ToPageRefreshError('recconect')
        else:
            for i in range(2):
                symbol = 'EURUSD' if i == 0 else 'USDJPY'

                position_bool = self.position_bool()
                self.fixa[symbol].refresh_pricelist(
                    self.ret_pricedic()[symbol])

                if position_bool[i][0]:
                    get_price = position_bool[i][2]
                    now_price = position_bool[i][3]
                    position = position_bool[i][1]
                    price_differ = (now_price - get_price)*position

                    if price_differ > self.dynamic_rik[symbol] or \
                            price_differ < -self.dynamic_son[symbol]:
                        self.settle_position(symbol)
                    time.sleep(3)

                else:
                    if self.zero_spread():
                        rate = self.ret_pricedic()[symbol]
                        pred = self.fixa[symbol].ret_prediction()

                        side = 'buy' if pred else 'sell'
                        sashine = rate + \
                            (self.fixa[symbol].sahine)*(1 if pred else -1)
                        gyaku_sashine = rate - \
                            (self.fixa[symbol].gyakusashine) * \
                            (1 if pred else -1)

                        self.make_nariyuki_order(symbol, side,
                                                 self.amount,
                                                 sashine, gyaku_sashine)
                        time.sleep(3)


# %%
amount = 1000

sashine_eurusd, gyaku_sashine_eurusd = round(0.06/150, 5), round(0.12/150, 5)
sashine_usdjpy, gyaku_sashine_usdjpy = 0.06, 0.12

dynamic_rik = {'EURUSD': 0.005/100, 'USDJPY': 0.005}
dynamic_son = {'EURUSD': round(0.1/150, 5), 'USDJPY': 0.1}

fixar = FIXAR_V2(amount,
                 sashine_eurusd, gyaku_sashine_eurusd,
                 sashine_usdjpy, gyaku_sashine_usdjpy,
                 dynamic_rik, dynamic_son)
# %%
# nt = NTraderDriver()
# %%
# nt.login()
# %%
# nt.select_symbol_position('USDJPY', 'buy')
# %%
try:
    fixar.login()
except HumanChallengeError as e:
    print(e)
time.sleep(30)
# %%
count = 0
while True:
    t = time.time()
    try:
        fixar.run()

    except ToPageRefreshError as e:
        print(e)
        fixar.driver.refresh()
        try:
            fixar.login()
        except HumanChallengeError as e:
            print(e)
            line.send_to_masaumi('human challenge needed \n{}'.format(e))
            break

    except ToDriverRefreshError as e:
        print(e)
        fixar.driver_refresh()
        try:
            fixar.login()
        except HumanChallengeError as e:
            print(e)
            line.send_to_masaumi('human challenge needed \n{}'.format(e))
            break

    except Exception as e:
        print(e)
        line.send_to_masaumi('fatal error occuered \n{}'.format(e))
        break

    count += 1
    print('{}\n{}\n__________\n'.format(count, datetime.now()))
    # if count % 30 == 0:
    #    fixar.login()

    # elif count % 5 == 0:
    #    fixar.driver.refresh()

    sleep_time = 60 - (time.time() - t)
    sleep_time = sleep_time if sleep_time > 0 else 0
    time.sleep(sleep_time)

# %%
