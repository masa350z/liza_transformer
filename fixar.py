# %%
from selenium.common.exceptions import StaleElementReferenceException, \
    WebDriverException, NoSuchElementException, \
    ElementClickInterceptedException, NoSuchWindowException, \
    InvalidSessionIdException
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome import service as fs
from selenium.webdriver.common.by import By

import undetected_chromedriver as uc

from modules import models, modules
import line

from datetime import datetime
import numpy as np
import random
import time


def ret_inpdata(hist, k):
    data_x01 = hist[-k:]
    data_x02 = hist[::-1][::2][::-1][-k:]
    data_x03 = hist[::-1][::3][::-1][-k:]

    inp_data = np.stack([data_x01, data_x02, data_x03]).T
    inp_data = np.expand_dims(inp_data, 0)

    return inp_data


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
    def __init__(self, symbol, sashine, gyakusashine,
                 k, pr_k):
        super().__init__(symbol, 1, k, pr_k)
        self.k, self.pr_k = k, pr_k

        self.price_list = []
        self.sahine, self.gyakusashine = sashine, gyakusashine
        self.symbol_get_price = None

    def refresh_pricelist(self, price):
        self.price_list.append(price)
        self.price_list = self.price_list[-(3*self.k+1):]

    def ret_prediction(self):
        if len(self.price_list) < 3*self.k:
            up_ = random.random() > 0.5
        else:
            inp_data = ret_inpdata(self.price_list, k)
            up_ = self.make_prediction(inp_data)

        return up_


class TraderDriver:
    def __init__(self):
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_experimental_option("prefs", {
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
            'translate': {'enabled': False}
        })

        options.headless = False

        chrome_service = fs.Service(
            executable_path=ChromeDriverManager().install())

        self.driver = uc.Chrome(service=chrome_service, options=options)
        self.wait = WebDriverWait(self.driver, 60)
        self.actions = ActionChains(self.driver)

        self.button_dic = {'USDJPY': {'sell': 0, 'buy': 1},
                           'EURUSD': {'sell': 2, 'buy': 3}}

    def login(self):
        if not self.driver.current_url\
                == 'https://web.thinktrader.com/account/login':

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
                # if 'relative; width:' in iframe.get_attribute('outerHTML'):
                #    raise HumanChallengeError()
            else:
                pass

        except NoSuchElementException:
            pass

    def select_symbol_position(self, symbol, position):
        error_count = 0
        while error_count < 3:
            try:
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "Item_iconContainer__10Rq0")
                self.actions.move_to_element(elements[0]).perform()

                elements = self.driver.find_elements(
                    By.CLASS_NAME, "BidAskSpread_tickerContainer__jjJnL")
                elements[self.button_dic[symbol][position]].click()

                error_count = 100

            except (StaleElementReferenceException, IndexError):
                # 要素がない場合。読み込みが完了していないことが多い
                error_count += 1
                time.sleep(3)
                continue

            except ElementClickInterceptedException:
                # 資産が不足していますウィンドウが出る場合
                time.sleep(60)
                ToPageRefreshError()
                break

        if error_count == 3:
            raise ToDriverRefreshError()

    def click_navbar01(self):
        error_count = 0
        while error_count < 3:
            try:
                rightnav = self.driver.find_element(
                    By.CLASS_NAME, 'Dashboard_rightNav__tUh0b')
                rightnav.find_elements(By.CLASS_NAME, 'Item_iconContainer__10Rq0')[
                    0].click()

            except (StaleElementReferenceException, IndexError):
                # 要素がない場合。読み込みが完了していないことが多い
                error_count += 1
                time.sleep(3)
                continue

            except ElementClickInterceptedException:
                # 資産が不足していますウィンドウが出る場合
                time.sleep(60)
                ToPageRefreshError()
                break

        if error_count == 3:
            raise ToDriverRefreshError()

    def make_order(self, symbol, position, amount,
                   sashine, gyaku_sashine):
        # try:
        self.select_symbol_position(symbol, position)
        time.sleep(1)

        error_count = 0
        while error_count < 3:
            try:
                tradeticket_container = self.driver.find_element(
                    By.CLASS_NAME, 'TradeTicket_container__2S2h7')
                error_count = 100

            except NoSuchElementException as e:
                print(e)
                error_count += 1
                self.click_navbar01()
                self.select_symbol_position(symbol, position)
                time.sleep(1)
                continue

        if error_count == 3:
            raise ToPageRefreshError()
        else:
            try:
                tradeticket_container.find_elements(
                    By.CLASS_NAME, 'checkbox')[0].click()
                tradeticket_container.find_elements(
                    By.CLASS_NAME, 'checkbox')[1].click()

                tradeticket_container = self.driver.find_element(
                    By.CLASS_NAME, 'TradeTicket_container__2S2h7')

                input_ = tradeticket_container.find_elements(
                    By.TAG_NAME, 'input')

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
            except IndexError as e:
                print(e)
                raise ToPageRefreshError(e)
        """
        except ElementClickInterceptedException as e:
            print('Error occured make_order \n{}'.format(e))
            raise ToPageRefreshError(e)

        except Exception as e:
            print('Error occured make_order \n{}'.format(e))
            raise ToDriverRefreshError(e)
        """

    def settle_all_position(self):
        error_count = 0
        while error_count < 3:
            try:
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "PositionGrid_triggerContainer__1yWG1")
                elements[1].click()
                time.sleep(1)
                elements = self.driver.find_elements(
                    By.CLASS_NAME, "Button_buttonLabel__3kVe6")
                elements[1].click()

            except StaleElementReferenceException as e:
                print(e)
                print('settle_all_position')
                error_count += 1
                time.sleep(3)
                continue

            except IndexError as e:
                print(e)
                print('settle_all_position')
                error_count = 100

        if error_count == 3:
            raise ToPageRefreshError()

    def settle_position(self, symbol):
        error_count = 0
        while error_count < 3:
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
                        By.CLASS_NAME,
                        "PositionGrid_closeAndMoreOptionsCell__1EeeS")
                    posi[button_dic[symbol]].click()
                    time.sleep(1)

                    elements = self.driver.find_elements(
                        By.CLASS_NAME, "Button_buttonLabel__3kVe6")
                    elements[1].click()

                else:
                    self.settle_all_position()

                error_count = 100

            except StaleElementReferenceException as e:
                print(e)
                print('settle_position')
                error_count += 1
                time.sleep(3)
                continue

        if error_count == 3:
            raise ToPageRefreshError()

    def get_price(self):
        error_count = 0
        usdjpy, eurusd = 0, 0
        while error_count < 3:
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
            except (StaleElementReferenceException,
                    NoSuchElementException):
                error_count += 1
                time.sleep(1)

        error_count = 0
        while error_count < 3:
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
            except (StaleElementReferenceException,
                    NoSuchElementException):
                error_count += 1
                time.sleep(1)

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

    def check_position_box(self):
        error_count = 0
        while error_count < 3:
            try:
                dx_row = self.driver.find_elements(By.CLASS_NAME, 'dx-row')
                dx_row = [i.text for i in dx_row]
                error_count = 100
            except StaleElementReferenceException as e:
                print('position_bool')
                print(e)
                error_count += 1
                time.sleep(1)
                continue

        if error_count == 3:
            raise ToPageRefreshError()

        return dx_row

    def get_double_position_status(self, dx_row):
        """
        両symbolのポジションを持つ場合のポジションの状態を返す。
        """
        get_price_eurusd = float(dx_row[2].split('\n')[3])
        now_price_eurusd = float(dx_row[2].split('\n')[5])

        get_price_usdjpy = float(dx_row[3].split('\n')[3])
        now_price_usdjpy = float(dx_row[3].split('\n')[5])

        position_eurusd = dx_row[2].split('\n')[0]
        position_usdjpy = dx_row[3].split('\n')[0]

        position_eurusd = 1 if position_eurusd == '買い' else -1
        position_usdjpy = 1 if position_usdjpy == '買い' else -1

        return get_price_eurusd, now_price_eurusd, position_eurusd, \
            get_price_usdjpy, now_price_usdjpy, position_usdjpy

    def get_single_position_status(self, dx_row):
        """
        片symbolのポジションを持つ場合のポジションの状態を返す。
        """
        get_price = float(dx_row[2].split('\n')[3])
        now_price = float(dx_row[2].split('\n')[5])

        position = dx_row[2].split('\n')[0]
        position = 1 if position == '買い' else -1

        return get_price, now_price, position

    def position_bool(self):
        """
        現在の全ポジションの状態を返す関数
        """
        get_price_eurusd, now_price_eurusd = 0, 0
        get_price_usdjpy, now_price_usdjpy = 0, 0
        position_eurusd, position_usdjpy = 0, 0

        dx_row = self.check_position_box()

        bool_eurusd = 'EURUSD' in dx_row
        bool_usdjpy = 'USDJPY' in dx_row

        if bool_eurusd and bool_usdjpy:  # 両symbolがポジションを持つ場合
            get_price_eurusd, now_price_eurusd, position_eurusd, \
                get_price_usdjpy, now_price_usdjpy, position_usdjpy \
                = self.get_double_position_status(dx_row)

        elif bool_eurusd:  # EURUSDのみポジションを持つ場合
            get_price_eurusd, now_price_eurusd, position_eurusd \
                = self.get_single_position_status(dx_row)

        elif bool_usdjpy:  # USDJPYのみポジションを持つ場合
            get_price_usdjpy, now_price_usdjpy, position_usdjpy \
                = self.get_single_position_status(dx_row)

        return [bool_eurusd, position_eurusd,
                get_price_eurusd, now_price_eurusd], \
            [bool_usdjpy, position_usdjpy,
                get_price_usdjpy, now_price_usdjpy]

    def set_sashine(self, symbol, position_bool, sashine, gyaku_sashine):
        retry_count = 0
        while retry_count < 3:
            position_bool = self.position_bool()
            if position_bool[0][0] and position_bool[1][0]:
                retry_count = 100
            else:
                retry_count += 1
                time.sleep(5)

        tbody = self.driver.find_elements(By.TAG_NAME, 'tbody')
        trigger = tbody[-1].find_elements(
            By.CLASS_NAME, 'PositionGrid_triggerContainer__1yWG1')

        if symbol == 'EURUSD':
            line_num = 0
        else:
            if position_bool[0][0]:  # EURUSDがある場合
                line_num = 1
            else:
                line_num = 0

        try:
            if line_num == 0:
                trigger[0].click()
            else:
                trigger[2].click()

            sashine_gyaku = self.driver.find_elements(
                By.CLASS_NAME, 'PositionGrid_PopupContainer__3AWXo')[-1]
            sashine_gyaku.click()

            self.submit_sashine(sashine, gyaku_sashine)

        # 建玉がない（逆指値決済された）
        except IndexError as e:
            print(e)

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


class FIXAR(TraderDriver):
    def __init__(self, amount, k, pr_k,
                 sashine_eurusd, gyaku_sashine_eurusd,
                 sashine_usdjpy, gyaku_sashine_usdjpy,
                 dynamic_rik, dynamic_son):
        super().__init__()

        self.dynamic_rik, self.dynamic_son = dynamic_rik, dynamic_son
        self.amount = amount
        self.fixa = {'EURUSD': FIXA('EURUSD',
                                    sashine_eurusd, gyaku_sashine_eurusd,
                                    k, pr_k),
                     'USDJPY': FIXA('USDJPY',
                                    sashine_usdjpy, gyaku_sashine_usdjpy,
                                    k, pr_k)}

    def ret_pricedic(self):
        usdjpy, eurusd = self.get_price()
        price_dic = {'EURUSD': eurusd,
                     'USDJPY': usdjpy}

        return price_dic

    def check_login(self):
        if self.driver.current_url\
                == 'https://web.thinktrader.com/account/login':
            self.login()
            time.sleep(10)

    def check_recconect(self):
        """
        再接続画面になっていないかの確認
        """
        try:
            recconect_button = self.driver.find_elements(
                By.CLASS_NAME, 'ReconnectModal_button__1DjYD')
        except WebDriverException as e:
            raise ToDriverRefreshError(e)

        if len(recconect_button) > 0:
            raise ToPageRefreshError('recconect')

    def delete_notif_banner(self):
        """
        通知バナーboxを削除
        """
        rnc__base = self.driver.find_elements(By.CLASS_NAME, 'rnc__base')
        if len(rnc__base) > 0:
            banner_element = "div.rnc__base"  # CSSセレクタを指定
            self.driver.execute_script(
                f"document.querySelector('{banner_element}').remove();")

    def judge_rik_son(self, symbol, position_bool, symbol_get_price=None):
        i = 0 if symbol == 'EURUSD' else 1

        if symbol_get_price is None:
            get_price = position_bool[i][2]
        else:
            get_price = self.fixa[symbol].symbol_get_price

        now_price = position_bool[i][3]
        position = position_bool[i][1]
        price_diff = (now_price - get_price)*position

        rik_diff = self.dynamic_rik[symbol]*get_price
        son_diff = self.dynamic_son[symbol]*get_price

        if price_diff > rik_diff or price_diff < -son_diff:
            return True
        else:
            return False

    def ret_sashine_gyakusashine(self, symbol, rate, pred):
        """
        指値と逆指値の値を計算する関数
        """
        sashine = rate + \
            (self.fixa[symbol].sahine)*(1 if pred else -1)
        gyaku_sashine = rate - \
            (self.fixa[symbol].gyakusashine)*(1 if pred else -1)

        return sashine, gyaku_sashine

    def calc_next_position(self, symbol, position_bool, rate):
        """
        次のポジションを計算する関数
        """
        i = 0 if symbol == 'EURUSD' else 1

        rik_son = self.judge_rik_son(symbol, position_bool,
                                     symbol_get_price=self.fixa[symbol].symbol_get_price)
        if rik_son:  # 利確 or 損切があった場合
            pred = self.fixa[symbol].ret_prediction()
            sashine, gyaku_sashine = self.ret_sashine_gyakusashine(
                symbol, rate, pred)

            pre_position = position_bool[i][1]
            new_position = 1 if pred else -1

            # 新ポジションと旧ポジションが同じ場合
            if pre_position == new_position:
                self.set_sashine(symbol, position_bool,
                                 sashine, gyaku_sashine)
                new_price = position_bool[i][3]
                self.fixa[symbol].symbol_get_price = new_price
                time.sleep(3)
            else:
                self.settle_position(symbol)
                self.fixa[symbol].symbol_get_price = None
                time.sleep(3)

                side = 'buy' if pred else 'sell'
                self.make_order(symbol, side,
                                self.amount,
                                sashine, gyaku_sashine)
                time.sleep(3)

    def calc_new_position(self, symbol, rate):
        """
        ノーポジから次のポジションを計算する関数
        """
        self.fixa[symbol].symbol_get_price = None
        if self.zero_spread():
            pred = self.fixa[symbol].ret_prediction()
            sashine, gyaku_sashine = self.ret_sashine_gyakusashine(
                symbol, rate, pred)
            side = 'buy' if pred else 'sell'
            self.make_order(symbol, side,
                            self.amount,
                            sashine, gyaku_sashine)
            time.sleep(3)

    def run(self):
        # 再接続画面になっていないかの確認
        self.check_recconect()
        # ログイン画面になっていないかの確認
        self.check_login()
        # 通知バナーboxを削除
        self.delete_notif_banner()

        rate_dic = self.ret_pricedic()

        # AI価格リスト更新
        for symbol in ['EURUSD', 'USDJPY']:
            self.fixa[symbol].refresh_pricelist(rate_dic[symbol])

        position_bool = self.position_bool()
        for i, symbol in enumerate(['EURUSD', 'USDJPY']):
            if position_bool[i][0]:  # すでにポジションがある場合
                self.calc_next_position(
                    symbol, position_bool, rate_dic[symbol])

            else:  # ポジションがない場合
                self.calc_new_position(symbol, rate_dic[symbol])

    def driver_refresh(self):
        try:
            self.driver.close()
        except (NoSuchWindowException, InvalidSessionIdException):
            pass

        super().__init__()

        time.sleep(10)


# %%
amount = 1000

sashine_eurusd, gyaku_sashine_eurusd = round(0.1/150, 5), round(0.1/150, 5)
sashine_usdjpy, gyaku_sashine_usdjpy = round(0.1, 3), round(0.1, 3)

dynamic_rik = {'EURUSD': 0.008/100, 'USDJPY': 0.008/100}
dynamic_son = {'EURUSD': 0.015/100, 'USDJPY': 0.015/100}

k, pr_k = 12, 12

fixar = FIXAR(amount, k, pr_k,
              sashine_eurusd, gyaku_sashine_eurusd,
              sashine_usdjpy, gyaku_sashine_usdjpy,
              dynamic_rik, dynamic_son)
# %%
try:
    fixar.login()
except HumanChallengeError as e:
    print(e)
time.sleep(30)
# %%
count = 0
error_count = 0
while error_count < 3:
    # try:
    t = time.time()
    fixar.run()

    """
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

    except Exception as e:
        print(e)
        fixar.driver_refresh()
        line.send_to_masaumi('fatal error occuered \n{}'.format(e))
        error_count += 1
    """

    count += 1
    print('{}\n{}\n__________\n'.format(count, datetime.now()))

    if count % 10 == 0:
        fixar.driver.refresh()
    """
    with open('hist_data/hist_data_{}.pickle'.format(num_hist), 'wb') as f:
        pickle.dump(hist_dic, f)
    """
    if fixar.driver.current_url == 'https://web.thinktrader.com/account/login':
        fixar.login()

    sleep_time = 60 - (time.time() - t)
    sleep_time = sleep_time if sleep_time > 0 else 0
    time.sleep(sleep_time)

line.send_to_masaumi('FIXAR stopped')
# %%
