# %%
from selenium.webdriver.common.by import By
import time

from fixar import FIXAR
# %%
fixar = FIXAR()
time.sleep(60)

elements = fixar.driver.find_elements(
    By.XPATH, '//tr[@class="dx-row dx-data-row dx-row-lines"]')
elements[1].text
