from sqlalchemy import create_engine
import hashlib, base64, json, re
from datetime import datetime
from pprint import pprint


def connect_db(usr, pwd, host, db):
    dbstring = "postgresql://{}:{}@{}:5432/{}"
    dbstring = dbstring.format(usr, pwd, host, db)
    return create_engine(dbstring, client_encoding = 'utf8')



from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options 


def init_driver(): 
    chrome_opts = Options()
    chrome_opts.add_argument("--headless")
    driver = webdriver.Chrome(chrome_options=chrome_opts)
    driver.wait = WebDriverWait(driver, 5) 
    return(driver)

regx_numeric_d = re.compile('\d+')
def cast_float(text):
    try:
        return(float(regx_numeric_d.search(text).group()))
    except ValueError:
        return(0)





