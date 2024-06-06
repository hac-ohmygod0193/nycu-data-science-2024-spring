from simplegmail.query import construct_query
from simplegmail import Gmail
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import random
import time
from selenium.webdriver.common.keys import Keys
import re

url = 'https://accounts.google.com/v3/signin/identifier?opparams=%253F&dsh=S-1525925196%3A1710245737330330&access_type=offline&client_id=952064420590-rd6lnom3ohdjlsu5aujh85j8bo6ntbgd.apps.googleusercontent.com&o2v=1&redirect_uri=https%3A%2F%2Fapi.stytch.groq.com%2Fv1%2Foauth%2Fcallback%2Foauth-callback-live-c141e357-8830-4675-b9ab-05c35f93da32&response_type=code&scope=openid+email+profile&service=lso&state=Google%3ANHA_OHJey6voMfWBdZi9mQmni6Gs_zyj0XfAV6ViJosv&theme=glif&flowName=GeneralOAuthFlow&continue=https%3A%2F%2Faccounts.google.com%2Fsignin%2Foauth%2Fconsent%3Fauthuser%3Dunknown%26part%3DAJi8hAOjyxawTrZgmuPdythFsR7-RH27tuvP5-bCpNGsYXBOoApN7cyu6S97LPWn-KdoWnKkHwvapnOudk2GcZ1jQXqOCVS5bZ2LCpGQLJcd9pRLTZatRREXbEGeczeKdFsfVPP2AKi05r3qQo4lVjG838zjua4B_YJqOuECHlTcjbb6yRNWHxZgrHGqYeay6FdT3YeTQcRkdCN4Ixo_9u19AJA261olISEd0QM292mCI0gyGm-6IPhU1cCCSqkXXI54mLsPFs3WGFkqK3mnVRB6AranUYHg7munBCZgpN8dtxGeqoPk5ML_3Qqv3spd3n649Ny0lNldn_Xug2jdtIZRbSpBi95KF-3ENP5g4jUAQUQrQal0qF37UIEeGuSus70zGl7kZ3vK9pd9C8xkvfe2_jt0G3VEOw56xgm5bC7EuggH5BrQA-Ni4JPnoYN5am5ayWBrUOLL_8asY2ogcSCrer0Sz3bWOT8UlzjI9hW3XJ5NLQcuVik%26flowName%3DGeneralOAuthFlow%26as%3DS-1525925196%253A1710245737330330%26client_id%3D952064420590-rd6lnom3ohdjlsu5aujh85j8bo6ntbgd.apps.googleusercontent.com%26theme%3Dglif%23&app_domain=https%3A%2F%2Fapi.stytch.groq.com&rart=ANgoxcc7m8rvQaymnq6BJIevRhcE-OJuKg_yGp-j4fxCKM-lFOqRoZzFWpP4JK4O7f9vh8XBAQN1Fa2elEq5hme3AJhwbFBkGXZ4dRHWNVprGXGAw7XWkdU'
options = Options()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
# options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

stytch_session = 'Mmub5Iggqbd4Ahb-qjxsHZz4wt8sPBlisPUIDOdwVtHz'

cookie = stytch_session
url = "https://groq.com/"
headers = {
    "Authorization": f"stytch_session={cookie}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}
driver.get(url)
time.sleep(3)
email_input = driver.find_element(
    "xpath", '/html/body/div[4]/div[2]/form/div/div/input')
email_input.send_keys('ohmygod0193.ee11@nycu.edu.tw')
time.sleep(3)
email_input.send_keys(Keys.RETURN)
gmail = Gmail()


def receive_email():
    query_params = {
        'newer_than': (1, 'min'),
        'older_than': (0,  'hour'),
        'unread': True,
        'sender': ['noreply@groq.com'],
    }
    messages = gmail.get_messages(query=construct_query(query_params))
    print(len(messages))
    for message in messages:
        link = re.search(
            r'(https://stytch.com/v1/magic_links/\S+)', message.plain)
        if link:
            link_url = link.group(1)
            print("Magic Link: " + link_url)
            driver.get(link_url)
            time.sleep(10)
            close_intro = driver.find_element(
                "xpath", '/html/body/div[4]/button/svg')
            close_intro.click()
            time.sleep(30)


receive_email()
