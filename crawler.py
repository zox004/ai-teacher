import urllib.request
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import os

def scroll_down(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")

            try:
                driver.find_element_by_class_name("mye4qd").click()
            except:

               if new_height == last_height:
                   break
        last_height = new_height
        
keyword = input('검색할 태그를 입력하세요 : ')
crawl_num = int(input("크롤링할 갯수: "))
url = 'https://www.google.com/search?q={}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjgwPKzqtXuAhWW62EKHRjtBvcQ_AUoAXoECBEQAw&biw=768&bih=712'.format(keyword)
path = "./data/train/" + keyword

driver = webdriver.Chrome()
driver.get(url)

time.sleep(1)

scroll_down(driver)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
images = soup.find_all('img', attrs={'class':'rg_i Q4LuWd'})

if not os.path.isdir(path):
    os.makedirs(path)

print('number of img tags: ', len(images))

n = 1
for i in images:

    try:
        imgUrl = i["src"]
    except:
        imgUrl = i["data-src"]
        
    with urllib.request.urlopen(imgUrl) as f:
        with open(f'{path}/' + keyword + str(n) + '.jpg', 'wb') as h:
            img = f.read()
            h.write(img)
    
    if n >= crawl_num:
        break

    n += 1
