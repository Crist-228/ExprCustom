# code for web crawling. 

import urllib.request
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import os

SCROLL_PAUSE_SEC = 2

def scroll_down():
    global driver
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_SEC)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            time.sleep(SCROLL_PAUSE_SEC)
            new_height = driver.execute_script("return document.body.scrollHeight")

            try:
                driver.find_element_by_class_name("mye4qd").click()
            except:

               if new_height == last_height:
                   break


        last_height = new_height

keyword = input('输入您要搜索的标签 : ')
keyword_mdf = keyword.replace(" ", "_")
directory = f"./real_reg/samples_{keyword_mdf}"
# "./real_reg/samples_cat"
image_directory = directory + "/" + keyword_mdf
# "./real_reg/samples_cat/cat"

os.makedirs(image_directory, exist_ok=True)

f=open(os.path.join(directory, "caption.txt"), "w")
f.close()
f=open(os.path.join(directory, "images.txt"), "w")
f.close()
f=open(os.path.join(directory, "urls.txt"), "w")
f.close()

url = 'https://www.google.com/search?q={}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjgwPKzqtXuAhWW62EKHRjtBvcQ_AUoAXoECBEQAw&biw=768&bih=712'.format(keyword)

driver = webdriver.Chrome()
driver.get(url)

time.sleep(1)

scroll_down()

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
images = soup.find_all('img', attrs={'class':'rg_i Q4LuWd'})

print('number of img tags: ', len(images))

n = 1

f1=open(os.path.join(directory, "caption.txt"), "a")
f2=open(os.path.join(directory, "images.txt"), "a")
f3=open(os.path.join(directory, "urls.txt"), "a")

for i in images:

    try:
        imgUrl = i["src"]
    except:
        imgUrl = i["data-src"]
        
    with urllib.request.urlopen(imgUrl) as f:
        with open(image_directory+ "/" + str(n) + '.jpg', 'wb') as h:
            img = f.read()
            h.write(img)

    f1.write(f"a image of {keyword_mdf}\n")
    f2.write(f"{image_directory}/{str(n)}.jpg\n")
    # 위에 문장에서 .jpg이후에 띄어쓰기가 있어서 FileNotFoundError가 발생했다. ㅠㅠ
    f3.write(f"{imgUrl}\n")

    if n ==20:
        break

    n += 1

f1.close()
f2.close()
f3.close()
