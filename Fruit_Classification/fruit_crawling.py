from selenium import webdriver
import urllib
import os
import time

driver = webdriver.Chrome('Chromedriver.exe')

fruit = ['banana', 'grape', 'strawberry', 'orange']
url_list = []
url_dict = {}

dataDir = 'C:\\Users\\gjust\\Documents\Github\\data\\fruit\\'
for f in fruit:
    try:
        saveDir = f
        driver.get('https://www.google.com/search?q=fruit+{}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwixzebA9LHtAhXRMd4KHQGTABQQ_AUoAXoECAcQAw&biw=374&bih=888'.format(f))
        
        break_out = 0
        while len(driver.find_elements_by_class_name("isv-r")) < 500:
            
            
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
            try:
                if driver.find_element_by_class_name('mye4qd'):
                    driver.find_element_by_class_name('mye4qd').click()

            except Exception as e:
                break_out += 1
                if break_out == 1000:
                    break
                print(e)
                pass
        img_list = driver.find_elements_by_class_name('rg_i')

        for i, img in enumerate(img_list):

            src = img.get_attribute('src')
            if not(os.path.isdir(dataDir+saveDir)):
                os.makedirs(dataDir+saveDir)
            try:
                urllib.request.urlretrieve(str(src), dataDir+saveDir+'\\{}{}.jpg'.format(f, i))
            except Exception as e:
                print(e, '\\{}{}.jpg'.format(f, i))
    except Exception as e:
        print(e)
        