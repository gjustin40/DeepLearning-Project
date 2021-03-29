from selenium import webdriver
from urllib.request import urlopen
import time
import os
import argparse

parser = argparse.ArgumentParser(description='Crawling')
parser.add_argument('--food', type=str, help='food list txt files')
parser.add_argument('--item', type=str, default='', help='food one item name')
parser.add_argument('--name', type=str, default='new_data', help='Image Folder Name')

opt = parser.parse_args()

def get_food_list():
    if opt.food:
        with open(opt.food) as f:
            lines = f.read().splitlines()
            food_list_en = [name.strip() for name in lines[0].split(',')]
            food_list_kr = [name.strip() for name in lines[1].split(',')]

    else:
        item_en_kr = opt.item.spli(',')
        food_list_en = item_en_kr[0].strip()
        food_list_kr = item_en_kr[1].strip()

    return food_list_en, food_list_kr

def crawling():
    base_url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
    driver = webdriver.Chrome('Chromedriver.exe')
    
    
    food_list_en, food_list_kr = get_food_list()
    print('아래의 음식 사진을 수집합니다.')
    print(food_list_kr)

    if os.path.isdir(opt.name):
        print('같은 이름의 폴더가 있습니다.')
        return 0
    os.makedirs(opt.name, exist_ok=True)

    image_total = 0
    for food_en, food_kr in zip(food_list_en, food_list_kr):
        print('-'*20, food_kr, '  시작', '-'*20)

        driver.get(base_url + food_kr)

        url_list_len = 0
        while True:
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
            time.sleep(0.7)
            driver.find_element_by_class_name('photo_tile')
            image_url_list = driver.find_elements_by_class_name('_image')

            if len(image_url_list) == url_list_len:
                break

            url_list_len = len(image_url_list)

        count = 1
        os.makedirs(food_en, exist_ok=True)
        for image_url in image_url_list:
            with urlopen(image_url.get_attribute('src')) as f:
                with open(f'{food_en}\{food_en}' + str(count) + '.jpg', 'wb') as file_name:
                    img = f.read()
                    file_name.write(img)
            count += 1
        image_total += len(image_url_list)

        print('-'*20, food_kr, '  종료', '-'*20)
        print('{}사진의 총 개수는 {}개입니다.'.format(food_kr, len(image_url_list)))

    print('크롤링 끝')
    print('수집된 이미지의 총 개수는 {}개입니다.'.format(image_total))

if __name__ == '__main__':
    crawling()