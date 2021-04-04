import os
import glob

image_path = 'data/fruit/'
fruit_name_list = os.listdir(image_path)

for name in fruit_name_list:
    image_list = glob.glob(image_path + '{}/*.jpg'.format(name))
    for i, url in enumerate(image_list):
        os.rename(url, 'data/fruit/{}\\{}{}.jpg'.format(name,name, i))