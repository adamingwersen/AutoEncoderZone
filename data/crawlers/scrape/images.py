
import sys
import requests
import shutil
import io
import os
sys.path.append("../../crawlers")

import warnings
warnings.filterwarnings("ignore")

from lxml import html, etree
import requests

from dbtools import config as config
import objects.image as imageObj
import objects.location as locationObj


house_table     = "houses"
location_table  = "locations"
image_table     = "images"
select_urls     = "SELECT url, id from %s;" % house_table
database        = dbname

dest_path        = os.path.abspath('../../../images/')

def collect(row, sql_engine, selenium_driver): 
    print(row)
    total = 0
    selenium_driver.get(row[0])
    b = fetch_geoloc(row, sql_engine, selenium_driver)
    if b != True:
        return None
    xpath = "//*[starts-with(@class, 'fade-pager')]/div"
    try:
        images = selenium_driver.find_elements_by_xpath(xpath)
        image_list = list()
        for image in images:
            image_url = image.get_attribute('data-src')
            image_url = image_url.replace('//', 'http://')
            image_item = imageObj.ImageObject(row[0], row[1], image_url)
            image_item.file_path = dest_path
            download_store_image(image_item.image_url, image_item.file_path, image_item.file_name)
            image_list.append(image_item)
        k = len(image_list)
        total += k
        print('\n --- %s image items collected --- \n' % k)
        [imageObj.insert_db(sql_engine, obj) for obj in image_list]
        print('\n --- %s image items succesfully stored in %s.%s ---\n' % (k, database, image_table))
    except: 
        print("Could not find elements by xpath")
    return None

def download_store_image(url, pathname, filename): 
    response = requests.get(url, stream=True)
    imgfile = os.path.join(pathname, filename)
    with open(imgfile, 'wb') as outfile:
        shutil.copyfileobj(response.raw, outfile)
    del response
    return

def fetch_geoloc(row, sql_engine, selenium_driver):
    xpath = "//*[@class='viamap_container']"
    selenium_driver.implicitly_wait(2)
    try:
        latitude = selenium_driver.find_element_by_xpath(xpath).get_attribute('data-lat')
        longtitude = selenium_driver.find_element_by_xpath(xpath).get_attribute('data-long')
        latitude, longtitude = float(latitude), float(longtitude)
        locitem = locationObj.LocationObject(row[0], row[1])
        locitem.loc_lat = latitude
        locitem.loc_lon = longtitude
        locationObj.insert_db(sql_engine, locitem)
        return True
    except:
        return False

def run():
    engine = config.connect_db(usr,  pwd, "localhost", dbname)

    url_list = list()
    with engine.connect() as cnxn:
        tbl = cnxn.execute(select_urls)
    for row in tbl:
        url_list.append(row)

    driver = config.init_driver()
    for url in url_list:
        collect(url, engine, driver)
    driver.quit()


if __name__ == '__main__':
    run()



