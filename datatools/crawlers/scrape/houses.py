""" Fix this with setup.py + pants/basil build at some point """
import sys
sys.path.append("../../crawlers")

from lxml import html, etree
import requests, re
import pandas as pd

import dbtools.config as config
import objects.house as houseObj

import warnings
warnings.filterwarnings("ignore")

endpoint = ""
database = dbname
table = "houses"

def collect(url, sql_engine):
    base_url = ""
    page = requests.get(url)
    tree = html.fromstring(page.content)
    err = False
    try:
        err = tree.xpath('//*[@class="searchwarning__text icon-warning"]')[0]
    except IndexError:
        pass
    if err != False:
        return
    main = tree.xpath("//*[@class='col-12 propertyitem propertyitem--list']")
    house_list = list()

    for i in range(3, len(main) + 3):
        base = "//*[@class='maincontent ']//*[@class='row'][%s]" % str(i)
        href = tree.xpath(base + "//*[@class='propertyitem__link']/@href")
        case_id = tree.xpath(base + "/div/@data-caseid")[0]

        try:
            price = tree.xpath(base + "//*[@class='propertyitem__price']//*[contains(text(), 'Pris:')]/text()")
            price = config.cast_float(price[0].replace('.', ''))
            house_type = tree.xpath(base + "//*[@class='propertyitem__owner--listview']/text()")[0].split(' ')[0]
            if "Ejerudgifter pr. md." in tree.xpath(base + "//*/table//*/th[8]/text()")[0]:
                owner_expense = tree.xpath(base + "//*/table//*/td[8]/text()")[0]
                owner_expense = float(owner_expense.replace('.', ''))

            elif "Boligyd. / Forbrugsafh." in tree.xpath(base + "//*/table//*/th[7]/text()")[0]:
                expense = tree.xpath(base + "//*/table//*/td[7]/text()")[0]
                print(expense)
                expenses = expense.split(' / ')
                expense_1, expense_2 = float(expenses[0].replace('.', '')), float(expenses[1].replace('.', ''))
                owner_expense = expense_1 + expense_2
            else:
                owner_expense = 0.0
            try:
                energy_label = tree.xpath(base + "//*[@class='propertyitem__owner--listview']/a/@title")[0]
                regexp = re.compile("(?<=Energimærke )(.*)(?=\s-)")
                energy_label = regexp.search(energy_label)[0]
            except: 
                energy_label = "None"
            if "m²" in tree.xpath(base + "//*/table//*/th[1]/text()")[0]:
                living_sqm = tree.xpath(base + "//*/table//*/td[1]/text()")[0]
            else: 
                living_sqm = 0
            if "Rum" in tree.xpath(base + "//*/table//*/th[3]/text()")[0]:
                rooms = tree.xpath(base + "//*/table//*/td[3]/text()")[0]
            else: 
                rooms = 0
            if "Byggeår" in tree.xpath(base + "//*/table//*/th[4]/text()")[0]:
                build_year = tree.xpath(base + "//*/table//*/td[4]/text()")[0]
            else:
                build_year = 0
            if "Liggetid" in tree.xpath(base + "//*/table//*/th[5]/text()")[0]:
                time_on_market = tree.xpath(base + "//*/table//*/td[5]/text()")[0]
            else: 
                time_on_market = 0
            address = tree.xpath(base + "//*[@class='propertyitem__wrapper']//*[@itemprop='streetAddress']/text()")[0]
            info = tree.xpath(base + "//*[@class='propertyitem__wrapper']//*[@class='propertyitem__zip--listview']/text()")
            zipcode = int(info[0].split()[0])
            if len(info[0].split()) > 2:
                city = ' '.join(info[0].split()[1:])
            else: 
                city = info[0].split()[1]

            house_object = houseObj.HouseObject(base_url + href[0])
            house_object.case_id = int(case_id)
            house_object.price = price
            house_object.owner_expense = owner_expense
            house_object.energy_label = energy_label
            house_object.house_type = house_type
            house_object.living_sqm = int(living_sqm.replace('.', ''))
            house_object.rooms = int(rooms.replace('.', ''))
            house_object.build_year = int(build_year)
            house_object.time_on_market = int(time_on_market.replace('.', ''))
            house_object.address = address
            house_object.zipcode = zipcode
            house_object.city = city

            if (i % 10 == 0):
                print("---- ITERATION: {}".format(i))
                print(house_object)

            house_list.append(house_object)
        except IndexError:
            pass
    k = len(house_list)
    print('\n --- {} items collected --- from: {}\n'.format(k, url))
    for item in house_list:
        try:
            houseObj.insert_db(sql_engine, item) 
        except: 
            print("constraint violated")
            pass
    print('\n --- %s items succesfully stored in %s.%s ---\n' % (k, database, table))

def run():
    engine = config.connect_db(usr,  pwd, "localhost", dbname)
    data = pd.read_csv('../zipcodes/zipcodes.csv', sep = ';')
    zipcodes = data['zipcode'].tolist()
    for code in zipcodes:
        ep = endpoint.format(code)
        print(ep)
        collect(ep, engine)


if __name__ == "__main__":
    run()










                
                
                




