import sys
sys.path.append("../../crawlers")

from lxml import html, etree
import requests, re
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

endpoint = "https://edemann.dk/liste-danske-postnumre-og-byer"

def collect(url):
    page = requests.get(url)
    tree = html.fromstring(page.content)
    main = tree.xpath("//*/tbody/tr")
    data = pd.DataFrame([])
    for i in range(1, len(main)):
        base = "//*/tbody/tr[%s]" % str(i)
        zipcode = tree.xpath(base + "/td[1]/text()")[0]
        city = tree.xpath(base + "/td[2]/text()")[0]
        data = data.append(pd.DataFrame({'zipcode': zipcode, 'city': city}, index=[0]), ignore_index=True)
    data.to_csv('zipcodes.csv', sep=';')

if __name__ == '__main__':
    collect(endpoint)
