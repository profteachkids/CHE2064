from selenium import webdriver
from lxml import html
import time
import asyncio
import random

class Drivers():
    def __init__(self, n):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.drivers = [(lambda: webdriver.Chrome(options=options))()]*n
    def __enter__(self):
        return self.drivers
    def __exit__(self, type, value, traceback):
        for driver in self.drivers:
            driver.close()


            
new_links=set()
processed = set()
new_links.add('/')
root='https://www.nasa.gov'
n_drivers=3


async def get_page_links(driver, f):
    global n_links, processed, new_links
    if len(new_links)==0:
        return []
    link = new_links.pop()
    await asyncio.sleep(random.uniform(0.05, 0.1))
    driver.get(root + link)
    await asyncio.sleep(0.5)
    processed.add(link)
    f.write(f'{len(processed)} {link}\n')
    f.flush()
    tree = html.fromstring(driver.page_source)
    return set(tree.xpath('//a[starts-with(@href,"/") and string-length(@href)>1]/@href'))

async def get_all_links():
    global processed, new_links
    with open('links.txt', 'w') as f:
        with Drivers(n_drivers) as drivers:
            while len(new_links)>0:
                coroutines = [get_page_links(drivers[i],f) for i in range(n_drivers)]
                page_links = set().union(*await asyncio.gather(*coroutines))
                links = page_links-processed
                new_links.update(links)
                print(f'processed: {len(processed)}  links: {len(links)}  new_page_links: {len(links)}  new_links: {len(new_links)}')

asyncio.run(get_all_links())