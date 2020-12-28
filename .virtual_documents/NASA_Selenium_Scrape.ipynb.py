from selenium import webdriver
from lxml import html
import time



options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
class Driver():
    def __init__(self):
        self.driver = webdriver.Chrome(options=options)
    def __enter__(self):
        return self.driver
    def __exit__(self, type, value, traceback):
        self.driver.close()

def crawl(url, f, driver):
    global num_links, processed
    if url in processed:
        return
    processed.add(url)
    num_links+=1
    f.write(f'{num_links} {url}\n')
    f.flush()
    print(f'{num_links} {url}')
    driver.get(root + url)
    time.sleep(1)
    tree = html.fromstring(driver.page_source)
    links=tree.xpath('//a[starts-with(@href,"/") and string-length(@href)>1]/@href')
    for link in links:
        crawl(link, f, driver)
        
root = 'https://www.nasa.gov'
processed=set()
num_links=0
with Driver() as driver:
  with open('links.txt', 'w') as f:
      crawl('', f, driver)


links = {'/'}
processed = set()
new_links=set()

with Driver() as driver:
  with open('links.txt', 'w') as f:
      while len(links)>0:
          for i, link in enumerate(links):
              driver.get(root + link)
              time.sleep(1)
              tree = html.fromstring(driver.page_source)
              page_links = tree.xpath('//a[starts-with(@href,"/") and string-length(@href)>1]/@href')
              new_links.update(page_links)
              processed.add(link)
              print(f'{i}/{len(links)}/{len(new_links)} {link}')
              f.write(f'{len(processed)} {link}\n')
              f.flush()
          links = new_links-processed
          print(f'processed: {len(processed)}  links: {len(links)}')



