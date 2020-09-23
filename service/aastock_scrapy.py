#-*- coding: utf-8 -*-
from configs.settings import ALLOW_OVERWRITE, SECOND_TIME_OUT, options, WEBDRIVER_PATH, SCROLL_PAUSE_TIME, SCROLL_NUM, MIN_TEXT_NUM, SCRAP_PAUSE_TIME
# from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium import webdriver
from bs4 import BeautifulSoup
import urllib.request
import time
import logging
import requests
import datetime
from dbhelper import DBHelper

class AASTOCK(object):
    def __init__(self):
        self.agent = 'aastock'
        self.logger = logging.getLogger("MainLogger")
        self.encoding = 'utf-8'
        self.site = 'http://www.aastocks.com'
        self.db = DBHelper()

    def get_market_news(self):
        urls = [
            # 'http://www.aastocks.com/tc/stocks/news/aafn',
            # 'http://www.aastocks.com/tc/stocks/news/aafn/top-news',
            # 'http://www.aastocks.com/tc/stocks/news/aafn/popular-news',
            # 'http://www.aastocks.com/tc/stocks/news/aafn/latest-news',
            # 'http://www.aastocks.com/tc/stocks/news/aafn/recommend-news',
            'http://www.aastocks.com/tc/stocks/news/aafn/positive-news',
            # 'http://www.aastocks.com/tc/stocks/news/aafn/positive-news/3',
            'http://www.aastocks.com/tc/stocks/news/aafn/negative-news'
            # 'http://www.aastocks.com/tc/stocks/news/aafn/negative-news/3',
        ]
        for url in urls:
            self.logger.info('scraping the news for %s ', url)
            if 'positive' in url:
                category = 'positive'
            else:
                category = 'negative'
            try:
                self._start_driver()
                self.logger.info('browseing ' + url)
                self.browser.get(url)
                WebDriverWait(self.browser, SECOND_TIME_OUT).until(EC.presence_of_element_located((By.CLASS_NAME, 'content')))
                for i in range(1, SCROLL_NUM):
                    self.logger.info('scrolling down')
                    self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                    time.sleep(SCRAP_PAUSE_TIME)

                self.logger.info('getting all urls ')
                soup = BeautifulSoup(self.browser.page_source, 'html.parser')
                for item in soup.find_all('a'):
                    link = item.get('href')
                    if link is not None and "/tc/stocks/news/aafn-con/" in str(link):
                        title = item.get('title')
                        url = self.site + link
                        if title is not None and not self.db.is_title_processed_for_today(title):
                            self.logger.info('processing ' + url)
                            # self.logger.info('title: ' + title)
                            self._get_market_news_core(url, category)
                        else:
                            self.logger.info('bypassing ' + url)
                        time.sleep(SCRAP_PAUSE_TIME)
                time.sleep(SCRAP_PAUSE_TIME)
            except Exception as e:
                message = 'Exception in get_market_news: %s' % e
                self.logger.exception(message)
            finally:
                self._close_driver()
            self.logger.info('scraping end')

    def _get_market_news_core(self, url, category):
        page_req = requests.get(url)
        html = page_req.text.encode(self.encoding)
        if page_req.status_code == requests.codes.ok: #pylint: disable=no-member
            soup = BeautifulSoup(html, 'html.parser')
            WebDriverWait(self.browser, SECOND_TIME_OUT).until(EC.presence_of_element_located((By.CLASS_NAME, 'grid_11')))
            article = soup.find("div", {"class": "grid_11"})
            if article is not None:
                contents = article.findAll('p')
                bullish_count = 0
                bearish_count = 0
                if contents is not None:
                    text = ''
                    for content in contents:
                        text += content.text
                    # self.logger.info(text)
                    title = article.find("div", {"class": "newshead5"}).text
                    # self.logger.info(title)
                    publish_time = article.select('div[class*="newstime5"]')[0].text
                    date_time_obj = datetime.datetime.strptime(publish_time, '%Y/%m/%d %H:%M')
                    # self.logger.info(date_time_obj)
                    bullish_count = article.select('div[class*="divBullish"]')[0].find("div", {"class": "value"}).text
                    # self.logger.info(bullish_count)
                    bearish_count = article.select('div[class*="divBearish"]')[0].find("div", {"class": "value"}).text
                    # self.logger.info(bearish_count)
                    if len(text) < MIN_TEXT_NUM:
                        text = title
                    self.db.insert_item(link=url, category=category, title=title, published_time=date_time_obj, bullish_count=bullish_count, bearish_count=bearish_count, text=text)

    def _start_driver(self):
        self.logger.info('starting driver...')
        try:
            # self.browser = webdriver.Firefox(executable_path = WEBDRIVER_PATH, firefox_options = options)
            self.browser = webdriver.Chrome(executable_path = WEBDRIVER_PATH, chrome_options = options)
        except Exception as e:
            message = 'Exception in _start_driver: %s' % e
            self.logger.exception(message)
            return message
        self.logger.info('started driver...')

    def _close_driver(self):
        self.logger.info('closing driver...')
        self.browser.quit()
        self.logger.info('closed!')    
