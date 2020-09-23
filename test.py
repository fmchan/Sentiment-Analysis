from service.aastock_scrapy import AASTOCK
from util.logger import Logger
import util.file_util as file_util
import util.sentence_util as sentence_util
import numpy
import collections
from jieba import posseg
from jieba import analyse
import jieba
from configs.settings import STOP_WORDS_FILE_PATH, CUSTOM_DICT_PATH
from itertools import groupby

def scrap_news():
  scraper = AASTOCK()
  scraper.get_market_news()

def spool_titles_files():
  file_util.write_bullish_file_for_titles()
  file_util.write_bearish_file_for_titles()

def spool_contents_files():
  file_util.write_bullish_file_for_contents()
  file_util.write_bearish_file_for_contents()

if __name__ == "__main__":
    logger = Logger("MainLogger").setup_system_logger()
    # print(dispatch_dict2('mul', 2, 8))
    # scrap_news()
    # spool_titles_files()
    # spool_contents_files()
    # file_util.write_all_to_csv()
    # a = numpy.zeros(shape=(3,3))
    # a[0] = [1,2,3]
    # a[1] = [4,5,6]
    # a[2] = [7,8,9]
    # print(a[1,:])
    # print(a[:,1])
    # c = collections.Counter('公海洋公園')
    # print(c)
    # d = collections.Counter('公海 洋 公 園')
    # print(d)
    # my_dict = {"oranges":4, "mangoes":5, "tomatoes":7, "bananas":6}
    # # print unpacked dictionary
    # print(*my_dict)
    jieba.load_userdict(CUSTOM_DICT_PATH)
    text = "卜架擂台本日推介"
    seg = jieba.posseg.cut(text)
    for w, f in seg:
        print(w, f)
    # exclude_list = file_util.read_exclude_headers()
    # print(exclude_list)
    # text = "明天重要數據公布"
    # # text = "股價上市過戶備忘"
    # print(any(text.find(word) >= 0 for word in exclude_list))