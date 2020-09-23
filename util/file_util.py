from dbhelper import DBHelper
import logging
from configs.settings import DATA_PATH, EXCLUDE_HEADERS_FILE_PATH, BULLISH_CONTENTS_ORIGINAL_FILE_PATH, BULLISH_TITLES_ORIGINAL_FILE_PATH, BEARISH_CONTENTS_ORIGINAL_FILE_PATH, BEARISH_TITLES_ORIGINAL_FILE_PATH
import csv

def read_exclude_headers():
    with open(EXCLUDE_HEADERS_FILE_PATH, mode='r',  encoding='utf-8') as f:
        lines = f.read().splitlines()
    return lines

def write_bullish_file_for_titles():
    db = DBHelper()
    items = db.query_titles_from_bullish_items()
    _write_file_core(BULLISH_TITLES_ORIGINAL_FILE_PATH, items)

def write_bearish_file_for_titles():
    db = DBHelper()
    items = db.query_titles_from_bearish_items()
    _write_file_core(BEARISH_TITLES_ORIGINAL_FILE_PATH, items)

def write_bullish_file_for_contents():
    db = DBHelper()
    items = db.query_contents_from_bullish_items()
    _write_file_core(BULLISH_CONTENTS_ORIGINAL_FILE_PATH, items)

def write_bearish_file_for_contents():
    db = DBHelper()
    items = db.query_contents_from_bearish_items()
    _write_file_core(BEARISH_CONTENTS_ORIGINAL_FILE_PATH, items)

def write_all_to_csv():
    db = DBHelper()
    items = db.query_all_items()
    with open(DATA_PATH + 'aastock-news-data.csv', mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('category', 'title', 'content', 'published_time', 'bullish_count', 'bearish_count'))
        for item in items:
            writer.writerow((str(column).strip('\r').strip() for column in item))

def _write_file_core(file_name, items):
    logger = logging.getLogger("MainLogger")
    try:
        with open(file_name, mode='w', newline='', encoding='utf-8') as the_file:
            for item in items:
                the_file.write(item.strip('\r').strip())
                the_file.write('\n')
    except Exception as e:
        message = "Exception in _write_file_core: %s" % e
        logger.exception(message)            
    # try:
    #     file = open(DATA_PATH + file_name, mode='w', newline='', encoding='utf-8')
    #     for item in items:
    #         file.write(item.strip('\r').strip())
    #         file.write('\n')
    #     # file.write(str(items))
    # except Exception as e:
    #     message = "Exception in _write_file_core: %s" % e
    #     logger.exception(message)
    # finally:
    #     file.close()