"""The file contains name settings."""
# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.options import Options
import platform
from pathlib import Path

EXCLUDE_HEADERS = ['content']

BULLISH_TITLES_ORIGINAL_FILE_PATH = 'data/bullish_file_for_titles.txt'
BEARISH_TITLES_ORIGINAL_FILE_PATH = 'data/bearish_file_for_titles.txt'
BULLISH_CONTENTS_ORIGINAL_FILE_PATH = 'data/bullish_file_for_contents.txt'
BEARISH_CONTENTS_ORIGINAL_FILE_PATH = 'data/bearish_file_for_contents.txt'
BULLISH_TITLES_FEEDBACK_FILE_PATH = 'data/bullish_feedback_for_titles.csv'
BEARISH_TITLES_FEEDBACK_FILE_PATH = 'data/bearish_feedback_for_titles.csv'
BULLISH_CONTENTS_FEEDBACK_FILE_PATH = 'data/bullish_feedback_for_contents.csv'
BEARISH_CONTENTS_FEEDBACK_FILE_PATH = 'data/bearish_feedback_for_contents.csv'

TITLE_VECTORIZER_PATH = 'predictor/train_title_vectorizer.m'
TITLE_TRANSFORMER_PATH = 'predictor/train_title_transformer.m'
CONTENT_VECTORIZER_PATH = 'predictor/train_content_vectorizer.m'
CONTENT_TRANSFORMER_PATH = 'predictor/train_content_transformer.m'
TITLE_SGD_MODEL_PATH = 'predictor/train_title_sgd_model.m'
CONTENT_SGD_MODEL_PATH = 'predictor/train_content_sgd_model.m'
TITLE_LINEAR_SVC_MODEL_PATH = 'predictor/train_title_linearsvc_model.m'
CONTENT_LINEAR_SVC_MODEL_PATH = 'predictor/train_content_linearsvc_model.m'
PREDICTOR_PATH = 'predictor/'

STOP_WORDS_FILE_PATH = 'dict/stop_words.txt'
CUSTOM_DICT_PATH = 'dict/dict.txt.custom'
EXCLUDE_HEADERS_FILE_PATH = 'dict/exclude_headers.txt'

# output settings
ALLOW_OVERWRITE = False # set False to read the data file instead of re-scraping

MIN_TEXT_NUM = 100
SCROLL_NUM = 20

# time setting
SECOND_TIME_OUT = 20
SCROLL_PAUSE_TIME = 1
SCRAP_PAUSE_TIME = 5

# flask config
HOST = '192.168.2.225'
PORT = 5050

# web driver config
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument("--start-maximized")
# options.add_argument('window-size=2560,1440')
# prefs = {"profile.default_content_setting_values.notifications" : 2}
# prefs = {"profile.managed_default_content_settings.images" : 2} # disable loading image
# options.add_experimental_option("prefs", prefs) # for chrome only
options.add_experimental_option("prefs", {
    "profile.default_content_setting_values.notifications" : 2,
    "profile.managed_default_content_settings.images" : 2
})

# config by platform
if platform.system() in ['Darwin', 'Windows']:
    LOG_PATH = 'C://temp//sentiment-analysis//log/'
    DATA_PATH = 'C://temp//sentiment-analysis//data/'
    WEBDRIVER_PATH = 'exe/chromedriver.exe'
    # WEBDRIVER_PATH = 'exe/geckodriver.exe'
    DB_PATH = 'sqlite:///aastock.db'
else:
    LOG_PATH = '/var/local/apps/data-extractor/'
    DATA_PATH = '/usr/local/apps/Training-Data-Extractor/data/'
    WEBDRIVER_PATH = '/usr/local/bin/chromedriver'
    # WEBDRIVER_PATH = '/usr/local/apps/Training-Data-Extractor/exe/chromedriver'
    # WEBDRIVER_PATH = '/usr/local/apps/Training-Data-Extractor/exe/phantomjs'
    # WEBDRIVER_PATH = '/usr/local/apps/Training-Data-Extractor/exe/geckodriver'
    DB_PATH = 'sqlite:////root/aastock.db'

Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)