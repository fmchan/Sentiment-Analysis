import os
import jieba
import jieba.analyse
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import joblib
from warnings import simplefilter
import re
import time
from classifier import Classifier
import pandas as pd
from util.logger import Logger
import logging
from util.sentence_util import tokenize_split_text_wo_pos, get_sentiment_from_tags
from util.file_util import read_exclude_headers
from configs.settings import BULLISH_CONTENTS_ORIGINAL_FILE_PATH, BULLISH_TITLES_ORIGINAL_FILE_PATH, BEARISH_CONTENTS_ORIGINAL_FILE_PATH, BEARISH_TITLES_ORIGINAL_FILE_PATH, BULLISH_TITLES_FEEDBACK_FILE_PATH, BEARISH_TITLES_FEEDBACK_FILE_PATH, BULLISH_CONTENTS_FEEDBACK_FILE_PATH, BEARISH_CONTENTS_FEEDBACK_FILE_PATH, TITLE_VECTORIZER_PATH, CONTENT_VECTORIZER_PATH, TITLE_TRANSFORMER_PATH, CONTENT_TRANSFORMER_PATH, TITLE_SGD_MODEL_PATH, CONTENT_SGD_MODEL_PATH, TITLE_LINEAR_SVC_MODEL_PATH, CONTENT_LINEAR_SVC_MODEL_PATH
from configs.settings import STOP_WORDS_FILE_PATH, CUSTOM_DICT_PATH

jieba.load_userdict(CUSTOM_DICT_PATH)
stop_list = [line.strip() for line in open(STOP_WORDS_FILE_PATH, 'r', encoding='utf-8').readlines()]

class Predictor(object):
    def __init__(self):
        self.logger = logging.getLogger("MainLogger")
        # self.title_model = joblib.load(TITLE_SGD_MODEL_PATH) # <100k data samples
        # self.content_model = joblib.load(CONTENT_SGD_MODEL_PATH) # <100k data samples
        self.title_vectorizer = joblib.load(TITLE_VECTORIZER_PATH)
        self.title_transformer = joblib.load(TITLE_TRANSFORMER_PATH)
        self.content_vectorizer = joblib.load(CONTENT_VECTORIZER_PATH)
        self.content_transformer = joblib.load(CONTENT_TRANSFORMER_PATH)

    def transform_predict(self, text, is_title, title_model, content_model, show_prob=True):
        if is_title and any(text.find(word) >= 0 for word in read_exclude_headers()):
            return "", ""
        sentiment_tags = get_sentiment_from_tags(text)
        if len(sentiment_tags) > 0:
            return max(list(dict(sentiment_tags).values())), 1.0
        else:
            token_text = tokenize_split_text_wo_pos(text)
            if is_title:
                prediction = self.__transform_predict(token_text, title_model, self.title_vectorizer, self.title_transformer, True)
            else:
                prediction = self.__transform_predict(token_text, content_model, self.content_vectorizer, self.content_transformer, True)
            result = prediction[0][0]
            probability = prediction[1][0][0] if prediction[1][0][0] > prediction[1][0][1] else prediction[1][0][1]
            return result, probability

    def __analyze_text(self, vectorizer, token_text):
        analyze = vectorizer.build_analyzer()
        analyzed_sentence = {item: analyze(" ".join(token_text)).count(item) for item in set(analyze(" ".join(token_text)))}
        return analyzed_sentence

    def __transform_predict(self, token_text, model, vectorizer, transformer, show_prob):
        x_test = vectorizer.transform(token_text)
        x_test = transformer.transform(x_test)
        if show_prob:
            return model.predict(x_test), model.predict_proba(x_test).tolist()
        else:
            return model.predict(x_test)

if __name__ == '__main__':
    logger = Logger("MainLogger").setup_system_logger()
    predictor = Predictor()
    classifer = Classifier()
    # text = "市場上調匯豐(00005.HK)目標價至65.7元"
    # text = "昨日重要數據公布結果"
    # text = "黃德几：華潤啤酒（00291）銷量增加"
    text = "【企業盈喜】金嗓子（06896）料去年純利增加最少60%"
    is_title = True
    title_model, content_model = classifer.reload_model()
    result, probability = predictor.transform_predict(text, is_title, title_model, content_model)
    print(result)
    print(probability)