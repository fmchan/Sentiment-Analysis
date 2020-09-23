import logging
import jieba
from jieba import posseg
from jieba import analyse
import re
import string
from configs.settings import STOP_WORDS_FILE_PATH, CUSTOM_DICT_PATH

jieba.load_userdict(CUSTOM_DICT_PATH)
stop_list = [line.strip() for line in open(STOP_WORDS_FILE_PATH, 'r', encoding='utf-8').readlines()]
punctuation_set = set(string.punctuation)

# slow processing but support POS, for training
def tokenize_split_text_with_pos(text):
    sentences = []
    words = []
    seg = jieba.posseg.cut(text)
    for word, flag in seg:
        if flag in ['n','vn','x','a','ag','ad','an','b','e','f','h','i','o','v','vg','vd','vi','vn','vq','z','positive','negative'] and word not in punctuation_set and word not in stop_list and not word.isspace() and re.match("^\d*?\%?\.?\d*?\%?$", word) is None and re.compile(r"[A-Za-z]").match(word) is None:
            words.append(word)
    sentence = " ".join(words)
    sentences.append(sentence)
    return sentences

# faster processing, for prediction
def tokenize_split_text_wo_pos(text):
        sentences = []
        words = []
        # seg = [t for t in jieba.cut(text) if len(t) > 1 and t not in stop_list and re.match("^\d*?\%?\.?\d*?\%?$", t) is None and re.compile(r"[A-Za-z]").match(t) is None]
        seg = [t for t in jieba.cut(text) if t not in punctuation_set and t not in stop_list and re.match("^\d*?\%?\.?\d*?\%?$", t) is None and re.compile(r"[A-Za-z]").match(t) is None]
        print(seg)
        for i in seg:
                words.append(i)
        sentence = " ".join(words)
        sentences.append(sentence)
        return sentences

def get_sentiment_from_tags(text):
        sentiment_tags = analyse.extract_tags(text, topK = 50, withWeight = False, allowPOS = ('negative','positive'), HMM = False, withFlag = True)
        return sentiment_tags