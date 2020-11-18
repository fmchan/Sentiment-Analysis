import json
import csv
import jieba
import jieba.analyse
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
from warnings import simplefilter
import re
import time
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
import os
import platform
from util.sentence_util import tokenize_split_text_wo_pos, tokenize_split_text_with_pos

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from configs.settings import BULLISH_CONTENTS_ORIGINAL_FILE_PATH, BULLISH_TITLES_ORIGINAL_FILE_PATH, BEARISH_CONTENTS_ORIGINAL_FILE_PATH, BEARISH_TITLES_ORIGINAL_FILE_PATH, BULLISH_TITLES_FEEDBACK_FILE_PATH, BEARISH_TITLES_FEEDBACK_FILE_PATH, BULLISH_CONTENTS_FEEDBACK_FILE_PATH, BEARISH_CONTENTS_FEEDBACK_FILE_PATH, TITLE_VECTORIZER_PATH, CONTENT_VECTORIZER_PATH, TITLE_TRANSFORMER_PATH, CONTENT_TRANSFORMER_PATH, TITLE_SGD_MODEL_PATH, CONTENT_SGD_MODEL_PATH, TITLE_LINEAR_SVC_MODEL_PATH, CONTENT_LINEAR_SVC_MODEL_PATH, PREDICTOR_PATH
from configs.settings import STOP_WORDS_FILE_PATH, CUSTOM_DICT_PATH

from util.logger import Logger
import logging

if platform.system() not in ["Darwin", "Windows"]:
    os.chdir("/usr/local/apps/ETNet-Sentiment-Analysis-Master")

simplefilter(action='ignore', category=FutureWarning)

jieba.load_userdict(CUSTOM_DICT_PATH)
stop_list = [line.strip() for line in open(STOP_WORDS_FILE_PATH, 'r', encoding='utf-8').readlines()]

class Classifier(object):
    def __init__(self):
        self.logger = logging.getLogger("MainLogger")

    def reload_model(self):
        self.logger.info("reloading model")
        title_model = joblib.load(TITLE_SGD_MODEL_PATH) # <100k data samples
        content_model = joblib.load(CONTENT_SGD_MODEL_PATH) # <100k data samples
        return title_model, content_model

    def evalute_models(self):
        self.logger.info("assessing models for titles")
        pos_title_data = self.__process_scrap_data(BULLISH_TITLES_ORIGINAL_FILE_PATH, "positive")
        neg_title_data = self.__process_scrap_data(BEARISH_TITLES_ORIGINAL_FILE_PATH, "negative")
        vectorizer, transformer, x_train, y_train, x_test, y_test = self.__transform_data(pos_title_data, neg_title_data, True)
        models = self.create_baseline_classifiers()
        summary = self.assess_models(x_train, y_train, x_test, y_test, models, is_title=True)
        output = self.extract_metric(summary, 'roc_auc')
        self.logger.info(output)

        self.logger.info("assessing models for contents")
        pos_content_data = self.__process_scrap_data(BULLISH_CONTENTS_ORIGINAL_FILE_PATH, "positive")
        neg_content_data = self.__process_scrap_data(BEARISH_CONTENTS_ORIGINAL_FILE_PATH, "negative")
        vectorizer, transformer, x_train, y_train, x_test, y_test = self.__transform_data(pos_title_data, neg_title_data, False)
        models = self.create_baseline_classifiers()
        summary = self.assess_models(x_train, y_train, x_test, y_test, models, is_title=False)
        output = self.extract_metric(summary, 'roc_auc')
        self.logger.info(output)

    def train_for_init(self):
        pos_title_data = self.__process_scrap_data(BULLISH_TITLES_ORIGINAL_FILE_PATH, "positive")
        neg_title_data = self.__process_scrap_data(BEARISH_TITLES_ORIGINAL_FILE_PATH, "negative")
        # pos_title_data = self.__process_feedback_data(BULLISH_TITLES_FEEDBACK_FILE_PATH, "positive")
        # neg_title_data = self.__process_feedback_data(BEARISH_TITLES_FEEDBACK_FILE_PATH, "negative")
        vectorizer, transformer, x_train, y_train, x_test, y_test = self.__transform_data(pos_title_data, neg_title_data, True)
        self.logger.info("training sgdclassifier by titles")
        train_sgd_model = self.__fit_by_sgdclassifier(x_train, y_train, True)
        self.__measure_coef(train_sgd_model, vectorizer, transformer, x_test, y_test, True)
        self.logger.info("training linearsvc by titles")
        train_linearsvc_model = self.__fit_by_linearsvc(x_train, y_train, True)
        self.__measure_coef(train_linearsvc_model, vectorizer, transformer, x_test, y_test, True)
        # raise SystemExit

        pos_content_data = self.__process_scrap_data(BULLISH_CONTENTS_ORIGINAL_FILE_PATH, "positive")
        neg_content_data = self.__process_scrap_data(BEARISH_CONTENTS_ORIGINAL_FILE_PATH, "negative")
        # pos_content_data = self.__process_feedback_data(BULLISH_CONTENTS_FEEDBACK_FILE_PATH, "positive")
        # neg_content_data = self.__process_feedback_data(BEARISH_CONTENTS_FEEDBACK_FILE_PATH, "negative")
        vectorizer, transformer, x_train, y_train, x_test, y_test = self.__transform_data(pos_content_data, neg_content_data, False)
        self.logger.info("training sgdclassifier by contents")
        train_sgd_model = self.__fit_by_sgdclassifier(x_train, y_train, False)
        self.__measure_coef(train_sgd_model, vectorizer, transformer, x_test, y_test, False)
        self.logger.info("training linearsvc by contents")
        train_linearsvc_model = self.__fit_by_linearsvc(x_train, y_train, False)
        self.__measure_coef(train_linearsvc_model, vectorizer, transformer, x_test, y_test, False)

    def train_for_increment(self, text, expect_output, is_title, max_iter = 1000):
        sentences_w_pos = tokenize_split_text_with_pos(text)
        sentences_wo_pos = tokenize_split_text_wo_pos(text)

        if is_title:
            model = joblib.load(TITLE_SGD_MODEL_PATH) # <100k data samples
            vectorizer = joblib.load(TITLE_VECTORIZER_PATH)
            transformer = joblib.load(TITLE_TRANSFORMER_PATH)
        else:
            model = joblib.load(CONTENT_SGD_MODEL_PATH) # <100k data samples
            vectorizer = joblib.load(CONTENT_VECTORIZER_PATH)
            transformer = joblib.load(CONTENT_TRANSFORMER_PATH)

        x_train_w_pos = vectorizer.transform(sentences_w_pos)
        x_train_w_pos = transformer.transform(x_train_w_pos)
        x_train_wo_pos = vectorizer.transform(sentences_wo_pos)
        x_train_wo_pos = transformer.transform(x_train_wo_pos)
        self.logger.info("input: %s" % text)
        self.logger.info("expect_output: %s" % expect_output)
        self.logger.info("result before re-training: %s" % model.predict(x_train_wo_pos))
        counter = 0
        for i in range (0, max_iter):
            model.partial_fit(x_train_w_pos, [expect_output])
            if(model.predict(x_train_wo_pos) == [expect_output]):
                counter = i
                break
        else:
            self.logger.info("fail to train")

        df = pd.DataFrame([[pd.datetime.now(), "".join(text.splitlines()), counter]])
        if is_title:
            joblib.dump(model, TITLE_SGD_MODEL_PATH)
            if expect_output == "positive":
                df.to_csv(BULLISH_TITLES_FEEDBACK_FILE_PATH, mode='a', index=False, header=None)
            else:
                df.to_csv(BEARISH_TITLES_FEEDBACK_FILE_PATH, mode='a', index=False, header=None)
        else:
            joblib.dump(model, CONTENT_SGD_MODEL_PATH)
            if expect_output == "positive":
                df.to_csv(BULLISH_CONTENTS_FEEDBACK_FILE_PATH, mode='a', index=False, header=None)
            else: 
                df.to_csv(BEARISH_CONTENTS_FEEDBACK_FILE_PATH, mode='a', index=False, header=None)
        self.logger.info("result after re-training: %s" % model.predict(x_train_wo_pos))

    def analyze_text(self, text, is_title):
        token_text = tokenize_split_text_with_pos(text)
        if is_title:
            vectorizer = joblib.load(TITLE_VECTORIZER_PATH)
        else:
            vectorizer = joblib.load(CONTENT_VECTORIZER_PATH)
        analyzer = vectorizer.build_analyzer()
        return {item: analyzer(" ".join(token_text)).count(item) for item in set(analyzer(" ".join(token_text)))}

    def __process_scrap_data(self, file_name, label):
        data = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for text in f:
                if len(text.strip()) > 0:
                    data.append(re.sub('\《(\w+)\》', '', text))
        return self.__process_data_core(data, label, file_name.split(".")[0] + ".csv")

    def __process_feedback_data(self, file_name, label):
        df = pd.read_csv(file_name, header=None)
        df = df.iloc[:, 1]
        return self.__process_data_core(list(df), label, file_name.split(".")[0] + ".csv")

    def __process_data_core(self, data, label, file_name):
        tokenized_data = []
        tokenized_content = []
        for text in data:
            tokenized_post = []
            for sent in text.split():
                print(sent)
                filtered = [t for t in jieba.cut(sent) if len(t) > 1 and t not in stop_list and re.match("^\d*?\%?\.?\d*?\%?$", t) is None and re.compile(r"[A-Za-z]").match(t) is None]
                if filtered:
                    # print(filtered)
                    tokenized_post += filtered
            tokenized_data.append((list(dict.fromkeys(tokenized_post)), label))
            tokenized_content.append(list(dict.fromkeys(tokenized_post)))

        df = pd.DataFrame(tokenized_content)
        # df.dropna()
        base = os.path.basename(file_name)
        df.to_csv("data/tokenized_" + base, index=False, header=None)
        return tokenized_data   

    def __transform_data(self, pos_data, neg_data, is_title):
        train_ratio = 0.95
        random.seed(42)
        random.shuffle(pos_data)
        random.shuffle(neg_data)

        max_size = len(neg_data) if len(pos_data) > len(neg_data) else len(pos_data)
        train_size = int(round(max_size * train_ratio))
        self.logger.info("total size: %s" % max_size)
        self.logger.info("training size: %s" % train_size)
        x_train, y_train, x_test, y_test = [], [], [], []
        for i in range(train_size):
            x_train.append(" ".join(pos_data[i][0]))
            x_train.append(" ".join(neg_data[i][0]))
            y_train.append(pos_data[i][1])
            y_train.append(neg_data[i][1])
        for i in range(train_size, max_size):
            x_test.append(" ".join(pos_data[i][0]))
            x_test.append(" ".join(neg_data[i][0]))
            y_test.append(pos_data[i][1])
            y_test.append(neg_data[i][1])
        # max_df = 0.8
        # min_df = 0
        max_df = 1.0 # default i.e. ignore less than 100% overall
        min_df = 1 # default i.e. ignore less than 1 doc
        vectorizer = CountVectorizer(max_df = max_df,
                        min_df = min_df,
                        token_pattern = r"(?u)\b\w+\b",
                            stop_words = frozenset(stop_list))
        x_train = vectorizer.fit_transform(x_train)
        transformer = TfidfTransformer()
        x_train = transformer.fit_transform(x_train)
        # feature_names = np.array(vectorizer.get_feature_names())
        # self.logger.info(feature_names[:20])
        if is_title:
            joblib.dump(vectorizer, TITLE_VECTORIZER_PATH)
            joblib.dump(transformer, TITLE_TRANSFORMER_PATH)
        else:
            joblib.dump(vectorizer, CONTENT_VECTORIZER_PATH)
            joblib.dump(transformer, CONTENT_TRANSFORMER_PATH)
        return vectorizer, transformer, x_train, y_train, x_test, y_test

    def __fit_by_sgdclassifier(self, x_train, y_train, is_title):
        sgd_model = SGDClassifier(random_state=42, loss='log')
        sgd_model.fit(x_train, y_train)
        if is_title:
            joblib.dump(sgd_model, TITLE_SGD_MODEL_PATH)
        else:
            joblib.dump(sgd_model, CONTENT_SGD_MODEL_PATH)
        return sgd_model

    def __fit_by_linearsvc(self, x_train, y_train, is_title):
        linearsvc_model = LinearSVC(random_state=42)
        linearsvc_model.fit(x_train, y_train)
        if is_title:
            joblib.dump(linearsvc_model, TITLE_LINEAR_SVC_MODEL_PATH)
        else:
            joblib.dump(linearsvc_model, CONTENT_LINEAR_SVC_MODEL_PATH)
        return linearsvc_model

    def __measure_coef(self, model, vectorizer, transformer, x_test, y_test, is_title):
        x_test = vectorizer.transform(x_test)
        x_test = transformer.transform(x_test)
        y_pred = model.predict(x_test)
        self.logger.info("accuracy: %s" % accuracy_score(y_test, y_pred))
        self.logger.info("confusion_matrix: %s" % confusion_matrix(y_test, y_pred))
        feature_names = np.array(vectorizer.get_feature_names())
        coef_index = model.coef_[0]
        df = pd.DataFrame({"Word":feature_names, "Coef": coef_index})
        self.logger.info(df.sort_values("Coef"))
        if is_title:
            df.sort_values("Coef").to_csv("data/features_coef_title_" + type(model).__name__ + ".csv", index=False, header=None)
        else:
            df.sort_values("Coef").to_csv("data/features_coef_content_" + type(model).__name__ + ".csv", index=False, header=None)

    def create_baseline_classifiers(self, seed=42):
        models = []
        models.append(('log', LogisticRegression(random_state=seed))) # not applicable for NLP
        models.append(('sgdc', SGDClassifier(random_state=seed)))
        models.append(('etc', ExtraTreesClassifier(random_state=seed))) # very slow of training time
        models.append(('gbmc', GradientBoostingClassifier(random_state=seed))) # poor accuracy
        models.append(('rfc', RandomForestClassifier(random_state=seed))) # slow of training time
        models.append(('svc', SVC(random_state=seed, probability=True))) # very slow of training time
        models.append(('mnb', MultinomialNB()))
        models.append(('xgbc', XGBClassifier(seed=seed))) # poor accuracy
        models.append(('lsvc', LinearSVC(random_state=seed)))
        models.append(('knn', KNeighborsClassifier(n_neighbors=5))) # not applicable for NLP
        models.append(('dtc', DecisionTreeClassifier(random_state=seed))) # slow of training time
        models.append(('bc', BaggingClassifier())) # not applicable

        # regression methods are for targeting confidence interval only
        # models.append(('ols', LinearRegression()))
        # models.append(('sgdr', SGDRegressor(random_state=seed)))
        # models.append(('etr', ExtraTreesRegressor(random_state=seed)))
        # models.append(('gbmr', GradientBoostingRegressor(random_state=seed)))
        # models.append(('rfr', RandomForestRegressor(random_state=seed)))
        # models.append(('svr', SVR()))
        # models.append(('xgbr', XGBRegressor(seed=seed)))
        return models

    def assess_models(self, x_train, y_train, x_test, y_test, models, cv=5, metrics=['roc_auc', 'f1'], is_title=False):
        summary = pd.DataFrame()
        file_name = ''
        encoded_y_train = self.label_encode_target(y_train)
        for model_name, model in models:
            try:
                self.logger.info("processing {}".format(model_name))
                result = pd.DataFrame(cross_validate(model, x_train, encoded_y_train, cv=cv, scoring=metrics))
                mean = result.mean().rename('{}_mean'.format)
                std = result.std().rename('{}_std'.format)
                summary[model_name] = pd.concat([mean, std], axis=0)
                # model.fit(x_train, y_train)
                # x_test = vectorizer.transform(x_test)
                # x_test = transformer.transform(x_test)
                # y_pred = model.predict(x_test)
                # logger.info("accuracy: %s" % accuracy_score(y_test, y_pred))
                # logger.info("confusion_matrix: %s" % confusion_matrix(y_test, y_pred))
                # if is_title:
                #     joblib.dump(vectorizer, "predictor/train_title_vectorizer.m")
                #     joblib.dump(transformer, "predictor/train_title_transformer.m")
                #     joblib.dump(model, "predictor/train_title_" + model_name + "_model.m")
                # else:
                #     joblib.dump(vectorizer, "predictor/train_content_vectorizer.m")
                #     joblib.dump(transformer, "predictor/train_content_transformer.m")
                #     joblib.dump(model, "predictor/train_content_" + model_name + "_model.m")
                # feature_names = np.array(vectorizer.get_feature_names())
                # coef_index = model.coef_[0]
                # features_df = pd.DataFrame({"Word":feature_names, "Coef": coef_index})
                # features_df.sort_values("Coef").to_csv('data/features_coef_{}.csv'.format(model_name), index=False, header=None)
                # scores = cross_val_score(model, x_test, y_pred, cv=10, scoring = 'accuracy')
                # logger.info("The mean score for {}: {:.3f}".format(model_name, scores.mean()))
            except Exception as e:
                message = "Exception in assess_models: %s" % e
                self.logger.error("error for models: {}".format(model_name))
                self.logger.exception(message + str(e))
        summary.sort_index(inplace=True)
        if is_title:
            file_name = PREDICTOR_PATH + 'title_models_summary.csv'
        else:
            file_name = PREDICTOR_PATH + 'content_models_summary.csv'
        summary.T.to_csv(file_name, sep='\t', encoding='utf-8')
        return summary

    def extract_metric(self, summary, metric):
        output = summary[summary.index.str.contains(metric)].T
        output.columns = output.columns.str.replace(f'test_{metric}_', '')
        output.sort_values(by='mean', ascending=False, inplace=True)
        output['lower'] = output['mean'] - 2*output['std']
        output['upper'] = output['mean'] + 2*output['std']
        return output

    def label_encode_target(self, y):
        lb_make = LabelEncoder()
        encoded_y = lb_make.fit_transform(y)
        print(encoded_y)
        return encoded_y

if __name__ == '__main__':
    logger = Logger("MainLogger").setup_system_logger()
    clf = Classifier()
    # clf.train_for_init()
    clf.evalute_models()
    print("done")
    # clf.train_for_increment(text = '市場下調匯豐(00005.HK)目標價至65.5元', expect_output = "negative", is_title = True)