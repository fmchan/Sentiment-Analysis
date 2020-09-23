import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import jieba
import jieba.analyse
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from warnings import simplefilter
import re
import time
from predictor import Predictor
from classifier import Classifier
from configs.settings import HOST, PORT
from configs.settings import BULLISH_CONTENTS_ORIGINAL_FILE_PATH, BULLISH_TITLES_ORIGINAL_FILE_PATH, BEARISH_CONTENTS_ORIGINAL_FILE_PATH, BEARISH_TITLES_ORIGINAL_FILE_PATH, BULLISH_TITLES_FEEDBACK_FILE_PATH, BEARISH_TITLES_FEEDBACK_FILE_PATH, BULLISH_CONTENTS_FEEDBACK_FILE_PATH, BEARISH_CONTENTS_FEEDBACK_FILE_PATH, TITLE_VECTORIZER_PATH, CONTENT_VECTORIZER_PATH, TITLE_TRANSFORMER_PATH, CONTENT_TRANSFORMER_PATH, TITLE_SGD_MODEL_PATH, CONTENT_SGD_MODEL_PATH, TITLE_LINEAR_SVC_MODEL_PATH, CONTENT_LINEAR_SVC_MODEL_PATH
from configs.settings import STOP_WORDS_FILE_PATH, CUSTOM_DICT_PATH

from util.logger import Logger

logger = Logger("MainLogger").setup_system_logger()

simplefilter(action='ignore', category=FutureWarning)

jieba.load_userdict(CUSTOM_DICT_PATH)
stop_list = [line.strip() for line in open(STOP_WORDS_FILE_PATH, 'r', encoding='utf-8').readlines()] 

app = Flask(__name__)
api = Api(app)

class MakePrediction(Resource):
    @staticmethod
    def get():
        predictor = Predictor()
        classifer = Classifier()
        title_model, content_model = classifer.reload_model()
        text = ""
        posted_data = request.get_json()
        text = posted_data["text"]
        category = posted_data["type"]

        if category and category == "title":
            result, probability = predictor.transform_predict(text, True, title_model, content_model)
        elif category and category == "content":
            result, probability = predictor.transform_predict(text, False, title_model, content_model)
        else:
            return "Error: Invalid type", 400

        return jsonify({
            'result': result,
            'probability': probability[0] if probability[0] > probability[1] else probability[1]
        })

api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', PORT))
    app.run(host=HOST, port=port, debug=True, threaded=True)