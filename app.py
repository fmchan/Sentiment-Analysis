#-*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import datetime as dt
from util.logger import Logger
import json
from configs.settings import HOST, PORT
import article
from classifier import Classifier
import pandas as pd
from predictor import Predictor
import service.stock_price as price
# from flask_caching import Cache

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['JSON_AS_ASCII'] = False
logger = Logger("MainLogger").setup_system_logger()

# cache = Cache(config={'CACHE_TYPE': 'simple'})
# cache.init_app(app)

def make_cache_key(*args, **kwargs):
    return request.form.get("publish_date")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_news', methods=['GET', 'POST'])
# @cache.cached(timeout=1800, key_prefix=make_cache_key) # 1 hour = 3600 # disable cache for incremental learning
def get_news():
    publish_date = request.form.get("publish_date")
    json_articles = []
    json_stat = []
    current_price = 0
    price_diff = 0
    price_diff_pct_change = 0
    hs_df = price.get_hist_stock_price('^HSI', publish_date)
    if len(hs_df.index) > 0:
        current_price = round(hs_df.iloc[-1].Close, 2)
        price_diff = round(hs_df.iloc[-1].Close - hs_df.iloc[-2].Close, 2)
        price_diff_pct_change = round(price_diff / hs_df.iloc[-2].Close * 100, 2)
    if publish_date is not None:
        publish_date = '{0:%Y-%m-%d}'.format(datetime.strptime(publish_date, '%Y-%m-%d'))
        logger.info("start retrieving news on [%s]", publish_date)
        json_stat, json_articles = article.get_articles(publish_date)
    if json_stat and json_articles:
        # return json2html.convert(json = jsonData, clubbing = False)
        return render_template('details.html', is_input=True, json_stat=json.loads(json_stat), json_articles=json.loads(json_articles), current_price=current_price, price_diff_pct_change=price_diff_pct_change)
    else:
        return "no news found for that date", publish_date

@app.route('/save_prediction', methods=['POST'])
def save_prediction():
    classifier = Classifier()
    title = request.form["title"]
    model_title_output = request.form["sentiment_title"]
    content = request.form["content"]
    model_content_output = request.form["sentiment_content"]
    expect_output = request.form["expect_output"]

    if(model_title_output != expect_output): # check for title
        classifier.train_for_increment(title, expect_output, True)

    if(model_content_output != expect_output): # check for content
        classifier.train_for_increment(content, expect_output, False)

    html = """<HTML>
        <body>
            <h1>Thank you! We've just learnt it</h1>
            <table>
                {0}
            </table>
        </body>
    </HTML>"""
    return html.format("<button onclick='self.close()'>Close</button>")

@app.route('/get_analysis', methods=['GET', 'POST'])
def get_analysis():
    classifer = Classifier()
    predictor = Predictor()
    title_model, content_model = classifer.reload_model()
    is_input = request.form["is_input"]
    if is_input == "true":
        text = request.form["text"]
        logger.info("input text [%s]", text)
        title = text
        content = text
        title_sentiment, title_sentiment_probability = predictor.transform_predict(title, True, title_model, content_model)
        content_sentiment, content_sentiment_probability = predictor.transform_predict(content, False, title_model, content_model)
    else:
        title = request.form["title"]
        content = request.form["content"]
        # content = request.values.get('content')
        # logger.info(content)
        title_sentiment, title_sentiment_probability = predictor.transform_predict(title, True, title_model, content_model)
        content_sentiment = ""
        content_sentiment_probability = ""
        if content:
            content_sentiment, content_sentiment_probability = predictor.transform_predict(content, False, title_model, content_model)

    result = {
        'title': title,
        'analyzed_title': classifer.analyze_text(title, True),
        'sentiment_title': title_sentiment,
        'probability_title': title_sentiment_probability,
        'content': content,
        'analyzed_content': classifer.analyze_text(content, False),
        'sentiment_content': content_sentiment,
        'probability_content': content_sentiment_probability,
    }
    logger.info(result)
    return render_template('feedback.html', is_input=is_input, result=result)

if __name__ == '__main__':
	port = int(os.environ.get('PORT', PORT))
	app.run(host = HOST, port = port, debug = True) # flask is in threaded mode by default
