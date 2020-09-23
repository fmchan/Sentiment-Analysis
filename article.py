from urllib.request import urlopen
import urllib.parse
import simplejson
import more_itertools as mit
from datetime import datetime, timedelta
from predictor import Predictor
from classifier import Classifier
import ujson
import pandas as pd
import joblib
from util.logger import Logger
import logging

def get_articles(input_date="2020-01-01"):
    classifer = Classifier()
    predictor = Predictor()
    logger = logging.getLogger("MainLogger")
    size = 500
    # input_date = "2020-03-16"
    previous_date = datetime.strftime(datetime.strptime(input_date, '%Y-%m-%d') - timedelta(1), '%Y-%m-%d')

    connection = urlopen('http://192.168.2.47:8983/solr/IetArticle2/select?fl=articleId,%20displayStart,%20headlineMain,%20contentText&fq=channelId:010%20AND%20displayEnd:([NOW%20TO%20*]%20OR%20(*:*%20NOT%20displayEnd:[*%20TO%20*]))%20AND%20displayStart:[' + previous_date + 'T16:00:00Z%20TO%20' + input_date + 'T15:59:59Z]&indent=on&q=*:*&rows=' + str(size) + '&sort=displayStart%20asc&wt=json')
    raw_response = ujson.load(connection)
    articles = []
    stat = []
    total_count = 0
    positive_title_count = 0
    positive_content_count = 0
    title_model, content_model = classifer.reload_model()
    for article in raw_response["response"]["docs"]:
        link = "https://inews.hket.com/article/" + str(article["articleId"])
        # publish_date = article["displayStart"]

        item = {"link": link}
        total_count += 1
        # item["publish_date"] = publish_date

        if "headlineMain" in article:
            title = article["headlineMain"]
            title_sentiment, title_sentiment_probability = predictor.transform_predict(title, True, title_model, content_model)
            item["title"] = title
            item["sentiment_title"] = title_sentiment
            item["probability_title"] = title_sentiment_probability
            if title_sentiment == "positive":
                positive_title_count += 1

        if "contentText" in article:
            content = article["contentText"]
            content_sentiment, content_sentiment_probability = predictor.transform_predict(content, False, title_model, content_model)
            item["content"] = content
            item["sentiment_content"] = content_sentiment
            item["probability_content"] = content_sentiment_probability
            if content_sentiment == "positive":
                positive_content_count += 1
        articles.append(item)

    stat_item = {"publish_date": input_date}
    stat_item["total_articles"] = str(total_count)
    if total_count > 0:
        stat_item["positive_title_over_total"] = str(round(positive_title_count / total_count * 100, 2)) + "%"
        stat_item["positive_content_over_total"] = str(round(positive_content_count / total_count * 100, 2)) + "%"
    stat.append(stat_item)

    json_articles = ujson.dumps(articles, ensure_ascii=False)
    json_stat = ujson.dumps(stat_item, ensure_ascii=False)
    return json_stat, json_articles