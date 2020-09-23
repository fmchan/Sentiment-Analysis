#-*- coding: utf-8 -*-
from sqlalchemy import create_engine, and_, or_, any_, DDL
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy import exc
from model.item import Item
from datetime import date, timedelta
import datetime as dt
from configs.settings import DB_PATH
from configs.base import Session, engine, Base
from util.logger import Logger
import logging
import json
import os
from datetime import datetime
import sqlite3
import csv
import pandas
from datetime import timedelta

class DBHelper:
    def __init__(self):
        self.encoding = 'utf-8'
        self.logger = logging.getLogger("MainLogger")

    def insert_item(self, link, category, title, published_time, bullish_count, bearish_count, text):
        try:
            self.logger.info("inserting item for link: {}".format(link))
            session = Session()
            result = session.query(Item).filter_by(title = title).first()
            if result is None:
                item = Item(link = link, category = category, title = title, published_time = published_time, bullish_count = bullish_count, bearish_count = bearish_count, text = text, created_date = datetime(datetime.today().year, datetime.today().month, datetime.today().day))
                session.add(item)
            else:
                self.logger.info("record existed, updating item for link: {}".format(link))
                result.bullish_count = bullish_count
                result.bearish_count = bearish_count
                result.category = category
                result.created_date = datetime.now()
            session.commit()
            return "done"
        except Exception as e:
            message = "Exception in insert_item: %s" % e
            self.logger.exception(message + str(e))
            return message
        finally:
            session.close()

    def is_title_processed_for_today(self, title):
        try:
            todays_datetime = datetime(datetime.today().year, datetime.today().month, datetime.today().day)
            session = Session()
            item = session.query(Item.title).filter_by(created_date = todays_datetime).filter(Item.title.like('%'+title+'%')).first()
            if item is None:
                return False
            else:
                return True
        except Exception as e:
            message = "Exception in is_title_processed_for_today: %s" % e
            self.logger.exception(message)
            return ''
        finally:
            session.close()

    def query_titles_from_bullish_items(self):
        try:
            session = Session()
            return [r[0] for r in session.query(Item.title).filter(Item.bullish_count * 1 > Item.bearish_count * 1, Item.category == 'positive').order_by(Item.published_time.desc()).all()]
        except Exception as e:
            message = "Exception in query_titles_from_bullish_items: %s" % e
            self.logger.exception(message)
            return ''
        finally:
            session.close()

    def query_titles_from_bearish_items(self):
        try:
            session = Session()
            return [r[0] for r in session.query(Item.title).filter(Item.bullish_count * 1 < Item.bearish_count * 1, Item.category == 'negative').order_by(Item.published_time.desc()).all()]
        except Exception as e:
            message = "Exception in query_titles_from_bearish_items: %s" % e
            self.logger.exception(message)
            return ''
        finally:
            session.close()

    def query_contents_from_bullish_items(self):
        try:
            session = Session()
            return [r[0] for r in session.query(Item.text).filter(Item.bullish_count * 1 > Item.bearish_count * 1, Item.category == 'positive').order_by(Item.published_time.desc()).all()]
        except Exception as e:
            message = "Exception in query_texts_from_bullish_items: %s" % e
            self.logger.exception(message)
            return ''
        finally:
            session.close()

    def query_contents_from_bearish_items(self):
        try:
            session = Session()
            return [r[0] for r in session.query(Item.text).filter(Item.bullish_count * 1 < Item.bearish_count * 1, Item.category == 'negative').order_by(Item.published_time.desc()).all()]
        except Exception as e:
            message = "Exception in query_texts_from_bearish_items: %s" % e
            self.logger.exception(message)
            return ''
        finally:
            session.close()

    def query_all_items(self):
        try:
            session = Session()
            return session.query(Item.category, Item.title, Item.text, Item.published_time, Item.bullish_count, Item.bearish_count).order_by(Item.published_time.desc()).all()
        except Exception as e:
            message = "Exception in query_all_items: %s" % e
            self.logger.exception(message)
            return ''
        finally:
            session.close()

    def custom_alter(self):
        add_column = DDL('ALTER TABLE items ADD COLUMN newcol VARCHAR(300)')
        engine.execute(add_column)

    def update_items(self):
        try:
            session = Session()
            session.query(Item).filter(Item.bullish_count * 1 == Item.bearish_count * 1).update({"category": 'positive'})
            session.commit()
            return "done"
        except Exception as e:
            message = "Exception in insert_item: %s" % e
            self.logger.exception(message + str(e))
            return message
        finally:
            session.close()

if __name__ == "__main__":
    logger = Logger("MainLogger").setup_system_logger()
    db = DBHelper()
    items = db.query_all_items()
    for item in items:
        # print(vars(item).items())
        print(item)
        print()
    print()
    # db.update_items()
    # items = db.query_titles_from_bullish_items()
    # items = db.query_titles_from_bearish_items()
    # print(len(items))
    # for item in items:
        # print(item)
        # print()
    # print(db.is_title_processed_for_today('恆指跌幅曾擴至逾400點略失27,300  騰訊/聯通/吉利/石藥/手機股領跌'))