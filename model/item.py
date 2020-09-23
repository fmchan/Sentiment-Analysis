from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, create_engine, ForeignKey, Table
from sqlalchemy.orm import relationship
from configs.base import Base
from configs.settings import DB_PATH

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key = True)
    link = Column(String(100), nullable = False)
    category = Column(String(10), nullable = False)
    title = Column(String(100), nullable = False)
    published_time = Column(DateTime, nullable = True)
    bullish_count = Column(Integer, nullable = False)
    bearish_count = Column(Integer, nullable = False)
    text = Column(String(500), nullable = True)
    created_date = Column(DateTime, nullable = False)

    def __init__(self, link, category, title, published_time, bullish_count, bearish_count, text, created_date):
        self.link = link
        self.category = category
        self.title = title
        self.published_time = published_time
        self.bullish_count = bullish_count
        self.bearish_count = bearish_count
        self.text = text
        self.created_date = created_date

    def dict(self):
        return {'id': self.id,
                'link': self.link,
                'category': self.category,
                'title': self.title,
                'published_time': self.published_time,
                'bullish_count': self.bullish_count,
                'bearish_count': self.bearish_count,
                'text': self.text,
                'created_date': self.created_date
               }

engine = create_engine(DB_PATH)
Base.metadata.create_all(engine)