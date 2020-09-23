import schedule
import time
from service.aastock_scrapy import AASTOCK
from util.logger import Logger
from apscheduler.schedulers.background import BackgroundScheduler

def aastock_sentiment_news_job():
    logger.info("aastock_sentiment_news_job scheduler start scraping news")
    aastock = AASTOCK()
    aastock.get_market_news()
    logger.info("scheduler task end")

if __name__ == "__main__":
    logger = Logger("MainLogger").setup_system_logger()
    logger.info("extractor scheduler just restarted")
    # schedule.every().day.at('23:50').do(aastock_sentiment_news_job)
    scheduler = BackgroundScheduler()
    scheduler.add_job(aastock_sentiment_news_job, 'cron', hour='23', minute='50')
    scheduler.start()
    while True:
        schedule.run_pending()
        time.sleep(1)
