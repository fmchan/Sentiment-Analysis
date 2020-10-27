from pandas_datareader import data
import pandas as pd
from datetime import datetime, timedelta
import logging

def get_hist_stock_price(sid, selected_date='2001-07-15', provider='YAHOO', market='HK'):
    try:
        date_before_selected_date = datetime.strptime(selected_date, '%Y-%m-%d') - timedelta(1)
        df = pd.DataFrame()
        if provider == 'YAHOO':
            df = data.get_data_yahoo(sid, start=date_before_selected_date, end=selected_date, retry_count=3, pause=1)
    except Exception as e:
        logger = logging.getLogger("MainLogger")
        logger.exception(e)
        return pd.DataFrame()
    return df