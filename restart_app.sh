#!/bin/bash
PID=$(cat /usr/local/apps/ETNet-Sentiment-Analysis-Master/app.pid)

if ! ps -p $PID > /dev/null
then
  rm -rf /usr/local/apps/ETNet-Sentiment-Analysis-Master/app.pid
  nohup /root/miniconda3/bin/python3.7 /usr/local/apps/ETNet-Sentiment-Analysis-Master/app.py >> sentiment.log & echo $! >> /usr/local/apps/ETNet-Sentiment-Analysis-Master/app.pid
fi

#*/1 * * * * /usr/local/apps/ETNet-Sentiment-Analysis-Master/restart_app.sh
