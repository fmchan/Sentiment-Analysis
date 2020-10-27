#!/bin/bash
PID=$(cat /usr/local/apps/sentiment-analysis/app.pid)

if ! ps -p $PID > /dev/null
then
  rm -rf /usr/local/apps/sentiment-analysis/app.pid
  nohup /root/miniconda3/envs/sentiment-analysis/bin/python3.6 /usr/local/apps/sentiment-analysis/app.py >> sentiment.log & echo $! >> /usr/local/apps/sentiment-analysis/app.pid
fi

#*/1 * * * * /usr/local/apps/sentiment-analysis/restart_app.sh
