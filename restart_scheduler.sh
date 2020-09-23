#!/bin/bash
PID=$(cat /usr/local/apps/Training-Data-Extractor/scheduler.pid)

if ! ps -p $PID > /dev/null
then
  rm -rf /usr/local/apps/Training-Data-Extractor/scheduler.pid
  nohup /usr/local/bin/python3.6 /usr/local/apps/Training-Data-Extractor/extractor_scheduler.py > extractor_scheduler.log & echo $! >> /usr/local/apps/Training-Data-Extractor/scheduler.pid
fi

#*/1 * * * * /usr/local/apps/Training-Data-Extractor/restart_scheduler.sh