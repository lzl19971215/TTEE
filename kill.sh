ps -aux | grep train.py | grep -v grep | awk '{printf $2}' | xargs kill -9