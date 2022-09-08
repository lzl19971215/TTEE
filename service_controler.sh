ps -aux | grep absa_service.py | grep -v grep | awk '{printf $2}' | xargs kill -9
nohup python ./service/absa_service.py -u > /dev/null &
echo "service starting..."