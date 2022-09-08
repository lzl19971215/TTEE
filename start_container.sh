NAME=$1
docker run -itd --name ${NAME} --gpus all -p 5320:22 -p 5321:5321 -p 5322:5322 -p 5323:5323 -v /data2/liutuozhen/lzl:/lzl tensorflow/tensorflow:2.7.0-gpu-jupyter 
