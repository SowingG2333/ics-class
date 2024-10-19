#!/bin/sh

. /usr/local/Ascend/ascend-toolkit/set_env.sh

export PATH=/usr/local/python3.7/bin:$PATH
export PYTHONPATH=/usr/local/python3.7/lib/python3.7/site-packages:$PYTHONPATH

alias python3=python3.7

cd /home/HwHiAiUser/DK200I

# GoogleNet
airloader -m /home/HwHiAiUser/models/googlenet/googlenet.om -p 5530 &
sleep 3
python3 googlenet-applet/gnet_applet.py -r tcp://localhost:5530 -p 5531 --daemon
python3 googlenet-applet/gnet_applet.py -r file:///home/HwHiAiUser/models/googlenet/googlenet.om -p 5532 --daemon

# LeNet
airloader -m /home/HwHiAiUser/models/lenet/mnist.om -p 5540 &
sleep 3

# Picasso
airloader -m /home/HwHiAiUser/models/style_transfer/bijiasuo_fp32_nchw_no_aipp.om -p 5550 &
sleep 3
python3 styletransfer-applet/picasso_applet.py -r tcp://localhost:5550 -p 5551 --daemon
python3 styletransfer-applet/picasso_applet.py -r file:///home/HwHiAiUser/models/style_transfer/bijiasuo_fp32_nchw_no_aipp.om -p 5552 --daemon

# YOLOv4 608x608 for py apps
airloader -m /home/HwHiAiUser/models/yolov4/yolov4_bs1.om -p 5560 &
sleep 3
python3 yolov4_car_traffic-applet/yolov4_applet.py -r tcp://localhost:5560 -p 5561 --daemon
python3 yolov4_car_traffic-applet/yolov4_applet.py -r file:///home/HwHiAiUser/models/yolov4/yolov4_bs1.om -p 5562 --daemon

# YOLOv4 416x416 for cc apps
airloader -m /home/HwHiAiUser/models/yolov4/yolov4_416x416.om -p 5565 &

# Cartoon
airloader -m /home/HwHiAiUser/models/cartoon/cartoonization.om -p 5570 &
sleep 3
python3 cartoon-applet/cartoon_applet.py -r tcp://localhost:5570 -p 5571 --daemon
python3 cartoon-applet/cartoon_applet.py -r file:///home/HwHiAiUser/models/cartoon/cartoonization.om -p 5572 --daemon

# DeepLabv3+
airloader -m /home/HwHiAiUser/models/deeplabv3_plus/deeplabv3_plus.om -p 5580 &
sleep 3
python3 deeplabv3_plus-applet/deeplabv3_applet.py -r tcp://localhost:5580 -p 5581 --daemon
python3 deeplabv3_plus-applet/deeplabv3_applet.py -r file:///home/HwHiAiUser/models/deeplabv3_plus/deeplabv3_plus.om -p 5582 --daemon

# Clustering
CLUSTER_HOSTS=$*

echo "Pusblish service to host: ${CLUSTER_HOSTS}"
for HOST in $CLUSTER_HOSTS ;
do

  python3 styletransfer-applet/picasso_applet.py -r tcp://localhost:5550 -t tcpd -p 6551 -o ${HOST} --daemon
  python3 styletransfer-applet/picasso_applet.py -r file:///home/HwHiAiUser/models/style_transfer/bijiasuo_fp32_nchw_no_aipp.om \
    -t tcpd -p 6552 -o ${HOST} --daemon

  python3 yolov4_car_traffic-applet/yolov4_applet.py -r tcp://localhost:5560 -t tcpd -p 6561 -o ${HOST} --daemon
  python3 yolov4_car_traffic-applet/yolov4_applet.py -r file:///home/HwHiAiUser/models/yolov4/yolov4_bs1.om \
    -t tcpd -p 6562 -o ${HOST} --daemon

  python3 cartoon-applet/cartoon_applet.py -r tcp://localhost:5570 -t tcpd -p 6571 -o ${HOST} --daemon
  python3 cartoon-applet/cartoon_applet.py -r file:///home/HwHiAiUser/models/cartoon/cartoonization.om \
    -t tcpd -p 6572 -o ${HOST} --daemon

  python3 deeplabv3_plus-applet/deeplabv3_applet.py -r tcp://localhost:5580 -t tcpd -p 6581 -o ${HOST} --daemon
  python3 deeplabv3_plus-applet/deeplabv3_applet.py -r file:///home/HwHiAiUser/models/deeplabv3_plus/deeplabv3_plus.om \
    -t tcpd -p 6582 -o ${HOST} --daemon

done
# Done.
