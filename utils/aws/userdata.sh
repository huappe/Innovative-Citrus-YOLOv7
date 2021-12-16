#!/bin/bash
# AWS EC2 instance startup script https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html
# This script will run only once on first instance start (for a re-start script see mime.sh)
# /home/ubuntu (ubuntu) or /home/ec2-user (amazon-linux) is working dir
# Use >300 GB SSD

cd home/ubuntu
if [ ! -d yolor ]; then
  echo "Running first-time script." # install dependencies, download COCO, pull Docker
  git clone -b main https://github.com/WongKinYiu/yolov7 && sudo chmod -R 777 yolov7
  cd yolov7
  bash d