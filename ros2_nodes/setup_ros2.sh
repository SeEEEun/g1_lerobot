#!/bin/bash
# setup_ros2.sh — 노트북에서 source하면 ROS2 Humble 환경 세팅
#
# 사용법:
#   source ~/g1_lerobot/ros2_nodes/setup_ros2.sh

source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID=0

echo "[ROS2] Humble sourced"
echo "[DDS]  RMW=default(FastDDS)  DOMAIN=${ROS_DOMAIN_ID}"
echo "[NET]  G1 PC2=192.168.123.123  Laptop=192.168.123.199"
