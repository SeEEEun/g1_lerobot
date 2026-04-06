#!/bin/bash
# setup_ros2.sh — 노트북에서 source하면 ROS2 Humble + CycloneDDS 크로스네트워크 설정 완료
#
# 사용법:
#   source ~/g1_lerobot/ros2_nodes/setup_ros2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${SCRIPT_DIR}/cyclonedds_laptop.xml"

echo "[ROS2] Humble sourced"
echo "[DDS]  RMW=rmw_cyclonedds_cpp  DOMAIN=${ROS_DOMAIN_ID}"
echo "[DDS]  CYCLONEDDS_URI=${CYCLONEDDS_URI}"
echo "[NET]  Peer=192.168.123.123 (G1 PC2) via enp46s0"
