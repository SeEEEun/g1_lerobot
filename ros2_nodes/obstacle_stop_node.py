#!/usr/bin/env python3
"""
obstacle_stop_node.py — 노트북에서 실행 (ROS2 Humble, Ubuntu 22.04)
/camera/depth/image_raw 구독 → 중앙 ROI depth < 0.6m → /obstacle/stop True 퍼블리시

구독 토픽:
  /camera/depth/image_raw   sensor_msgs/Image  (16UC1)

퍼블리시 토픽:
  /obstacle/stop            std_msgs/Bool

실행:
  source /opt/ros/humble/setup.bash
  export ROS_DOMAIN_ID=0
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  python3 obstacle_stop_node.py
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

DEPTH_THRESHOLD_M = 0.6    # 정지 거리 (미터)
DEPTH_SCALE       = 0.001  # RealSense D435I 기본값: 1 count = 0.001 m
ROI_CENTER_FRAC   = 0.3    # 중앙 30% × 30% 영역 검사


class ObstacleStopNode(Node):

    def __init__(self):
        super().__init__('obstacle_stop')

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self._callback, best_effort)
        self.pub = self.create_publisher(Bool, '/obstacle/stop', 10)

        self.get_logger().info(
            f'ObstacleStop ready  threshold={DEPTH_THRESHOLD_M}m  '
            f'ROI={int(ROI_CENTER_FRAC*100)}% center'
        )

    # ------------------------------------------------------------------ #

    def _callback(self, msg: Image):
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width)

        # 중앙 ROI
        h, w = depth.shape
        r = ROI_CENTER_FRAC / 2
        y0, y1 = int(h * (0.5 - r)), int(h * (0.5 + r))
        x0, x1 = int(w * (0.5 - r)), int(w * (0.5 + r))
        roi = depth[y0:y1, x0:x1]

        valid = roi[roi > 0]
        if len(valid) == 0:
            return

        # 최근접 5th percentile (노이즈 제거)
        min_dist_m = float(np.percentile(valid, 5)) * DEPTH_SCALE

        stop = min_dist_m < DEPTH_THRESHOLD_M
        self.pub.publish(Bool(data=stop))

        if stop:
            self.get_logger().warn(
                f'OBSTACLE DETECTED: {min_dist_m:.3f}m < {DEPTH_THRESHOLD_M}m → STOP',
                throttle_duration_sec=1.0,
            )


def main():
    rclpy.init()
    node = ObstacleStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
