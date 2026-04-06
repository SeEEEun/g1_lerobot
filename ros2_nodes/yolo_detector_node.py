#!/usr/bin/env python3
"""
yolo_detector_node.py — 노트북에서 실행 (ROS2 Humble, Ubuntu 22.04)
/camera/color/image_raw 구독 → YOLO best.pt 추론 → 결과 퍼블리시

구독 토픽:
  /camera/color/image_raw   sensor_msgs/Image  (bgr8)

퍼블리시 토픽:
  /yolo/detections          std_msgs/String    (JSON)
  /yolo/annotated_image     sensor_msgs/Image  (bgr8, bbox 시각화)

Detection JSON 구조:
  {
    "stamp_sec": int,
    "detections": [
      {"class": "yellow_cube", "conf": 0.92, "cx": 320.0, "cy": 240.0,
       "bbox": [x1, y1, x2, y2]}
    ]
  }

실행:
  source /opt/ros/humble/setup.bash
  export ROS_DOMAIN_ID=0
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  conda activate g1_grasp   # ultralytics 설치된 환경
  python3 yolo_detector_node.py
"""

import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO

MODEL_PATH = '/home/jairlab/best.pt'

# best.pt 학습 시 클래스 순서 (0~3)
CLASSES = ['blue_cube', 'blue_cylinder', 'yellow_cube', 'yellow_cylinder']

# Phase 1: cube만 사용 (cylinder는 미학습 데이터)
ACTIVE_CLASSES = {'yellow_cube', 'blue_cube'}


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f'YOLO loaded: {MODEL_PATH}')
        self.get_logger().info(f'Active classes: {ACTIVE_CLASSES}')

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._callback, best_effort)

        self.det_pub = self.create_publisher(String, '/yolo/detections',      10)
        self.img_pub = self.create_publisher(Image,  '/yolo/annotated_image', best_effort)

        self._frame_count = 0

    # ------------------------------------------------------------------ #

    def _callback(self, msg: Image):
        # sensor_msgs/Image (bgr8) → numpy uint8 H×W×3
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)

        results = self.model(arr, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f'cls{cls_id}'

            # Phase 1: cube 이외 필터링 (주석 해제하면 cylinder도 포함)
            if cls_name not in ACTIVE_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            cx = round((x1 + x2) / 2, 1)
            cy = round((y1 + y2) / 2, 1)

            detections.append({
                'class': cls_name,
                'conf':  round(conf, 3),
                'cx':    cx,
                'cy':    cy,
                'bbox':  [x1, y1, x2, y2],
            })

        # ── Publish detections (JSON) ─────────────────────────────────── #
        det_msg = String()
        det_msg.data = json.dumps({
            'stamp_sec': msg.header.stamp.sec,
            'detections': detections,
        })
        self.det_pub.publish(det_msg)

        # ── Publish annotated image ───────────────────────────────────── #
        annotated = results.plot()   # BGR numpy
        out = Image()
        out.header   = msg.header
        out.height   = annotated.shape[0]
        out.width    = annotated.shape[1]
        out.encoding = 'bgr8'
        out.step     = out.width * 3
        out.data     = annotated.tobytes()
        self.img_pub.publish(out)

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            n = len(detections)
            self.get_logger().info(f'frame={self._frame_count}  detections={n}')


def main():
    rclpy.init()
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
