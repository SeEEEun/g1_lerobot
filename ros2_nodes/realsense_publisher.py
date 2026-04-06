#!/usr/bin/env python3
"""
realsense_publisher.py — G1 PC2에서 실행 (ROS2 Foxy, Ubuntu 20.04)
Intel RealSense D435I → ROS2 토픽 퍼블리시

퍼블리시 토픽:
  /camera/color/image_raw   sensor_msgs/Image  (bgr8, 640x480 @ 30Hz)
  /camera/depth/image_raw   sensor_msgs/Image  (16UC1, 640x480 @ 30Hz)
  /camera/color/camera_info sensor_msgs/CameraInfo

실행:
  source /opt/ros/foxy/setup.bash
  export ROS_DOMAIN_ID=0
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  python3 realsense_publisher.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import pyrealsense2 as rs
import numpy as np


WIDTH  = 640
HEIGHT = 480
FPS    = 30
FRAME_ID = 'camera_color_optical_frame'


class RealSensePublisher(Node):

    def __init__(self):
        super().__init__('realsense_publisher')

        # --- Publishers ---
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.color_pub = self.create_publisher(Image,      '/camera/color/image_raw',   qos)
        self.depth_pub = self.create_publisher(Image,      '/camera/depth/image_raw',   qos)
        self.info_pub  = self.create_publisher(CameraInfo, '/camera/color/camera_info', qos)

        # --- RealSense ---
        self.pipeline = rs.pipeline()
        self.align    = rs.align(rs.stream.color)

        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)

        profile = self.pipeline.start(cfg)
        depth_sensor      = profile.get_device().first_depth_sensor()
        self.depth_scale  = depth_sensor.get_depth_scale()
        self.intr         = (profile.get_stream(rs.stream.color)
                             .as_video_stream_profile().get_intrinsics())

        self.get_logger().info(
            f'RealSense D435I OK  {WIDTH}x{HEIGHT}@{FPS}fps  '
            f'depth_scale={self.depth_scale:.6f} m/count  '
            f'fx={self.intr.fx:.1f} fy={self.intr.fy:.1f}'
        )

        # --- Timer ---
        self.timer = self.create_timer(1.0 / FPS, self._publish)

    # ------------------------------------------------------------------ #

    def _make_camera_info(self, stamp) -> CameraInfo:
        i = self.intr
        msg = CameraInfo()
        msg.header.stamp    = stamp
        msg.header.frame_id = FRAME_ID
        msg.width  = i.width
        msg.height = i.height
        msg.k = [i.fx, 0.0, i.ppx,
                 0.0, i.fy, i.ppy,
                 0.0, 0.0, 1.0]
        msg.d = list(i.coeffs)
        msg.distortion_model = 'plumb_bob'
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        msg.p = [i.fx, 0.0, i.ppx, 0.0,
                 0.0, i.fy, i.ppy, 0.0,
                 0.0, 0.0, 1.0,   0.0]
        return msg

    def _publish(self):
        frames  = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        stamp = self.get_clock().now().to_msg()

        # ── Color ────────────────────────────────────────────────────── #
        color_bgr = np.asanyarray(color_frame.get_data())   # uint8 H×W×3
        c = Image()
        c.header.stamp    = stamp
        c.header.frame_id = FRAME_ID
        c.height   = color_bgr.shape[0]
        c.width    = color_bgr.shape[1]
        c.encoding = 'bgr8'
        c.step     = c.width * 3
        c.data     = color_bgr.tobytes()

        # ── Depth ────────────────────────────────────────────────────── #
        depth_arr = np.asanyarray(depth_frame.get_data())   # uint16 H×W
        d = Image()
        d.header.stamp    = stamp
        d.header.frame_id = FRAME_ID
        d.height   = depth_arr.shape[0]
        d.width    = depth_arr.shape[1]
        d.encoding = '16UC1'
        d.step     = d.width * 2
        d.data     = depth_arr.tobytes()

        self.color_pub.publish(c)
        self.depth_pub.publish(d)
        self.info_pub.publish(self._make_camera_info(stamp))

    # ------------------------------------------------------------------ #

    def destroy_node(self):
        self.pipeline.stop()
        self.get_logger().info('RealSense pipeline stopped.')
        super().destroy_node()


def main():
    rclpy.init()
    node = RealSensePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
