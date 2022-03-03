#!/usr/bin/env python3
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TestVideoPlayerNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("test_video_player")
        self.declare_parameter('video_path', 'NA')
        self.timer = self.create_timer(0.1, self.callback)
        video_path = self.get_parameter(
            'video_path').get_parameter_value().string_value
        self.get_logger().info(f"{video_path}")
        self.cap = cv2.VideoCapture(video_path)
        self.img_pub = self.create_publisher(Image, 'test_image', 10)
        self.bridge = CvBridge()

    def callback(self):
        ret_val, frame = self.cap.read()

        if ret_val:
            rgb_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            rgb_img_msg.header.frame_id = 'base_link'
            self.img_pub.publish(rgb_img_msg)
        else:
            video_path = self.get_parameter(
                'video_path').get_parameter_value().string_value
            self.cap = cv2.VideoCapture(video_path)


def main(args=None):
    rclpy.init()
    node = TestVideoPlayerNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
