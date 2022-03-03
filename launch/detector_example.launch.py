from asyncio import base_subprocess
from email.mime import base

from sqlalchemy import true
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
import launch_ros
from pathlib import Path


def generate_launch_description():
    base_path = os.path.realpath(get_package_share_directory(
        'deep_sort_ros2'))
    yolov4_weight_path = (Path(base_path) / "configs" /
                          "yolov4.h5").as_posix()
    video_path = (Path(base_path) / "configs" / "cars.mp4").as_posix()

    return LaunchDescription([
        Node(
            package='deep_sort_ros2',
            executable='yolov4_detector_node',
            name='yolov4_detector',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'yolo_v4_model_path': yolov4_weight_path},
                {"image_topic": "test_image"},
                {"debug": True}
            ]
        ),
        Node(
            package='deep_sort_ros2',
            executable='test_video_player_node',
            name='test_video_player_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'video_path': video_path}
            ]
        )
    ])
