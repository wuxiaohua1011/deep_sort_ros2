from setuptools import setup
import os
from glob import glob
from urllib import request

package_name = 'deep_sort_ros2'
submodules = "deep_sort_ros2/deep_sort"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,
              "deep_sort_ros2/deep_sort",
              "deep_sort_ros2/core",
              "deep_sort_ros2/data",
              "deep_sort_ros2/tools"],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join(os.path.join('share', package_name), "configs"), glob('configs/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roar',
    maintainer_email='wuxiaohua1011@berkeley.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "yolov4_detector_node = deep_sort_ros2.yolov4_detector_node:main",
            "deep_sort_tracker_node = deep_sort_ros2.deep_sort_tracker_node:main",
            "test_video_player_node = deep_sort_ros2.test_video_player_node:main"
        ],
    },
)
