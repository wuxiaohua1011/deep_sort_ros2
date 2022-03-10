#!/usr/bin/env python3
from urllib import request
# Define the remote file to retrieve
remote_url = 'https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT'
# Define the local filename to save data
local_file = 'deep_sort_ros2/data/yolov4.weights'
# Download remote and save locally
request.urlretrieve(remote_url, local_file)

remote_url = 'https://drive.google.com/file/d/1P1TtmbWkBlJFnwuUa7lNV2k-aT4TG2BK/view?usp=sharing'
# Define the local filename to save data
local_file = 'configs/yolov4.h5'
# Download remote and save locally
request.urlretrieve(remote_url, local_file)
print("Download success!")


