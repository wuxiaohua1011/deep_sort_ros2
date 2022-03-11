#!/usr/bin/env python3
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


class TestNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("test_node")
        self.front_left_img = message_filters.Subscriber(
            self, Image, "/carla/ego_vehicle/front_left_rgb/image"
        )
        # self.front_left_cam_info = message_filters.Subscriber(
        #     self, Image, "/carla/ego_vehicle/front_left_rgb/camera_info"
        # )
        self.center_lidar = message_filters.Subscriber(
            self, PointCloud2, "/carla/ego_vehicle/center_lidar"
        )
        queue_size = 30
        self.ts = message_filters.TimeSynchronizer(
            [self.front_left_img, self.center_lidar], queue_size,
        )
        self.ts.registerCallback(self.callback)
        self.bridge = CvBridge()

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=500, height=500)
        self.pcd = o3d.geometry.PointCloud()
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.points_added = False

    def callback(self, left_cam_msg, center_lidar_pcl_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_cam_msg)
        # cv2.imshow("left_img", left_img)
        # cv2.waitKey(1)

        pcd_as_numpy_array = np.array(list(read_points(center_lidar_pcl_msg)))[:, :3]
        self.o3d_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pcd_as_numpy_array)
        )
        outlier_cloud = self.o3d_pcd  # .voxel_down_sample(voxel_size=2)

        # # normal removal method
        # outlier_cloud.estimate_normals()
        # normals = np.array(outlier_cloud.normals)
        # normals_avgs = np.average(normals, axis=0)
        # coords = np.where(normals[:, 2] < normals_avgs[2])
        # points = np.array(outlier_cloud.points)
        # outlier_cloud.points = o3d.utility.Vector3dVector(points[coords])
        # print(outlier_cloud)
        outlier_cloud.paint_uniform_color([0, 0, 0])
        # find plane
        plane_model, inliers = outlier_cloud.segment_plane(
            distance_threshold=1, ransac_n=10, num_iterations=1000
        )
        colors = np.array(outlier_cloud.colors)
        colors[inliers] = [1.0, 0.0, 0.0]
        outlier_cloud.colors = o3d.utility.Vector3dVector(colors)
        # inlier_cloud = outlier_cloud.select_by_index(inliers)
        # inlier_cloud.paint_uniform_color([1.0, 0, 0])
        # outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

        # remove things that are further and taller than a certain threshold
        # points = np.array(outlier_cloud.points)
        # coords = np.where(
        #     (points[:, 0] > 1) & (points[:, 2] < 1.5)
        # )  # further, and taller
        # points = points[coords]
        # outlier_cloud.points = o3d.utility.Vector3dVector(points)

        # outlier_cloud, ind = outlier_cloud.remove_radius_outlier(nb_points=5, radius=5)

        cloud = outlier_cloud  # self.o3d_pcd.select_by_index(ind, invert=True)
        self.non_blocking_pcd_visualization(cloud, should_show_axis=True, axis_size=10)
        print()

    def non_blocking_pcd_visualization(
        self,
        pcd: o3d.geometry.PointCloud,
        should_center=False,
        should_show_axis=False,
        axis_size: float = 1,
    ):
        """
            Real time point cloud visualization.
            Args:
                pcd: point cloud to be visualized
                should_center: true to always center the point cloud
                should_show_axis: true to show axis
                axis_size: adjust axis size
            Returns:
                None
            """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if should_center:
            points = points - np.mean(points, axis=0)

        if self.points_added is False:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            if should_show_axis:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=axis_size, origin=np.mean(points, axis=0)
                )
                self.vis.add_geometry(self.coordinate_frame)
            self.vis.add_geometry(self.pcd)
            self.points_added = True
        else:
            # print(np.shape(np.vstack((np.asarray(self.pcd.points), points))))
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            if should_show_axis:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=axis_size, origin=np.mean(points, axis=0)
                )
                self.vis.update_geometry(self.coordinate_frame)
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()


def main(args=None):
    rclpy.init()
    node = TestNode()
    rclpy.spin(node)


## The code below is "ported" from
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
# I'll make an official port and PR to this repo later:
# https://github.com/ros2/common_interfaces
import sys
from collections import namedtuple
import ctypes
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.
    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), "cloud is not a sensor_msgs.msg.PointCloud2"
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = ">" if is_bigendian else "<"

    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print(
                "Skipping unknown PointField datatype [%d]" % field.datatype,
                file=sys.stderr,
            )
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


if __name__ == "__main__":
    main()
