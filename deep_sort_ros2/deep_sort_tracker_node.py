#!/usr/bin/env python3
from distutils.log import debug
from tkinter.messagebox import NO
import rclpy.node
import rclpy
from unicodedata import name
from deep_sort_ros2.tools import generate_detections as gdet
from deep_sort_ros2.deep_sort.tracker import Tracker
from deep_sort_ros2.deep_sort.detection import Detection
from deep_sort_ros2.deep_sort import preprocessing, nn_matching
from PIL import Image
from deep_sort_ros2.core.config import cfg
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from deep_sort_ros2.core.yolov4 import filter_boxes
import deep_sort_ros2.core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
import tensorflow as tf
import time
import os
from pathlib import Path
import sys
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ROSImage
import matplotlib.pyplot as plt
from typing import List
from geometry_msgs.msg import Pose2D
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    BoundingBox2D,
    ObjectHypothesisWithPose,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DeepSORTNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("DeepSORT_Node")
        # ROS Node variables
        self.declare_parameter("model_data_path", "")
        self.declare_parameter("image_topic", "test_image")
        self.declare_parameter("debug", False)

        self.debug = self.get_parameter("debug").get_parameter_value().bool_value

        self.subscription = self.create_subscription(
            ROSImage,
            self.get_parameter("image_topic").get_parameter_value().string_value,
            self.on_image_received,
            10,
        )
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.detection_publisher = self.create_publisher(
            Detection2DArray, "deep_sort_output", 10
        )

        # prepare DeepSORT variables
        self.max_cosine_distance = 0.4
        self.nn_budget = None
        self.nms_max_overlap = 1.0

        # initialize deep sort
        self.model_filename = (
            self.get_parameter("model_data_path").get_parameter_value().string_value
        )

        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget
        )
        # initialize tracker
        self.tracker = Tracker(metric)

        # load configuration for object detector
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS, is_tiny=False)

        paths = [
            Path(p) / "deep_sort_ros2/checkpoints/yolov4-416"
            for p in sys.path
            if "deep_sort" in p
        ]
        correct_path = None
        for p in paths:
            if p.exists():
                correct_path = p.as_posix()
        assert correct_path is not None, f"Did not find {paths}"
        self.saved_model_loaded = tf.saved_model.load(
            correct_path, tags=[tag_constants.SERVING]
        )
        self.infer = self.saved_model_loaded.signatures["serving_default"]
        self.frame_num = 0
        self.input_size = 416

    def execute_detection(self, frame: np.ndarray) -> List[Detection]:
        # execute detection
        image = Image.fromarray(frame)
        self.frame_num += 1
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        (
            boxes,
            scores,
            classes,
            valid_detections,
        ) = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.5,
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0 : int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0 : int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0 : int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # custom allowed classes (uncomment line below to customize tracker for only car)
        allowed_classes = ["car"]

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        cv2.putText(
            frame,
            "Objects being tracked: {}".format(count),
            (5, 35),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            2,
            (0, 255, 0),
            2,
        )

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = self.encoder(frame, bboxes)
        detections = [
            Detection(bbox, score, class_name, feature)
            for bbox, score, class_name, feature in zip(bboxes, scores, names, features)
        ]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, self.nms_max_overlap, scores
        )
        detections = [detections[i] for i in indices]
        return detections

    def draw_bbox(self, frame, track, colors):
        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(
            frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2,
        )
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1] - 30)),
            (
                int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                int(bbox[1]),
            ),
            color,
            -1,
        )
        cv2.putText(
            frame,
            class_name + "-" + str(track.track_id),
            (int(bbox[0]), int(bbox[1] - 10)),
            0,
            0.75,
            (255, 255, 255),
            2,
        )

    def on_image_received(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        raw_frame = frame.copy()
        start_time = time.time()

        detections = self.execute_detection(frame=frame)

        # initialize color map
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # draw bbox on screen
            self.draw_bbox(frame, track, colors)

        # send it out via ROS msg
        detections = self.tracker_to_detection2d(self.tracker)
        self.detection_publisher.publish(detections)

        if self.debug:
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.imshow("Output Video", cv2.resize(frame, dsize=(800, 600)))
            cv2.waitKey(1)

    @staticmethod
    def tracker_to_detection2d(tracker: Tracker) -> Detection2DArray:
        detections = Detection2DArray()
        for track in tracker.tracks:
            bbox = track.to_tlbr()
            (min_x, min_y), (max_x, max_y) = (
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
            )
            center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
            bbox2d: BoundingBox2D = BoundingBox2D(
                center=Pose2D(x=center[0], y=center[1])
            )

            detection2D = Detection2D()
            detection2D.bbox = bbox2d
            detection2D.is_tracking = True
            detections.detections.append(detection2D)
        return detections


def main(args=None):
    rclpy.init()
    node = DeepSORTNode()
    rclpy.spin(node)


if __name__ == "__main__":
    print("Hello world")
