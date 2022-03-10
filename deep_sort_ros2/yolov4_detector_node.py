#!/usr/bin/env python3
from matplotlib.pyplot import box, xscale, yscale
from sqlalchemy import false
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from geometry_msgs.msg import Pose2D
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    BoundingBox2D,
    ObjectHypothesisWithPose,
)

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class YoloV4DetecotrNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("YoloV4_vehicle_detector_node")
        self.declare_parameter("yolo_v4_model_path", "")
        self.declare_parameter("image_topic", "test_image")
        self.declare_parameter("debug", False)
        self.declare_parameter("height", 896)
        self.declare_parameter("width", 1536)
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.HEIGHT = self.get_parameter("height").get_parameter_value().integer_value
        self.WIDTH = self.get_parameter("width").get_parameter_value().integer_value
        self.model = YOLOv4(
            input_shape=(self.HEIGHT, self.WIDTH, 3),
            anchors=YOLOV4_ANCHORS,
            num_classes=80,
            training=False,
            yolo_max_boxes=20,
            yolo_iou_threshold=0.5,
            yolo_score_threshold=0.73,
        )

        self.yolo_v4_weight_path = Path(
            self.get_parameter("yolo_v4_model_path").get_parameter_value().string_value
        )
        self.get_logger().info(f"{self.yolo_v4_weight_path}")
        self.model.load_weights(self.yolo_v4_weight_path.as_posix())
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            self.get_parameter("image_topic").get_parameter_value().string_value,
            self.on_image_received,
            10,
        )
        self.subscription  # prevent unused variable warning

        self.detection_publisher = self.create_publisher(
            Detection2DArray, "yolov4_output", 10
        )

    def find_boxes_in_orig_image(self, orig_img, predicted_img, boxes):

        x_ = predicted_img.shape[0]
        y_ = predicted_img.shape[1]

        x_scale = orig_img.shape[0] / x_
        y_scale = orig_img.shape[1] / y_

        new_boxes = []
        for box in boxes:
            # note that x and y are flipped in numpy vs in cv2
            (x_pred, y_pred, max_x_pred, max_y_pred) = box
            x = int(x_pred * y_scale)
            y = int(y_pred * x_scale)
            xmax = int(max_x_pred * y_scale)
            ymax = int(max_y_pred * x_scale)
            new_boxes.append(((x, y), (xmax, ymax)))

        return new_boxes

    def on_image_received(self, msg):
        rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        start = time.time()
        try:
            result_img, boxes, scores = self.proccess_frame(
                tf.convert_to_tensor(rgb_img), self.model
            )

            # convert the bounding boxes back to original image's size
            new_boxes = self.find_boxes_in_orig_image(
                orig_img=rgb_img, predicted_img=result_img, boxes=boxes
            )
            detections = Detection2DArray()
            # convert them to vision_msg format
            for box, score in zip(new_boxes, scores):
                (min_x, min_y), (max_x, max_y) = box
                center = ((max_x + min_x) / 2, (max_y + min_y) / 2)

                bbox2d: BoundingBox2D = BoundingBox2D(
                    center=Pose2D(x=center[0], y=center[1])
                )

                detection2D = Detection2D()
                detection2D.bbox = bbox2d
                # detection2D.source_img = rgb_img[min_x:max_x, min_y:max_y]
                detection2D.is_tracking = False
                detection2D.results.append(ObjectHypothesisWithPose(score=float(score)))

                detections.detections.append(detection2D)
            self.detection_publisher.publish(detections)

            if self.debug:
                visualization_img = rgb_img.copy()
                for box in new_boxes:
                    cv2.rectangle(
                        visualization_img, box[0], box[1], (0, 255, 0), thickness=2
                    )
                cv2.putText(
                    visualization_img,
                    f"FPS: {1 / (time.time()-start)}",
                    (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                cv2.imshow(
                    "Detection Result", cv2.resize(visualization_img, (800, 600))
                )
                cv2.imshow("Raw", cv2.resize(rgb_img, (800, 600)))
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"{e}")

    def detected_photo(self, boxes, scores, classes, detections, image):
        boxes = (boxes[0] * [self.WIDTH, self.HEIGHT, self.WIDTH, self.HEIGHT]).astype(
            int
        )
        scores = scores[0]
        classes = classes[0].astype(int)
        detections = detections[0]

        CLASSES = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        ########################################################################

        image_cv = image.numpy()
        result_boxes = []
        result_scores = []
        for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):

            if score > 0:
                if class_idx == 2:  # show bounding box only to the "car" class

                    #### Draw a rectangle ##################
                    # convert from tf.Tensor to numpy

                    box = (int(xmin), int(ymin), int(xmax), int(ymax))

                    cv2.rectangle(
                        image_cv,
                        (int(xmin), int(ymin)),
                        (int(xmax), int(ymax)),
                        (0, 255, 0),
                        thickness=2,
                    )

                    # Add detection text to the prediction
                    text = CLASSES[class_idx] + ": {0:.2f}".format(score)
                    cv2.putText(
                        image_cv,
                        text,
                        (int(xmin), int(ymin) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    result_boxes.append(box)
                    result_scores.append(score)
        # image_cv = cv2.normalize(
        #     image_cv, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return image_cv, result_boxes, result_scores

    def proccess_frame(self, photo, model):
        images = self.resize_image(photo)
        boxes, scores, classes, detections = model.predict(images)
        result_img = self.detected_photo(boxes, scores, classes, detections, images[0])
        return result_img

    def resize_image(self, image):
        # Resize the output_image:
        image = tf.image.resize(image, (self.HEIGHT, self.WIDTH))
        # Add a batch dim:
        images = tf.expand_dims(image, axis=0) / 255
        return images


def main(args=None):
    rclpy.init()
    node = YoloV4DetecotrNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
