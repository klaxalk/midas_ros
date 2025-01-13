#!/bin/sh
"exec" "`dirname $0`/../python-env/bin/python3" "$0" "$@"

import rospy
import rosnode
import time

import tensorflow as tf
import sys
import os
import time
import datetime

from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import CameraInfo

import cv2
from cv_bridge import CvBridge

import numpy as np

print("before pytorch")

import torch

print("after pytorch")

class Midas_ROS:

    def __init__(self):

        rospy.init_node("midas", anonymous=True)

        ## | ----------------------- load params ---------------------- |

        model_type = rospy.get_param("~model_type")
        device = rospy.get_param("~device")

        ## | ---------------------- params loaded --------------------- |

        rospy.loginfo('ros node initialized')

        rospy.loginfo('opening cv bridge')

        self.bridge = CvBridge()

        rospy.loginfo('cv bridge opened')

        rospy.loginfo('loading midas mode {}'.format(model_type))

        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

        rospy.loginfo('initializing device')

        self.device = torch.device("cpu")

        self.midas.to(self.device)
        self.midas.eval()

        rospy.loginfo('loading transforms')

        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        rospy.loginfo('midas loaded')

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

        rospy.Subscriber("/uav85/camera_front_throttled/image_raw/compressed", CompressedImage, self.callback)
        rospy.Subscriber("/uav85/camera_front_throttled/camera_info", CameraInfo, self.callbackCameraInfo)

        self.publisher_image = rospy.Publisher("/uav85/camera_front_depth/image_raw", Image, queue_size=1)
        self.publisher_cam_info = rospy.Publisher("/uav85/camera_front_depth/camera_info", CameraInfo, queue_size=1)

        self.is_initialized = True

        rospy.spin()

    def callback(self, msg):

        if not self.is_initialized:
            return

        rospy.loginfo_once('getting images')

        self.got_image = True

        with torch.no_grad():

            # for raw image
            # original_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # for compressed image
            np_arr = np.fromstring(msg.data, np.uint8)
            original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            input_batch = self.transform(original_image).to(self.device)

            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=original_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            output = prediction.cpu().numpy()

            image_message = self.bridge.cv2_to_imgmsg(output, encoding="32FC1")
            image_message.header = msg.header

            self.publisher_image.publish(image_message)

            rospy.loginfo('publishing depth')

    def callbackCameraInfo(self, msg):

        if not self.is_initialized:
            return

        rospy.loginfo_once('getting camera info')

        self.publisher_cam_info.publish(msg)

if __name__ == '__main__':
    try:
        pydnet_ros = Midas_ROS()
    except rospy.ROSInterruptException:
        pass
