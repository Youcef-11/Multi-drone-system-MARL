import sys
import rospy
import cv2 as cv
import numpy as np
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from bebop_msgs.msg import CommonCommonStateBatteryStateChanged
import argparse

parser = argparse.ArgumentParser(description="checkimage")
parser.add_argument("--ns", type = str, default = "bebop", help = "namespace of the parrot bebop")
params = parser.parse_args()
class Image():

    def __init__(self):
        self.namespace = params.ns
        self.node = rospy.init_node(f"Image_{self.namespace}")
        self.bridge = CvBridge()
        self.sub_im = rospy.Subscriber(f"/{self.namespace}/image_raw/compressed", CompressedImage, self.cb_im, queue_size= 1)
        self.sub_battery = rospy.Subscriber(f"/{self.namespace}/states/common/CommonState/BatteryStateChanged", CommonCommonStateBatteryStateChanged, self.cb_battery)
        self.battery = -1
        # self.sub_odom  =  rospy.Subscriber("/bebop/odom", Odometry , self.cb_odom, queue_size= 1)
    

    def cb_battery(self, msg):
        self.battery = msg.percent


    def cb_im(self, img):
        self.image_cam = self.bridge.compressed_imgmsg_to_cv2(img,"bgr8")
        cv.putText(self.image_cam, f"battery : {self.battery}%", (0,30),cv.FONT_HERSHEY_PLAIN,  fontScale = 2, color = (0,0,0), thickness = 1)

        cv.imshow("Parrot img", self.image_cam)
        cv.waitKey(1)
    
    def cb_odom(self, msg):
        print(msg)




I = Image()
rospy.spin()