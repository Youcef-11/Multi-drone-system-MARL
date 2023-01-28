#!/usr/bin/env python
import numpy as np 
import rospy 
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Empty



class test:
    def __init__(self, namespace):
        rospy.init_node("test")
        rospy.Subscriber(f"/{namespace}/ground_truth/odometry", Odometry, self.cb_odom)

        self.gaz_pub = rospy.Publisher("/gazebo/set_model_state", ModelState) 
        self.reset_pub = rospy.Publisher(f"/{namespace}/fake_driver/reset_pose", Empty)


        r = rospy.Rate(10)
        while True:
            odom_pose = self.odom.pose.pose
            A = ModelState()
            A.pose.position  = odom_pose.position
            A.pose.position.z = 2
            A.pose.orientation = odom_pose.orientation


            input("Enter to publish")
            self.gaz_pub.Publsih(A)
            
            self.reset_pub.publish(Empty())
            
            r.sleep()

    
    def cb_odom(self, msg : Odometry):
        self.odom = msg


