#!/usr/bin/env python
# Import necessary libraries
import rospy
from std_msgs.msg import Empty


if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('take_off_drones')
    # Send a takeoff command to the drones
    takeoff_pub_L = rospy.Publisher('/L_bebop2/takeoff', Empty, queue_size=1)
    takeoff_pub_R = rospy.Publisher('/R_bebop2/takeoff', Empty, queue_size=1)

    takeoff_pub_L.publish(Empty())
    takeoff_pub_L.publish(Empty())
    takeoff_pub_L.publish(Empty())
    takeoff_pub_L.publish(Empty())
    takeoff_pub_R.publish(Empty())
    takeoff_pub_R.publish(Empty())
    takeoff_pub_R.publish(Empty())
    takeoff_pub_R.publish(Empty())
