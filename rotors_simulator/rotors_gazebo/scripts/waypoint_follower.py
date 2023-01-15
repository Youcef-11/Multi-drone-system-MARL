#!/usr/bin/env python
# Import necessary libraries
import rospy
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Empty

# Initialize the ROS node
rospy.init_node('waypoint_follower')

# Define the set of waypoints as a PoseArray object
waypoints = PoseArray()
waypoints.header.frame_id = "world"

# Add the desired waypoints to the PoseArray object
waypoints.poses.append(Pose(x=0, y=0, z=2))
waypoints.poses.append(Pose(x=2, y=2, z=2))
waypoints.poses.append(Pose(x=4, y=0, z=2))

# Publish the waypoints to the drone
waypoint_pub = rospy.Publisher('/bebop/waypoints', PoseArray, queue_size=10)
waypoint_pub.publish(waypoints)

# Send a takeoff command to the drone
takeoff_pub = rospy.Publisher('/bebop/takeoff', Empty, queue_size=10)
takeoff_pub.publish(Empty())

# Wait for the drone to reach the first waypoint
rospy.sleep(5)

# Send a command to the drone to start following the waypoints
follow_pub = rospy.Publisher('/bebop/follow_waypoints', Empty, queue_size=10)
follow_pub.publish(Empty())

# Keep the script running until the node is shut down
rospy.spin()

