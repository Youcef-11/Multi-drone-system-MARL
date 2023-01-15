#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Twist, Transform

def send_waypoints():
    # Initialize the ROS node and publisher
    rospy.init_node("waypoint_sender")
    pub = rospy.Publisher("/L_bebop2/command/trajectory", MultiDOFJointTrajectory, queue_size=10)

    # Define the waypoints
    waypoints = [Transform(translation=Twist(linear=[1, 0, 1]), rotation=[0, 0, 0, 1]),
                 Transform(translation=Twist(linear=[2, 0, 1]), rotation=[0, 0, 0, 1]),
                 Transform(translation=Twist(linear=[3, 0, 1]), rotation=[0, 0, 0, 1])]

    # Create the MultiDOFJointTrajectory message and set the header, joint_names, and points
    trajectory = MultiDOFJointTrajectory()
    trajectory.header.stamp = rospy.Time.now()
    trajectory.joint_names = []
    for waypoint in waypoints:
        point = MultiDOFJointTrajectoryPoint(transforms=[waypoint])
        trajectory.points.append(point)

    # Publish the trajectory message
    pub.publish(trajectory)

if __name__ == "__main__":
    try:
        send_waypoints()
    except rospy.ROSInterruptException:
        pass

