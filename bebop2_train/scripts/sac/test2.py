#!/usr/bin/env python
import rospy
import numpy as np 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose

class PID:
    def __init__(self):
        
def compute_cmd(dist):
    if abs(dist) > 3:
        cmd = np.sign(dist) * 1
    elif abs(dist) > 2:
        cmd = np.sign(dist) * 0.7
    elif abs(dist) > 1:
        cmd = np.sign(dist) * 0.4
    elif abs(dist) > 0.4:
        cmd = np.sign(dist) * 0.2
    elif abs(dist) >0.1:
        cmd = np.sign(dist) * 0.1
    elif abs(dist) > 0.05:
        cmd = np.sign(dist) * 0.05
    else :
        cmd = 0

        
    return cmd

class follower: 
    
    def __init__(self, master, follower, offset = 0):
        self.master = master
        self.follower=  follower

        rospy.init_node("master_follower")

        self.m_odom_sub = rospy.Subscriber(f"/{self.master}/ground_truth/odom", Odometry, self.cb_odom_m)
        self.f_odom_sub = rospy.Subscriber(f"/{self.follower}/ground_truth/odom", Odometry, self.cb_odom_f)

        self.f_cmd_pub = rospy.Publisher(f"/{self.follower}/cmd_vel", Twist, queue_size = 1)
        
        rate= rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                # m_pose = self.m_odom.pose.pose.position
                # f_pose = self.f_odom.pose.pose.position
                m_pose = rospy.wait_for_message(f"/{self.master}/ground_truth/odometry", Odometry).pose.pose.position
                f_pose = rospy.wait_for_message(f"/{self.follower}/ground_truth/odometry", Odometry).pose.pose.position
                #f_pose.y = f_pose.y + 1

                dist_x = (m_pose.x - f_pose.x)
                dist_y =  (1 + m_pose.y - f_pose.y)
                dist_z = (m_pose.z - f_pose.z)

                cmd = Twist()
                
                cmd_x = compute_cmd(dist_x)
                cmd_y = compute_cmd(dist_y)
                cmd_z = compute_cmd(dist_z)

                cmd.linear.x = cmd_x
                cmd.linear.y = cmd_y
                cmd.linear.z = cmd_z

                self.f_cmd_pub.publish(cmd)

                rate.sleep()
            except (KeyboardInterrupt, rospy.ROSInterruptException):
                print("sortie")
        
        


    def cb_odom_m(self, msg: Odometry):
        self.m_odom = msg

    def cb_odom_f(self, msg):
        self.f_odom = msg



if __name__ == "__main__":
    f = follower("L_bebop2", "R_bebop2", 0)


