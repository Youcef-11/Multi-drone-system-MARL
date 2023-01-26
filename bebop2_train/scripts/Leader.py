#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose


class LeaderBebop:
    def __init__(self, namespace):
        rospy.init_node("Leader_bebop")
        self.odom_sub = rospy.Subscriber(f"/{namespace}/ground_truth/odometry", Odometry, self.odom_cb)
        self.cmd_pub = rospy.Publisher(f"/{namespace}/cmd_vel", Twist, queue_size=1)
        self.stop_until = 0

    

    def odom_cb(self, msg : Odometry):
        self.odom = msg
        self.pose = self.odom.pose.pose


    def hasardous_move(self, min_altitude = 0.3, stop_chance = 5):
        """Avec stop_chance %  de chance, le leader s'arrête pendant 3s
        Sinon, il choisis une action au hasard en faisant attention a ne pas passer en dessous de l'altitude 'min_altitude'
        """



        cmd = Twist()
        if rospy.get_rostime().to_sec() < self.stop_until:
            self.cmd_pub.publish(cmd)
            return

        if np.random.uniform(0,1,1) < stop_chance/100:
            self.stop_until = rospy.get_rostime().to_sec() + 3 
            self.cmd_pub.publish(cmd)

        else: 
            action = np.random.uniform(-1,1,4)
            cmd.linear.x = action[0]
            cmd.linear.y = action[1]

            if self.pose.position.z < min_altitude and action[2] < 0:
                cmd.linear.z = -action[2]

            cmd.angular.z = action[3]
            self.cmd_pub.publish(cmd)



