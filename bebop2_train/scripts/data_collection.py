#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
import numpy as np
import signal
import random


class Datacollection:
    def __init__(self):

        self.namespace1 = "L_bebop2"
        self.namespace2 = "R_bebop2"

        self.stop_until = 0

        self.L_cmd_pub = rospy.Publisher(f"/{self.namespace1}/cmd_vel", Twist, queue_size=1)

        self.R_cmd_pub = rospy.Publisher(f"/{self.namespace2}/cmd_vel", Twist, queue_size=1)

        self.stop_chance = 0.05

        self.min_altitude = 0.2

        self.random_action = [1,2,3,4]

        self.action_save = []
        self.L_obs = []
        self.R_obs = []
        self.terminate = False
        signal.signal(signal.SIGINT, self.signal_handle)

        self.rate = rospy.Rate(10)

    def signal_handle(self, sig, frame):
        self.terminate = True

    def take_hasardous_action(self, min_altitude = 0.3, stop_chance = 0.05):
        """Avec stop_chance %  de chance, le leader s'arrête pendant 3s
        Sinon, il choisis une action au hasard en faisant attention a ne pas passer en dessous de l'altitude 'min_altitude'
        """

        twist = Twist()

        if np.random.uniform(0,1,1) < stop_chance/100:
            self.stop_until = rospy.get_rostime().to_sec() + 3 
            self.L_cmd_pub.publish(twist)
            self.R_cmd_pub.publish(twist)
            rospy.logwarn("3 seconds pause !")

        else: 
            action = np.random.uniform(-1,1,4)
            choix = random.choice(self.random_action)
            if choix == 1:
                twist.linear.x = action[0]
            elif choix == 2:     
                twist.linear.y = action[1]

            elif choix == 3:
                L_altitude = rospy.wait_for_message("/L_bebop2/ground_truth/odometry",Odometry)
                R_altitude = rospy.wait_for_message("/R_bebop2/ground_truth/odometry",Odometry)
                if  (L_altitude.pose.pose.position.z < min_altitude or R_altitude.pose.pose.position.z < min_altitude ) and action[2] < 0:
                    twist.linear.z = 1
                    rospy.logwarn("Near to ground!")
                else:
                    twist.linear.z = action[2]
            # elif choix == 4:
            #     twist.angular.z = action[3]

            self.L_cmd_pub.publish(twist)
            self.R_cmd_pub.publish(twist)
        
        return twist

    def record(self):
        print("Collection data begin...")

        while True:
            twist = self.take_hasardous_action()
            obs_L = rospy.wait_for_message("/L_bebop2/ground_truth/odometry",Odometry)
            obs_R = rospy.wait_for_message("/R_bebop2/ground_truth/odometry", Odometry)
            self.action_save.append(twist)
            self.L_obs.append(obs_L.pose.pose)
            self.R_obs.append(obs_R.pose.pose)
            if self.terminate:
                break
            self.rate.sleep()
        
        print("Saving...")
        dic = {"action" : self.action_save,"L_obs" : self.L_obs, "R_obs" : self.R_obs}

        np.save("data_simu.npy", dic, allow_pickle=True)



if __name__ == "__main__":
    rospy.init_node("double_bebop_teleop", anonymous=True)
    data = Datacollection()
    data.record()

