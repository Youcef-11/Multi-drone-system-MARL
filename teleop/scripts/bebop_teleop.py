#!/usr/bin/env python
import rospy
from pynput.keyboard import Key, Listener
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import numpy as np 
import time


def compare_twist(t1, t2):
    if t1.angular.x == t2.angular.x and t1.angular.y == t2.angular.y and t1.angular.z == t2.angular.z and t1.linear.z == t2.linear.z:
        return True
    return False

class teleop:
    def __init__(self,namespace, keybindings):
        """initialize the teleop object with the keybindings

        Args:
            keybindings (dict): keybindings of the teleoperation (up, down, forward, backward, left, right, rotate_right, rotate_left, takeoff, land)
        """

        # Start the Listener
        rospy.init_node("bebop_teleop")
        self.lis = Listener(on_press= self.action)
        self.lis.start()
        self.timeout = 0.2

        self.namespace = namespace
        self.keys = keybindings

        self.cmd_pub = rospy.Publisher(f"/{self.namespace}/cmd_vel", Twist, queue_size=1)
        self.takeoff_pub = rospy.Publisher(f"/{self.namespace}/takeoff", Empty, queue_size=1)
        self.land_pub = rospy.Publisher(f"/{self.namespace}/land", Empty, queue_size=1)

        self.twist = Twist()
        self.accelerate_mode = False
        self.speed_tab = [0.2, 0.5, 0.8]

        # speed choice (indice in the speed_tab)
        self.sc = 0
        self.last_pub = 0


        self.display_help()


    def action(self, key):

        if rospy.get_rostime().to_sec() > self.last_pub + self.timeout:
            # On reset le twist dans ce cas la 
            self.twist = Twist()

        if self.test_action(key, "mode"):
            self.accelerate_mode = not self.accelerate_mode
            print(f"{self.namespace} : mode accelerate : {self.accelerate_mode}")

        if self.test_action(key, "change_speed"):
            self.sc = (self.sc + 1) %3 
            print(f"{self.namespace} : Speed changed to {self.speed_tab[self.sc]}")

        if self.test_action(key, "takeoff/land"):
            if self.has_takeoff:
                self.land_pub.publish(Empty()) 
            else:
                self.takeoff_pub.publish(Empty()) 

        if not self.accelerate_mode:
            self.do_action(key)
        else: 
            self.do_action_accelerate(key)


        self.twist = self.clip_twist(self.twist)
        self.cmd_pub.publish(self.twist)
        self.last_pub = rospy.get_rostime().to_sec()

        



    def clip_twist(self,twist):
        twist.linear.x = np.clip(twist.linear.x, -1, 1)
        twist.linear.y = np.clip(twist.linear.y, -1, 1)
        twist.linear.z = np.clip(twist.linear.z, -1, 1)
        twist.angular.z = np.clip(twist.angular.z, -1, 1)
        return twist

    def do_action(self, key):
        if self.test_action(key, "up"):
            self.twist.linear.z = self.speed_tab[self.sc]
            
        if self.test_action(key, "down"):
            self.twist.linear.z = -self.speed_tab[self.sc]

        if self.test_action(key, "forward"):
            self.twist.linear.x = self.speed_tab[self.sc]

        if self.test_action(key, "backward"):
            self.twist.linear.x = -self.speed_tab[self.sc]

        if self.test_action(key, "left"):
            self.twist.linear.y = self.speed_tab[self.sc]

        if self.test_action(key, "right"):
            self.twist.linear.y = -self.speed_tab[self.sc]

        if self.test_action(key, "rotate_right"):
            self.twist.angular.z = -self.speed_tab[self.sc] * 2

        if self.test_action(key, "rotate_left"):
            self.twist.angular.z = self.speed_tab[self.sc]* 2

        if self.test_action(key, "stop"):
            self.twist = Twist()


    def do_action_accelerate(self, key):
        speed_increment = 0.05
        if self.test_action(key, "up"):
            self.twist.linear.z += speed_increment
            
        if self.test_action(key, "down"):
            self.twist.linear.z -= speed_increment

        if self.test_action(key, "forward"):
            self.twist.linear.x += speed_increment

        if self.test_action(key, "backward"):
            self.twist.linear.x -= speed_increment

        if self.test_action(key, "left"):
            self.twist.linear.y += speed_increment

        if self.test_action(key, "right"):
            self.twist.linear.y -= speed_increment


        
        if self.test_action(key, "rotate_right"):
            self.twist.angular.z -= 0.3

        if self.test_action(key, "rotate_left"):
            self.twist.angular.z += 0.3

        if self.test_action(key, "stop"):
            self.twist = Twist()

        




    def test_action(self, key, action):
        return key == self.keys[action] or str(key)[1:-1] == self.keys[action]

    def display_help(self):
        print(f"touches for {self.namespace} : ") 
        print("--------------------------------")
        for action in self.keys:
            print(f"{action} : {self.keys[action]}")
        print("---------------------------------")



L_keybinds = {
    "up" : '7',
    "down" : '9',
    "forward" : '8',
    "backward" : '2',
    "left" : '4',
    "right" : '6',
    "rotate_right" :'3',
    "rotate_left" : '1',
    "takeoff/land" : Key.enter,
    "stop" : '0',
    "mode" : Key.f1,
    "change_speed" : Key.ctrl_r

}

R_keybinds = {
    "up" : 'a',
    "down" : 'e',
    "forward" : 'z',
    "backward" : 's',
    "left" : 'q',
    "right" : 'd',
    "rotate_right" :'c',
    "rotate_left" : 'w',
    "takeoff/land" : Key.space,
    "stop" : 'x',
    "mode" : Key.f2,
    "change_speed" : Key.ctrl_l
}

Left = teleop("L_bebop2", R_keybinds)
#Right = teleop("R_bebop2", R_keybinds)

rospy.spin()
