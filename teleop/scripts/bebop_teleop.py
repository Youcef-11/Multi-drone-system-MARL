#!/usr/bin/env python
import rospy
from pynput.keyboard import Key, Listener
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
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

        self.namespace = namespace
        self.keys = keybindings

        self.cmd_pub = rospy.Publisher(f"/{self.namespace}/cmd_vel", Twist, queue_size=1)
        self.takeoff_pub = rospy.Publisher(f"/{self.namespace}/takeoff", Empty, queue_size=1)
        self.land_pub = rospy.Publisher(f"/{self.namespace}/land", Empty, queue_size=1)

        self.twist = Twist()

        self.display_help()


    def action(self, key):
        self.do_action(key)



    def do_action(self, key):
        if self.test_action(key, "up"):
            self.twist.linear.z += 0.05
            
        if self.test_action(key, "down"):
            self.twist.linear.z -= 0.05

        if self.test_action(key, "forward"):
            self.twist.linear.x += 0.05

        if self.test_action(key, "backward"):
            self.twist.linear.x -= 0.05

        if self.test_action(key, "left"):
            self.twist.linear.y += 0.05

        if self.test_action(key, "right"):
            self.twist.linear.y -= 0.05

        if self.test_action(key, "takeoff"):
            self.takeoff_pub.publish(Empty()) 
        
        if self.test_action(key, "rotate_right"):
            self.twist.angular.z -= 0.5

        if self.test_action(key, "rotate_left"):
            self.twist.angular.z += 0.5

        if self.test_action(key, "land"):
            self.land_pub.publish(Empty()) 

        if self.test_action(key, "stop"):
            self.twist = Twist()



        self.cmd_pub.publish(self.twist)


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
    "takeoff" : '+',
    "land" : Key.enter,
    "stop" : '0'
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
    "takeoff" : Key.space,
    "land" : 'f',
    "stop" : 'x'
}

Left = teleop("L_bebop2", L_keybinds)
Right = teleop("R_bebop2", R_keybinds)

rospy.spin()
