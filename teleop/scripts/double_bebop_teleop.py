import rospy
from pynput.keyboard import Key, Listener
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
import numpy as np
import time
import signal


# def compare_twist(t1, t2):
#     if t1.angular.x == t2.angular.x and t1.angular.y == t2.angular.y and t1.angular.z == t2.angular.z and t1.linear.z == t2.linear.z:
#         return True
#     return False

class teleop:
    def __init__(self, keybindings):
        """initialize the teleop object with the keybindings

        Args:
            keybindings (dict): keybindings of the teleoperation (up, down, forward, backward, left, right, rotate_right, rotate_left, takeoff, land)
        """

        # Start the Listener
        self.lis = Listener(on_press= self.action)
        self.lis.start()

        self.namespace1 = "L_bebop2"
        self.namespace2 = "R_bebop2"
        self.cmd_pub = []
        self.obs = []

        self.keys = keybindings

        self.L_cmd_pub = rospy.Publisher(f"/{self.namespace1}/cmd_vel", Twist, queue_size=1)
        self.L_takeoff_pub = rospy.Publisher(f"/{self.namespace1}/takeoff", Empty, queue_size=1)
        self.L_land_pub = rospy.Publisher(f"/{self.namespace1}/land", Empty, queue_size=1)


        self.R_cmd_pub = rospy.Publisher(f"/{self.namespace2}/cmd_vel", Twist, queue_size=1)
        self.R_takeoff_pub = rospy.Publisher(f"/{self.namespace2}/takeoff", Empty, queue_size=1)
        self.R_land_pub = rospy.Publisher(f"/{self.namespace2}/land", Empty, queue_size=1)

        self.twist = Twist()

        self.display_help()
        self.action_save = []
        self.L_obs = []
        self.R_obs = []
        self.terminate = False
        signal.signal(signal.SIGINT, self.signal_handle)

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.action_save.append(self.twist)
            obs_L = rospy.wait_for_message("/L_bebop2/odom",Odometry)
            obs_R = rospy.wait_for_message("/R_bebop2/odome", Odometry)
            self.L_obs.append(obs_L)
            self.R_obs.append(obs_R)
            if self.terminate:
                break
            rate.sleep()
        
        dic = {"action" : self.action_save,"L_obs" : self.L_obs, "R_obs" : self.R_obs}

        np.save("../data/data_real.npy", dic, allow_pickle=True)

    def action(self, key):
        self.do_action(key)

    def signal_handle(self, sig, frame):
        self.terminate = True

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
            self.R_takeoff_pub.publish(Empty()) 
            self.L_takeoff_pub.publish(Empty()) 
        
        if self.test_action(key, "rotate_right"):
            self.twist.angular.z -= 0.5

        if self.test_action(key, "rotate_left"):
            self.twist.angular.z += 0.5

        if self.test_action(key, "land"):
            self.L_land_pub.publish(Empty()) 
            self.R_land_pub.publish(Empty()) 

        if self.test_action(key, "stop"):
            self.twist = Twist()
        
        self.twist.linear.x = np.clip(self.twist.linear.x, -1, 1)
        self.twist.linear.y = np.clip(self.twist.linear.y, -1, 1)
        self.twist.linear.z = np.clip(self.twist.linear.z, -1, 1)
        self.twist.angular.z = np.clip(self.twist.angular.z, -1, 1)

        self.L_cmd_pub.publish(self.twist)
        self.R_cmd_pub.publish(self.twist)



    def test_action(self, key, action):
        return key == self.keys[action] or str(key)[1:-1] == self.keys[action]

    def display_help(self):
        print(f"touches : ") 
        print("--------------------------------")
        for action in self.keys:
            print(f"{action} : {self.keys[action]}")
        print("---------------------------------")



keybinds = {
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


if __name__ == "__main__":
    rospy.init_node("double_bebop_teleop", anonymous=True)
    tele = teleop(keybinds)

