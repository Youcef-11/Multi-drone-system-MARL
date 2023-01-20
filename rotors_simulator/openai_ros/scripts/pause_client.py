#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty

def main():
    rospy.init_node("gazebo_pause_physics")
    rospy.wait_for_service("/gazebo/pause_physics")
    client = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    try:
        client()
        rospy.loginfo("Physics paused")
    except (Exception, rospy.ROSInterruptException) as e :
        rospy.logerr(f"service call failed {e}")

if __name__ == '__main__' :
    main()

