#!/usr/bin/env python


import rospy
rospy.init_node("test")
print(rospy.get_rostime().to_sec())
rospy.sleep(2)

print(rospy.get_rostime().to_sec())
# print(rospy.get_time().to_sec())