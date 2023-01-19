#!/usr/bin/env python
import gym
import numpy
import time
# import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.bebop2 import double_bebop2_task


if __name__ == '__main__':

    rospy.init_node('double_bebop_train', log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('DoubleBebop2Env-v0')
    rospy.logwarn("GYM ENVIRONMENT DONE")

    env.reset()



    env.close()
