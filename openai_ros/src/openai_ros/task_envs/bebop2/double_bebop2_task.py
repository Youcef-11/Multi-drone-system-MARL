from gym import spaces
from openai_ros.robot_envs import double_bebop2_env
from openai_ros.openai_ros_common import ROSLauncher
from gym.envs.registration import register
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import rospy
from geometry_msgs.msg import Vector3, Pose
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import time
import numpy as np
from pathlib import Path
import math

# The path is __init__.py of openai_ros, where we import the MovingCubeOneDiskWalkEnv directly
MAX_STEP = 1000 # Can be any Value

register(
        id='DoubleBebop2Env-v0',
        entry_point='openai_ros.task_envs.bebop2.double_bebop2_task:DoubleBebop2TaskEnv',
        max_episode_steps=MAX_STEP,
    )

class DoubleBebop2TaskEnv(double_bebop2_env.DoubleBebop2Env):
    def __init__(self):

        # On load les paramètres 
        LoadYamlFileParamsTest(rospackage_name="openai_ros", rel_path_from_package_to_file="src/openai_ros/task_envs/bebop2/config", 
                               yaml_file_name="bebop2.yaml")       

        number_actions = rospy.get_param('/bebop2/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        #parrotdrone_goto utilisait une space bos, pas nous pour l'instant.

        # Lancement de la simulation
        ROSLauncher(rospackage_name="rotors_gazebo", launch_file_name="mav_2_bebop.launch", 
                    ros_ws_abspath=str(Path(__file__).parent.parent.parent.parent.parent.parent.parent))
        
        # Paramètres
        self.linear_forward_speed = rospy.get_param( '/bebop2/linear_forward_speed')
        self.angular_turn_speed = rospy.get_param('/bebop2/angular_turn_speed')
        self.angular_speed = rospy.get_param('/bebop2/angular_speed')

        self.init_linear_speed_vector = Vector3()

        self.init_linear_speed_vector.x = rospy.get_param( '/bebop2/init_linear_speed_vector/x')
        self.init_linear_speed_vector.y = rospy.get_param( '/bebop2/init_linear_speed_vector/y')
        self.init_linear_speed_vector.z = rospy.get_param( '/bebop2/init_linear_speed_vector/z')

        self.init_angular_turn_speed = rospy.get_param( '/bebop2/init_angular_turn_speed')


        # On charge methodes et atributs de la classe mere
        super(DoubleBebop2TaskEnv, self).__init__()


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        Appelée lorsqu'on reset la simulation
        """
        # On reset cmd_vel
        self.publish_cmd("both", 0,0,0,0)
        # self.gazebo.pauseSim()
        # self.gazebo.resetSim()
        # self.gazebo.unpauseSim()
        # Il est important dans notre cas de reset_pub juste apres le resest, c'est pour ca on reset (c'est pas grv si on resset 2 fois)
        # il est necessaire de reset deux fois pour que cela soit pris en compte 


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """


        self.reset_pub()
        rospy.sleep(0.1)
        self.gazebo.unpauseSim()

        while True:
            rospy.sleep(0.1)
            self.takeoff()
            dist_x = abs(self.L_odom.pose.pose.position.x - self.R_odom.pose.pose.position.x)
            dist_y = abs(self.L_odom.pose.pose.position.y - self.R_odom.pose.pose.position.y)
            dist_z = abs(self.L_odom.pose.pose.position.z - self.R_odom.pose.pose.position.z) 

            if self.compute_dist(dist_x, dist_y, dist_z) >1.02:
                rospy.logerr("Problème detecté, reset")
                self._reset_sim()
            else:
                break

        self.gazebo.pauseSim()
        self.number_step = 0


    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        On utilise PPO continue, les actions seront un vecteur de taille 4 continue
        action = [linear.x, linear.y, linear.z, angular.z]
        On fait bouger le R_bebop qui suit le L_bebop
        """
        lin_x, lin_y, lin_z, ang_z = action
        self.publish_cmd("R_bebop2",lin_x,lin_y,lin_z,ang_z)
        self.number_step += 1
        

    def _get_obs(self):
        """
        Obsevations : 
        -   distance between drones obs[0...2]
        -   L drone speed, R drone speed : obs[3...6], obs[6...9] 
        -   Euler orientaiton of both drones

        """
        # Distance
        dist_x = abs(self.L_odom.pose.pose.position.x - self.R_odom.pose.pose.position.x)
        dist_y = abs(self.L_odom.pose.pose.position.y - self.R_odom.pose.pose.position.y)
        dist_z = abs(self.L_odom.pose.pose.position.z - self.R_odom.pose.pose.position.z)


        # Speed
        L_speed_x = self.L_odom.twist.twist.linear.x
        L_speed_y = self.L_odom.twist.twist.linear.y
        L_speed_z = self.L_odom.twist.twist.linear.z
        L_angular_z= self.L_odom.twist.twist.angular.x

        R_speed_x = self.R_odom.twist.twist.linear.x
        R_speed_y = self.R_odom.twist.twist.linear.y
        R_speed_z = self.R_odom.twist.twist.linear.z
        R_angular_z= self.R_odom.twist.twist.angular.x

        # Orientation
        L_roll, L_pitch, L_yaw = self.get_orientation_euler(self.L_odom.pose.pose.orientation)
        R_roll, R_pitch, R_yaw = self.get_orientation_euler(self.R_odom.pose.pose.orientation)

        # wrap angles between -PI and PI degrees
        L_roll, L_pitch, L_yaw = self.wrap_angle(L_roll), self.wrap_angle(L_pitch), self.wrap_angle(L_yaw) 
        R_roll, R_pitch, R_yaw = self.wrap_angle(R_roll), self.wrap_angle(R_pitch), self.wrap_angle(R_yaw)
        


        observation = np.array([dist_x, dist_y, dist_z, L_speed_x, L_speed_y, L_speed_z, L_angular_z, R_speed_x, R_speed_y, R_speed_z, R_angular_z, L_roll, L_pitch, L_yaw, R_roll, R_pitch, R_yaw])
        return  observation
    



    def wrap_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))



    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        
        L'episode se finit si: 
        -   l'un des drones se retourne.
        -   les drones sont trop éloignés ou trop proche
        -   le drone a fait un bon nombre de step sans perdre
        """
        done = False

        #Check if one of the two UAV is upside down
        L_roll = observations[11]
        L_pitch = observations[12]

        R_roll = observations[14]
        R_pitch = observations[15]
        L_yaw, R_yaw = observations[13], observations[16]
        
        yaw_error = abs(L_yaw-R_yaw)

        if abs(L_pitch) >= 1.57 or abs(L_roll) >= 1.57:
            rospy.logdebug("Le L_bebop s'est retourné")
            done = True

        if abs(R_pitch) >= 1.57 or abs(R_roll) >= 1.57:
            rospy.logdebug("Le R_bebop s'est retourné")
            done = True
        
        # Check the distance between the drone
        dist_x, dist_y, dist_z = observations[0:3]
        distance = self.compute_dist(dist_x, dist_y)

        if distance > 1.5 or distance < 0.5: done = True
        if dist_z > 0.2: done = True
        if yaw_error > 0.53 : done = True # ~ 30 degrees 
        if self.L_odom.pose.pose.position.z < 0.2 or self.R_odom.pose.pose.position.z < 0.2 : done = True

        return done
    

    def _compute_reward(self, observations, done):
        """ On veut que les drones soient synchronisé avec un espace de 1 metre entre eux
            Par la suite on voudra aussi que le drones esquives les obstacles (pas pour l'instant)
        On utilisera une reward linéiar en fonction de la distance entre les drones
        """
        # System rewards 1 2 ou 3 
        reward = self.reward_system3(observations, done)
        return reward
        
    # Internal TaskEnv Methods


    def compute_dist(self,*args):
        d = np.array(args)
        dist = np.sum(d**2)**0.5
        return dist


    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw


##### REWARD SYSTEM : 

    def reward_system1(self, observations, done):
        L_roll = observations[11]
        L_pitch = observations[12]

        R_roll = observations[14]
        R_pitch = observations[15]

        dist_x, dist_y, dist_z = observations[0:3]
        distance = self.compute_dist(dist_x, dist_y)
        reward = 0
        
        if not done: 
            if abs(1 - distance) < 0.1: 
                reward += 10
            elif distance > 1:
                #Pourcentage
                reward += -100*(distance - 1)/0.5
                # assert reward <= 0
                
            elif distance < 1:
                # Pourcentage 
                reward += -300*(distance - 0.5)/0.5
                # assert reward <= 0

            if dist_z > 0.1:
                reward -= 300 
            else: reward +=30

            # if self.L_pose.position.z < 0.2 or self.R_pose.position.z < 0.2 :
            #     reward -= 30
            # else:
            #     reward += 3

        
        else:
            if distance < 0.5:
                # Si l'episode se termine acvec les drones trop proche (pret a se cogner)
                reward += -500
            elif distance > 1.5:
                # Si les drones se sont trop éloigné
                reward += -200
            else:
                reward += 10
            
            if dist_z > 0.1:
                reward -= 300 
            else: reward +=30
            

            if abs(L_pitch) >= 1.57 or abs(L_roll) >= 1.57 or abs(R_pitch) >= 1.57 or abs(R_roll) >= 1.57:
                reward += -500
            else: 
                reward += 10

            
            if self.number_step >= MAX_STEP:
                #Dans ce cas on aura fini l'épiosde sans acro
                reward += 1000
        return reward
    
    def reward_system2(self, observations, done):

        #end episode
        out_reward = -150
        too_near_reward = -200
        total_bad_altitude_reward = -150
        good_yaw = 15
        bad_yaw = -20
        good_distance_reward = 20
        good_altitude_reward = 20
        bad_distance_reward = -40
        bad_altitude_reward = -40
        step_reward = 5


        dist_x, dist_y, dist_z = observations[0:3]
        L_yaw, R_yaw = observations[13], observations[16]
        
        yaw_error = abs(L_yaw-R_yaw)

        distance = self.compute_dist(dist_x, dist_y)
        if done :
            reward = 0
            if distance > 1.5:
                reward += out_reward
            if distance < 0.5:
                reward += too_near_reward
            if dist_z > 0.2:
                reward += total_bad_altitude_reward
            return reward

        else:
            reward = 0
            if abs(distance -1 ) < 0.1:
                reward += good_distance_reward
            else : 
                reward += bad_distance_reward
            if dist_z < 0.1: 
                reward += good_altitude_reward
            else:
                reward += bad_altitude_reward

            if yaw_error < 0.2 : # ~11 degrees
                reward += good_yaw
            else:
                reward += bad_yaw

            reward += step_reward
            return reward

    def reward_system3(self, observations, done):

        out_reward = -150
        too_near_reward = -200
        total_bad_altitude_reward = -150
        good_yaw = 15
        bad_yaw = -20
        good_distance_reward = 20
        good_altitude_reward = 20
        step_reward = 5

        speed_penalty = -100


        dist_x, dist_y, dist_z = observations[0:3]
        Lvx, Lvy, Lvz, Laz, Rvx, Rvy, Rvz, Raz = observations[3:11]
        L_yaw, R_yaw = observations[13], observations[16]
        
        yaw_error = abs(L_yaw-R_yaw)

        distance = self.compute_dist(dist_x, dist_y)
        if done :
            reward = 0
            if distance > 1.5:
                reward += out_reward
            if distance < 0.5:
                reward += too_near_reward
            if dist_z > 0.2:
                reward += total_bad_altitude_reward
            return reward

        else:
            reward = 0
            if abs(distance -1 ) < 0.1:
                reward += good_distance_reward

            if dist_z < 0.1: 
                reward += good_altitude_reward

            if yaw_error < 0.2 : # ~11 degrees
                reward += good_yaw
            else:
                reward += bad_yaw
            
            # Il faut que les vitesses soient les memes (les drones vont au meme endroit)
            if Lvx*Rvx < 0  and abs(Lvx - Rvx) > 0.05:
                reward += speed_penalty

            if Lvy*Rvy < 0  and abs(Lvy - Rvy) > 0.05:
                reward += speed_penalty

            if Lvz*Rvz < 0  and abs(Lvz - Rvz) > 0.05:
                reward += speed_penalty

            if Laz*Raz < 0  and abs(Laz - Raz) > 0.05:
                reward += speed_penalty

            reward += step_reward
            return reward


            
            
            




