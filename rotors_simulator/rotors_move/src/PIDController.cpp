#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath>

#include "rotors_move/PIDController.h"


int main(int argc, char** argv) {
  ros::init(argc, argv, "pid_controller_node");  

  PIDController pid_controller;

  std::string waypoints_file = "/home/youcef/drones_ws/src/rotors_simulator/rotors_move/launch/waypoints.txt";

  pid_controller.readWaypointsFromFile(waypoints_file);
  
  ros::spin();

  return 0;
}