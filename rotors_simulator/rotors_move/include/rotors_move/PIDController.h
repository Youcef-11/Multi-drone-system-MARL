#ifndef PIDCONTROLLER_H
#define PIDCONTROLLER_H

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <cmath>

#include <fstream>
#include <vector>

struct Waypoint {
  double x;
  double y;
  double z;
  double yaw;
};

class PIDController {
 public:
  PIDController();
  ~PIDController() {}

  void readWaypointsFromFile(const std::string& filename);
  void setWaypoint(Waypoint& wp);
  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void update();
  ros::NodeHandle nh_;
  int waypoint_index_;
  std::vector<Waypoint> waypoints_;

 private:
  ros::Publisher cmd_vel_pub_;
  ros::Subscriber odom_sub_;
  
  double kp_x_, ki_x_, kd_x_, waypoint_tolerance_x_;
  double kp_y_, ki_y_, kd_y_, waypoint_tolerance_y_;
  double kp_z_, ki_z_, kd_z_, waypoint_tolerance_z_;
  double kp_yaw_, ki_yaw_, kd_yaw_, waypoint_tolerance_yaw_;
  double x_, y_, z_, yaw_;
  double x_des_, y_des_, z_des_, yaw_des_;
  double prev_x_error_, prev_y_error_, prev_z_error_, prev_yaw_error_;
  double i_error_x_, i_error_y_, i_error_z_, i_error_yaw_;
  ros::Time last_time_;
};

PIDController::PIDController() {
  // Initialize class variables here
  waypoint_tolerance_x_ = 0.05;
  waypoint_tolerance_y_ = 0.05;
  waypoint_tolerance_z_ = 0.05;
  waypoint_tolerance_yaw_ = 0.1;
  waypoint_index_ = 0;
  prev_x_error_ = 0.0;
  prev_y_error_ = 0.0;
  prev_z_error_ = 0.0;
  prev_yaw_error_ = 0.0;
  i_error_x_ = 0.0;
  i_error_y_ = 0.0;
  i_error_z_ = 0.0;
  i_error_yaw_ = 0.0;
  last_time_ = ros::Time::now();

  // Initialize tpics
  cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(
                    "/L_bebop/cmd_vel",1);

  odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("/L_bebop/ground_truth/odometry", 10, &PIDController::odomCallback, this);

  // Read k factors from the parameter server
  nh_.getParam("kp_x", kp_x_);
  nh_.getParam("kp_y", kp_y_);
  nh_.getParam("kp_z", kp_z_);
  nh_.getParam("kp_yaw", kp_yaw_);
  nh_.getParam("ki_x", ki_x_);
  nh_.getParam("ki_y", ki_y_);
  nh_.getParam("ki_z", ki_z_);
  nh_.getParam("ki_yaw", ki_yaw_);
  nh_.getParam("kd_x", kd_x_);
  nh_.getParam("kd_y", kd_y_);
  nh_.getParam("kd_z", kd_z_);
  nh_.getParam("kd_yaw", kd_yaw_);
}

#endif // PIDCONTROLLER_H