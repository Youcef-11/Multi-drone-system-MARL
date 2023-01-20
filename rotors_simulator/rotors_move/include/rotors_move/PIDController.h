#ifndef PIDCONTROLLER_H
#define PIDCONTROLLER_H

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
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
  void update(const ros::TimerEvent&);
  std::vector<Waypoint> waypoints_;
  int waypoint_index_;
  
 private:
  ros::Publisher cmd_vel_pub_;
  ros::Subscriber odom_sub_;
  ros::NodeHandle nh_;
  ros::Timer timer_;
  
  
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

  // Initialize tpics
  cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(
                    "/L_bebop2/cmd_vel",1);

  odom_sub_ = nh_.subscribe("/L_bebop2/ground_truth/odometry", 
                            10, &PIDController::odomCallback, this);

  timer_ = nh_.createTimer(ros::Duration(1/100), &PIDController::update, this);



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



void PIDController::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  x_ = msg->pose.pose.position.x;
  y_ = msg->pose.pose.position.y;
  z_ = msg->pose.pose.position.z;

  tf2::Quaternion quat(
    msg->pose.pose.orientation.x,
    msg->pose.pose.orientation.y,
    msg->pose.pose.orientation.z,
    msg->pose.pose.orientation.w
  );
  tf2::Matrix3x3 mat(quat);
  double roll, pitch, yaw;
  mat.getRPY(roll, pitch, yaw);
  this->yaw_ = yaw;

  ROS_INFO("Odom :\nx : %f\ny : %f\nz : %f\nyaw : %f", x_, y_, z_, yaw_);

}







void PIDController::readWaypointsFromFile(const std::string& file_path) {
  std::ifstream file(file_path);

  if (!file.is_open()) {
    ROS_ERROR("Could not open waypoints file at %s", file_path.c_str());
    return;
  }
  
  double x, y, z, yaw;
  
  while (file >> x >> y >> z >> yaw) {
      Waypoint wp;
      wp.x = x;
      wp.y = y;
      wp.z = z;
      wp.yaw = yaw * M_PI / 180.0;  // convert yaw from degrees to radians;
      waypoints_.push_back(wp);
  }
  ROS_INFO("Read %d waypoints.", (int )waypoints_.size());
}

void PIDController::setWaypoint(Waypoint& wp){
    x_des_ = wp.x;
    y_des_ = wp.y;
    z_des_ = wp.z;
    yaw_des_ = wp.yaw;
}




void PIDController::update(const ros::TimerEvent& event) {
  double dt = 1/100; //(ros::Time::now() - last_time_).toSec();
  // last_time_ = ros::Time::now();

  setWaypoint(waypoints_[waypoint_index_]);

  double x_error = x_des_ - x_;
  double y_error = y_des_ - y_;
  double z_error = z_des_ - z_;
  double yaw_error = yaw_des_ - yaw_;

  ROS_INFO("Destination :\nx_des : %f\ny_des : %f\nz_des : %f\nyaw : %f", x_des_, y_des_, z_des_, yaw_des_);
  // wrap yaw error between -PI and PI degrees
  if (yaw_error > M_PI) {
    yaw_error -= 2*M_PI;
  } else if (yaw_error < -M_PI) {
    yaw_error += 2*M_PI;
  }

  // Compute the integral error
  i_error_x_ += x_error * dt;
  i_error_y_ += y_error * dt;
  i_error_z_ += z_error * dt;
  i_error_yaw_ += yaw_error * dt;

  // Compute the derivative error
  double d_error_x = (x_error - prev_x_error_) / dt;
  double d_error_y = (y_error - prev_y_error_) / dt;
  double d_error_z = (z_error - prev_z_error_) / dt;
  double d_error_yaw = (yaw_error - prev_yaw_error_) / dt;

  // Compute the control outputs
  double vx = kp_x_ * x_error + ki_x_ * i_error_x_ + kd_x_ * d_error_x;
  double vy = kp_y_ * y_error + ki_y_ * i_error_y_ + kd_y_ * d_error_y;
  double vz = kp_z_ * z_error + ki_z_ * i_error_z_ + kd_z_ * d_error_z;
  double vyaw = kp_yaw_ * yaw_error + ki_yaw_ * i_error_yaw_ + kd_yaw_ * d_error_yaw;

  // Publish the control outputs
  geometry_msgs::Twist twist;
  twist.linear.x = vx;
  twist.linear.y = vy;
  twist.linear.z = vz;
  twist.angular.z = vyaw;
  cmd_vel_pub_.publish(twist);

  ROS_INFO("Velocities :\nVx : %f\nVy : %f\nVz : %f\nVyaw : %f", vx, vy, vz, vyaw);

  // Save the errors for the next iteration
  prev_x_error_ = x_error;
  prev_y_error_ = y_error;
  prev_z_error_ = z_error;
  prev_yaw_error_ = yaw_error;
  
  ROS_INFO("Errors :\nerr_x : %f\nerr_y : %f\nerr_z : %f\nerr_yaw : %f", x_error, y_error, z_error, yaw_error);

  // Check if waypoint is reached
  if (fabs(x_error) < waypoint_tolerance_x_ && fabs(y_error) < waypoint_tolerance_y_ && fabs(z_error) < waypoint_tolerance_z_ && fabs(yaw_error) < waypoint_tolerance_yaw_ ) {
    if (waypoint_index_ < waypoints_.size()-1) {
      ROS_INFO("Waypoint %d reached", (int )waypoint_index_+1); 
      ++waypoint_index_;
    } else {
      ROS_INFO("All waypoints reached!");
    }
  }
}


#endif // PIDCONTROLLER_H