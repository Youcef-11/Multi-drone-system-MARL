#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <math.h>
#include <algorithm>
#include <iostream>

#include "set_position/SetPositionMsgAction.h"

//class containing action server methods 
class MoveDroneAction

{
    protected:
    ros::NodeHandle nh_; //a NodeHandle to create subscribers and publishers
    actionlib::SimpleActionServer<set_position::SetPositionMsgAction> action_server_; //declare a simple action server type object
    std::string action_name_; //variable holding the name of our action server node 
    set_position::SetPositionMsgFeedback feedback_; //variable stores the feedback/intermediate results
    set_position::SetPositionMsgResult result_; //variable stores the final output
    //define your subscribers here
    ros::Subscriber gt_pos_sub_;
    geometry_msgs::Pose pos_info_;
    //define your publishers here
    ros::Publisher cmd_vel_pub_;
    ros::Publisher posctrl_pub_; 
    ros::Publisher takeoff_pub_;
    std_msgs::Empty lift_;
    double kp, kd, ki;
    double kp_yaw, ki_yaw, kd_yaw;
    double yaw_, yaw_error;
    double x_error;
    double y_error;
    double z_error;
    double prev_x_error_, prev_y_error_, prev_z_error_, prev_yaw_error_;
    double i_error_x_, i_error_y_, i_error_z_, i_error_yaw_;

    public:
    MoveDroneAction(std::string name) : 
        action_server_(nh_, name, boost::bind(&MoveDroneAction::actionCb, this, _1),false), 
        action_name_(name)
    {
        initializeSubscribers();
        initializePublishers();
        nh_.getParam("kp", kp);
        nh_.getParam("kd", kd);
        nh_.getParam("ki", ki);
        nh_.getParam("kp_yaw", kp_yaw);
        nh_.getParam("kd_yaw", kd_yaw);
        nh_.getParam("ki_yaw", ki_yaw);
        prev_x_error_ = 0.0;
        prev_y_error_ = 0.0;
        prev_z_error_ = 0.0;
        prev_yaw_error_ = 0.0;
        i_error_x_ = 0.0;
        i_error_y_ = 0.0;
        i_error_z_ = 0.0;
        i_error_yaw_ = 0.0;
        action_server_.start();
    }

    ~MoveDroneAction(void) 
    {
        //stuff for destructor
    }  

    

    private: 
    //initialize subscribers here
    void initializeSubscribers(void)
    {
        gt_pos_sub_ = nh_.subscribe("/L_bebop2/ground_truth/odometry", 1, &MoveDroneAction::subscriberCb, this);
        ROS_INFO("Subscribers Initialized");
    }
    
    //initialize publishers here
    void initializePublishers(void)
    {
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/L_bebop2/cmd_vel", 1);
        // takeoff_pub_ = nh_.advertise<std_msgs::Empty>("/L_bebop2/takeoff", 1);
        ROS_INFO("Publishers Initialized");
    }

    
    //callback for our gt_vel subscriber
    //it receives velocity information in the form of Twist message
    void subscriberCb(const nav_msgs::Odometry::ConstPtr& info)
    {
        pos_info_.position.x = info->pose.pose.position.x;
        pos_info_.position.y = info->pose.pose.position.y;
        pos_info_.position.z = info->pose.pose.position.z;
        tf2::Quaternion quat(
            info->pose.pose.orientation.x,
            info->pose.pose.orientation.y,
            info->pose.pose.orientation.z,
            info->pose.pose.orientation.w
        );

        tf2::Matrix3x3 mat(quat);
        double roll, pitch, yaw;
        mat.getRPY(roll, pitch, yaw);
        this->yaw_ = yaw;
    }

    

    //helper function to calculate euclidean distance between two 3d points in space
    double calDistance(geometry_msgs::Pose current, const set_position::SetPositionMsgGoalConstPtr &goal)
    {
        double dist;
        dist = sqrt(pow((goal->x - current.position.x),2) + pow((goal->y - current.position.y),2) + pow((goal->z - current.position.z),2));
        //std::cout<<"dist: "<<dist<<'\n';
        return dist;
    }

    

    //main action server callback
    void actionCb(const set_position::SetPositionMsgGoalConstPtr &goal)
    {   
        ros::Rate rate(100);
        bool success = true;
        //do the cool stuff here - i have to move the drone!
        geometry_msgs::Twist move;

        //ros::Duration(0.5).sleep();
        // takeoff_pub_.publish(lift_); //take off drone - though ideally should not be required. may be some simulation bug
        
        std::cout<<"Position coordinates received are:\n";
        std::cout<<"x: "<<goal->x<<"\ny: "<<goal->y<<"\nz: "<<goal->z<<"\nyaw: "<<goal->yaw<<"\n";
        do
        {
            // if (goal->x - pos_info_.position.x > 0)
            //     move.linear.x = k_x*std::min(1.0, (double )(goal->x - pos_info_.position.x));
            // else
            //     move.linear.x = k_x*std::max(-1.0, (double )(goal->x - pos_info_.position.x));

            // if (goal->y - pos_info_.position.y > 0)
            //     move.linear.y = k_y*std::min(1.0, (double )(goal->y - pos_info_.position.y));
            // else
            //     move.linear.y = k_y*std::max(-1.0, (double )(goal->y - pos_info_.position.y));

            // if (goal->z - pos_info_.position.z > 0)
            //     move.linear.z = k_z*std::min(1.0, (double )(goal->z - pos_info_.position.z));
            // else
            //     move.linear.z = k_z*std::max(-1.0, (double )(goal->z - pos_info_.position.z));
            
            yaw_error = goal->yaw * M_PI / 180.0 - yaw_;

            // wrap yaw error between -PI and PI degrees
            if ((double )(yaw_error) > M_PI) {
                yaw_error -= 2*M_PI;
            } else if (yaw_error < -M_PI) {
                yaw_error += 2*M_PI;
            }

            x_error = (double )(goal->x - pos_info_.position.x);
            y_error = (double )(goal->y - pos_info_.position.y);
            z_error = (double )(goal->z - pos_info_.position.z);

            // Compute the integral error
            i_error_x_ += x_error;
            i_error_y_ += y_error;
            i_error_z_ += z_error;
            i_error_yaw_ += yaw_error;

            // Compute the derivative error
            double d_error_x = (x_error - prev_x_error_);
            double d_error_y = (y_error - prev_y_error_);
            double d_error_z = (z_error - prev_z_error_);
            double d_error_yaw = (yaw_error - prev_yaw_error_);

            // Compute the control outputs
            double vx = kp * x_error + ki * i_error_x_ + kd * d_error_x;
            double vy = kp * y_error + ki * i_error_y_ + kd * d_error_y;
            double vz = kp * z_error + ki * i_error_z_ + kd * d_error_z;
            double vyaw = kp_yaw * yaw_error + ki_yaw * i_error_yaw_ + kd_yaw * d_error_yaw;

            ROS_INFO("Velocities :\nVx : %f\nVy : %f\nVz : %f\nVyaw : %f", vx, vy, vz, vyaw);

            // Save the errors for the next iteration
            prev_x_error_ = x_error;
            prev_y_error_ = y_error;
            prev_z_error_ = z_error;
            prev_yaw_error_ = yaw_error;
            
            // send velocity
            move.linear.x = vx;
            move.linear.y = vy;
            move.linear.z = vz;
            move.angular.z = vyaw;

            cmd_vel_pub_.publish(move);

            feedback_.distance = calDistance(pos_info_, goal);;

            action_server_.publishFeedback(feedback_); //echo /action_server/feedback
            ROS_INFO("Velocities :\nCurrent Yaw : %f\nTarget Yaw : %f", yaw_*180.0/M_PI, goal->yaw);
            //take care of preemption here
            if (action_server_.isPreemptRequested() || !ros::ok())
            {
                ROS_INFO("%s: Preempted", action_name_.c_str());
                // set the action state to preempted
                action_server_.setPreempted();
                success = false;
                break;
            }
            rate.sleep();
        }
        while(feedback_.distance > 0.08 || yaw_error > 0.04);

        move.linear.x = 0;
        move.linear.y = 0;
        move.linear.z = 0;
        move.angular.x = 0;
        move.angular.y = 0;
        move.angular.z = 0; 
        cmd_vel_pub_.publish(move);

        //check if succeeded--yes-->return result_
        if(success)
        {
            result_.status = "Destination Arrived!!";
            ROS_INFO("%s: Succeeded", action_name_.c_str());
            // set the action state to succeeded
            action_server_.setSucceeded(result_);
        }
    }

};

int main(int argc, char** argv)

{
    ros::init(argc, argv, "position_controller");
    MoveDroneAction drone("position_controller");
    ros::Rate rate(100);
    while (ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
}