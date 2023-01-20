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
#include <tf2/LinearMath/Vector3.h>
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
    // double kp_yaw, ki_yaw, kd_yaw;
    // double kp_rpy, ki_rpy, kd_rpy;
    // double yaw_;
    // double roll_, pitch_;
    // double roll_error, pitch_error;
    // double yaw_error;
    double x_error;
    double y_error;
    double z_error;


    // Define the previous error
    double prev_x_error_, prev_y_error_, prev_z_error_;
    // double prev_roll_error_, prev_pitch_error_, prev_yaw_error_; 

    // Define the integral and derivative errors
    double i_error_x_, i_error_y_, i_error_z_;
    // double i_error_roll_, i_error_pitch_, i_error_yaw_;

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
        // nh_.getParam("kp_yaw", kp_yaw);
        // nh_.getParam("kd_yaw", kd_yaw);
        // nh_.getParam("ki_yaw", ki_yaw);
        // nh_.getParam("kp_rpy", kp_rpy);
        // nh_.getParam("kd_rpy", kd_rpy);
        // nh_.getParam("ki_rpy", ki_rpy);

        prev_x_error_ = 0.0;
        prev_y_error_ = 0.0;
        prev_z_error_ = 0.0;

        // prev_roll_error_ = 0.0;
        // prev_pitch_error_ = 0.0;
        // prev_yaw_error_ = 0.0;

        i_error_x_ = 0.0;
        i_error_y_ = 0.0;
        i_error_z_ = 0.0;

        // i_error_roll_ = 0.0;
        // i_error_pitch_ = 0.0;
        // i_error_yaw_ = 0.0;

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
        gt_pos_sub_ = nh_.subscribe("/bebop/odom", 1, &MoveDroneAction::subscriberCb, this);
        ROS_INFO("Subscribers Initialized");
    }
    
    //initialize publishers here
    void initializePublishers(void)
    {
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/bebop/cmd_vel", 1);
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
        // pos_info_.orientation.x = info->pose.pose.orientation.x;
        // pos_info_.orientation.y = info->pose.pose.orientation.y,
        // pos_info_.orientation.z = info->pose.pose.orientation.z;
        // pos_info_.orientation.w = info->pose.pose.orientation.w;

        // tf2::Quaternion quat(
        //     info->pose.pose.orientation.x,
        //     info->pose.pose.orientation.y,
        //     info->pose.pose.orientation.z,
        //     info->pose.pose.orientation.w
        // );

        // tf2::Matrix3x3 mat(quat);
        // double roll, pitch, yaw;
        // mat.getRPY(roll, pitch, yaw);
        // this->roll_ = roll;
        // this->pitch_ = pitch;
        // this->yaw_ = yaw;
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
        prev_x_error_ = 0;
        prev_y_error_ = 0;
        prev_z_error_ = 0;
        // prev_yaw_error_ = 0;

        //ros::Duration(0.5).sleep();
        // takeoff_pub_.publish(lift_); //take off drone - though ideally should not be required. may be some simulation bug
        
        std::cout<<"Position coordinates received are:\n";
        std::cout<<"x: "<<goal->x<<"\ny: "<<goal->y<<"\nz: "<<goal->z<<"\n";
        do
        {   
            // tf2::Quaternion desired_quat, current_quat(pos_info_.orientation.x, pos_info_.orientation.y, 
            //                                            pos_info_.orientation.z,pos_info_.orientation.w);

            // Set the desired quaternion using roll, pitch, and yaw
            // desired_quat.setRPY(0, 0, goal->yaw * M_PI / 180.0);

            

            x_error = (double )(goal->x - pos_info_.position.x);
            y_error = (double )(goal->y - pos_info_.position.y);
            z_error = (double )(goal->z - pos_info_.position.z);
            //Compute the error quaternion
            // tf2::Quaternion error_quat = desired_quat*current_quat.inverse();
            // Compute the angular velocity error
            // tf2::Vector3 error = error_quat.getAngle() * error_quat.getAxis();
            
            // roll_error = error.x();
            // pitch_error = error.y();
            // yaw_error = error.z();

            // wrap yaw error between -PI and PI degrees
            // if ((double )(yaw_error) > M_PI) {
            //     yaw_error -= 2*M_PI;
            // } else if (yaw_error < -M_PI) {
            //     yaw_error += 2*M_PI;
            // }

            // Compute the integral error
            // i_error_x_ += x_error;
            // i_error_y_ += y_error;
            // i_error_z_ += z_error;
            // i_error_roll_ += roll_error;
            // i_error_pitch_ += pitch_error;
            // i_error_yaw_ += yaw_error;

            // Compute the derivative error
            double d_error_x = (x_error - prev_x_error_);
            double d_error_y = (y_error - prev_y_error_);
            double d_error_z = (z_error - prev_z_error_);
            // double d_error_roll = (roll_error - prev_roll_error_);
            // double d_error_pitch = (pitch_error - prev_pitch_error_);
            // double d_error_yaw = (yaw_error - prev_yaw_error_);  

            // Compute the control outputs
            // double vx = kp * x_error + ki * i_error_x_ + kd * d_error_x;
            // double vy = kp * y_error + ki * i_error_y_ + kd * d_error_y;
            // double vz = kp * z_error + ki * i_error_z_ + kd * d_error_z;
            // double vroll = kp_rpy * roll_error + ki_rpy * i_error_roll_ + kd_rpy * d_error_roll;
            // double vpitch = kp_rpy * pitch_error + ki_rpy * i_error_pitch_ + kd_rpy * d_error_pitch;
            // double vyaw = kp_rpy * yaw_error + ki_rpy * i_error_yaw_ + kd_rpy * d_error_yaw;
            double vx = kp * x_error + kd * d_error_x;
            double vy = kp * y_error + kd * d_error_y;
            double vz = kp * z_error + kd * d_error_z;  
            // double vyaw = kp_rpy * yaw_error + kd_rpy * d_error_yaw;   
            

            // Important : Don't forget to normalize commands 
            if (vx > 1) vx=1; else if(vx < -1) vx=-1;
            if (vy > 1) vy=1; else if(vy < -1) vy=-1;
            if (vz > 1) vz=1; else if(vz < -1) vz=-1;
            // if (vroll > 1) vroll=1; else vroll=-1;
            // if (vpitch > 1) vpitch=1; else vpitch=-1;
            // if (vyaw > 1) vyaw=1; else if(vyaw < -1) vyaw=-1;



            ROS_INFO("Velocities :\nVx : %f\nVy : %f\nVz : %f", vx, vy, vz);

            // Save the errors for the next iteration
            prev_x_error_ = x_error;
            prev_y_error_ = y_error;
            prev_z_error_ = z_error;
            // prev_roll_error_ = roll_error;
            // prev_pitch_error_ = pitch_error;
            // prev_yaw_error_ = yaw_error;
            
            // send velocity
            move.linear.x = vx;
            move.linear.y = vy;
            move.linear.z = vz;
            // move.angular.x = vroll;
            // move.angular.y = vpitch;
            // move.angular.z = vyaw;

            cmd_vel_pub_.publish(move);

            feedback_.distance = calDistance(pos_info_, goal);;

            action_server_.publishFeedback(feedback_); //echo /action_server/feedback

            // ROS_INFO("Velocities :\nCurrent Yaw : %f\nTarget Yaw : %f\nYaw error : %f", yaw_*180.0/M_PI, goal->yaw, yaw_error);

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
        while(feedback_.distance > 0.08);

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