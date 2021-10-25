#include<ros/ros.h>
#include<iostream>
#include<ros_tutorial/counterAction.h>
#include<actionlib/server/simple_action_server.h>

typedef actionlib::SimpleActionServer<ros_tutorial::counterAction> Server;

void execute(const ros_tutorial::counterGoalConstPtr& counter_goal, Server* as){

	ros::Rate r(1);
	ros_tutorial::counterFeedback feedback;

	ROS_INFO("THE GOAL IS:%d", counter_goal->goal_num);

	for(int count = 0; count<counter_goal->goal_num; ++count){
		feedback.complete_percent = count;
		as -> publishFeedback(feedback);	

		r.sleep();
	}
	
	ROS_INFO("COUNT DONE");
	
	as -> setSucceeded();
}

int main(int argc, char **argv){

	ros::init(argc, argv, "action_server");
	ros::NodeHandle nh;

	Server server(nh, "action_demo", boost::bind(&execute, _1, &server), false);

	server.start();

	ros::spin();
	return 0;
	
}
