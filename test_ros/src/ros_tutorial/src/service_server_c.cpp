#include<ros/ros.h>
#include<ros_tutorial/add.h>
#include<iostream>

using namespace std;

bool add(ros_tutorial::add::Request &req, ros_tutorial::add::Response &resp)
{
	resp.sum = req.A+req.B;
	ROS_INFO("service add handle...");

	return true;
}

int main(int argc, char **argv){ //char *argvp[]
	
	ros::init(argc, argv, "service_add");

	ros::NodeHandle n;

	ros::ServiceServer service = n.advertiseService("serviceadd", add);

	ros::spin();

	return 0;

}
