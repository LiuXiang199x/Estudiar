#include<iostream>
#include<ros/ros.h>
#include<ros_tutorial/add.h>
#include<unistd.h>

int main(int argc, char **argv){

	ros::init(argc, argv, "client_add");

	ros::NodeHandle n;

	ros::ServiceClient client = n.serviceClient<ros_tutorial::add>("serviceadd");

	ros_tutorial::add srv;
	

	for(int i=0; i<50; i++){
		std::cout << i << std::endl;

		srv.request.A = i;
		srv.request.B = ++i;

		if(client.call(srv)){
			ROS_INFO("client_add success, result:%d", srv.response.sum);
			// return 1;
		}
		else{
			ROS_INFO("client_add failed...");
			// return 0;	
		}
		sleep(1);
	}
	return 0;
}
