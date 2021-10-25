#include<ros/ros.h>
#include<ros_tutorial/counterAction.h>
#include<iostream>
#include<actionlib/client/simple_action_client.h>

void doneCb(const actionlib::SimpleClientGoalState& state, 
	const ros_tutorial::counterResultConstPtr& result){
	
	ROS_INFO("DONE");
	ros::shutdown();
}

void activeCb(){
	ROS_INFO("ACTIVE");
}

void feedbackCb(const ros_tutorial::counterFeedbackConstPtr& feedback){
	ROS_INFO("THE NUMBER RIGHT NOM OS:%f", feedback->complete_percent);
}

int main(int argc, char **argv){
		
	/*初始化节点*/
	ros::init(argc, argv, "action_client");

	/* 定义一个客户端 */
	actionlib::SimpleActionClient<ros_tutorial::counterAction> client("action_demo", true);
	
	ROS_INFO("WAITING FOR ACTION SERVER TO START!");
	client.waitForServer();
	ROS_INFO("ACTION SERVER START!");
	
	/* 创建一个目标对象 */
	ros_tutorial::counterGoal counter_goal;
	counter_goal.goal_num = 100;  /* 设置目标对象的值 */

	/* 发送目标，并且定义回调函数 */
	client.sendGoal(counter_goal, &doneCb, &activeCb, &feedbackCb);

	ros::spin();
	return 0;
}
