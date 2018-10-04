#include <iostream>
#include <fstream>
#include <signal.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>

using namespace std;

ofstream ofd;

void shutdown(int s)
{
  ofd.close();
  _exit(0);
}

int 
main(int argc, char * argv[])
{
  ros::init(argc, argv, "getMonkeys");
  ros::NodeHandle n;
  if (argc < 1) 
    return -1;
  ofd.open(argv[1]);
  tf::TransformListener listener(n);
  signal(SIGINT, shutdown);
  
  tf::StampedTransform leftShoulder;
  tf::StampedTransform rightShoulder; 
  bool foundLeft = true, foundRight = true; 
  bool first = true;
  ros::Time t0L, t0R;
  while (n.ok())
    {
      foundLeft = foundRight = true; 
      cout << "Loop"<< endl;
      // Compute the relative position of the left hand wrt the head
      ros::Time now = ros::Time::now();
      try {
	listener.waitForTransform("openni_depth_frame", "left_shoulder_1", now, ros::Duration(0.1));
	listener.lookupTransform("openni_depth_frame", "left_shoulder_1", now, leftShoulder);
      }
      catch (tf::TransformException ex) {
	ROS_ERROR("Left shoulder: %s",ex.what());
	foundLeft = false;
      }
      // Compute the relative position of the right hand wrt the head
      try {
	listener.waitForTransform("openni_depth_frame", "right_shoulder_1", now, ros::Duration(0.1));
	listener.lookupTransform("openni_depth_frame", "right_shoulder_1", now, rightShoulder);
      }
      catch (tf::TransformException ex) {
	ROS_ERROR("Right shoulder: %s",ex.what());
	foundRight = false;
      }
      //ros::spin();
      if (foundRight & foundLeft)
	{
	  if (first)
	    {
	      t0L = leftShoulder.stamp_;
	      t0R = rightShoulder.stamp_;
	      first = false;
	    }
	  
	  ofd << (leftShoulder.stamp_ - t0L)  << " "
	      << leftShoulder.getOrigin().x() << " "
	      << leftShoulder.getOrigin().y() << " "
	      << leftShoulder.getOrigin().z() << " "
	      << (rightShoulder.stamp_ - t0R)  << " "
	      << rightShoulder.getOrigin().x() << " "
	      << rightShoulder.getOrigin().y() << " "
	      << rightShoulder.getOrigin().z() << endl;
	}
    }
  
  return 0;
}
