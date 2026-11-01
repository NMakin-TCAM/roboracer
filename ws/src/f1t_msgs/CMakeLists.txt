cmake_minimum_required(VERSION 3.8)
project(f1t_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
  "msg/Wpnt.msg"
  "msg/WpntArray.msg"
  "msg/OTWpntArray.msg"
  "msg/OppWpnt.msg"
  "msg/OpponentTrajectory.msg"
  "msg/ProjOppPoint.msg"
  "msg/ProjOppTraj.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_package()
