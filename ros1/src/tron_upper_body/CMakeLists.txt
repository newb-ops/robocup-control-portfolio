cmake_minimum_required(VERSION 3.0.2)
project(tron_upper_body)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
)

add_message_files(
  FILES
  MotorCommand.msg
  MotorFeedback.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
