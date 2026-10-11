cmake_minimum_required(VERSION 3.5)
project(control)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Install Python modules (looks for a folder named 'control/' in your package root)
ament_python_install_package(${PROJECT_NAME})

# Install Python executable scripts into lib/control
install(PROGRAMS
  scripts/pure_pursuit_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files into share/control
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install waypoint files into share/control
install(DIRECTORY waypoints
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ==============================================================================
# MISSING STEP: Register dependencies and package with the Ament Index
# ==============================================================================
ament_export_dependencies(
  rclpy
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2_ros
)

ament_package()