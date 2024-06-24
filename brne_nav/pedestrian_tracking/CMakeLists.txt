cmake_minimum_required(VERSION 3.8)
project(pedestrian_tracking)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(crowd_nav_interfaces REQUIRED)
find_package(zed_interfaces REQUIRED)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})


add_executable(simulate_pedestrians src/simulate_pedestrians.cpp)
ament_target_dependencies(simulate_pedestrians
  rclcpp
  std_msgs
  std_srvs
  tf2
  tf2_ros
  geometry_msgs
  visualization_msgs
  nav_msgs
  sensor_msgs
  image_geometry
  crowd_nav_interfaces)

add_executable(show_pedestrians src/show_pedestrians.cpp)
  ament_target_dependencies(show_pedestrians
  rclcpp
  std_msgs
  std_srvs
  tf2
  tf2_ros
  geometry_msgs
  visualization_msgs
  nav_msgs
  sensor_msgs
  image_geometry
  crowd_nav_interfaces)

add_executable(convert_zed src/convert_zed.cpp)
  ament_target_dependencies(convert_zed
  rclcpp
  std_msgs
  std_srvs
  tf2
  tf2_ros
  geometry_msgs
  visualization_msgs
  nav_msgs
  sensor_msgs
  image_geometry
  crowd_nav_interfaces
  zed_interfaces)

target_link_libraries(convert_zed ${ARMADILLO_LIBRARIES} "${cpp_typesupport_target}")

add_executable(simulate_odom src/simulate_odom.cpp)
  ament_target_dependencies(simulate_odom
  rclcpp
  std_msgs
  std_srvs
  tf2
  tf2_ros
  geometry_msgs
  visualization_msgs
  nav_msgs
  sensor_msgs
  image_geometry
  crowd_nav_interfaces)

install(TARGETS simulate_pedestrians show_pedestrians convert_zed simulate_odom
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
