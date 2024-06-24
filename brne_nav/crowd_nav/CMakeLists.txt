cmake_minimum_required(VERSION 3.8)
project(crowd_nav)

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
# find_package(brnelib REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})

# add_executable(brne_nav src/brne_nav.cpp)

# ament_target_dependencies(brne_nav
#   rclcpp
#   std_msgs
#   std_srvs
#   tf2
#   tf2_ros
#   geometry_msgs
#   visualization_msgs
#   nav_msgs
#   sensor_msgs
#   image_geometry
#   crowd_nav_interfaces
#   OpenMP)

# target_link_libraries(brne_nav brnelib::brnelib  ${ARMADILLO_LIBRARIES} "${cpp_typesupport_target}")

# add_executable(controller src/controller.cpp)

# ament_target_dependencies(controller
#   rclcpp
#   std_msgs
#   std_srvs
#   tf2
#   tf2_ros
#   geometry_msgs
#   visualization_msgs
#   nav_msgs
#   sensor_msgs
#   image_geometry
#   crowd_nav_interfaces)

add_executable(pub_goal src/pub_goal.cpp)

ament_target_dependencies(pub_goal
  rclcpp
  std_srvs
  geometry_msgs
  crowd_nav_interfaces)

install(TARGETS pub_goal # brne_nav controller 
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
