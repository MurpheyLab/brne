# pedestrian_tracking
Author: Katie Hughes

This is a ROS 2 package designed to publish the location of pedestrians and the robot, either through the use of the ZED camera or simulated. 

## Nodes
* `convert_zed`: This converts the ZED pedestrian message type (`zed_interfaces::msg::ObjectsStamped`) into my custom pedestrian message type (`crowd_nav_interfaces::msg::PedestrianArray`) which is more succinct. It also converts the pedestrian locations into the `brne_odom` frame by creating a `tf` frame for them. 
* `show_pedestrians`: This node converts the `PedestrianArray` message type to markers are visualized as cylinders in RVIZ, with arrows that show the pedestrian's velocity.
* `simulate_pedestrians`: Publishes simulated pedestrian data. You can set the number of static pedestrians as a parameter, or choose to simulate a moving pedestrian that walks towards the robot.

## Launchfiles
* `zed.launch.xml`: Launches the ZED using the ZED ROS 2 wrapper with the appropriate configuration.
* `perception.launch.xml`: Launches entire perception stack, either real or simulated. This includes conversion to my custom message types and setting up the `tf` frames to be continuous. 