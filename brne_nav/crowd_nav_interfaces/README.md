# crowd_nav_interfaces
Author: Katie Hughes

This is a ROS 2 package that contains some useful message types for crowd navigation.

## Messages
* `Pedestrian`: Contains a pedestrian's pose, velocity, id, and timestamp.
* `PedestrianArray`: An array of `Pedestrian` messages.
* `TwistArray`: An array of `Twist` messages. This is the output of the trajectory plan.

## Service
* `GoalReq`: Used to publish a goal pose request (specify the x and y coordinates), which could be useful if you don't have rviz but still want to select a goal.