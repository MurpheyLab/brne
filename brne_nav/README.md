# BRNE Social Navigation

This set of ROS 2 packages implements Bayes Rule Nash Equilibrium (BRNE) crowd navigation as well as pedestrian tracking and odometry updates using a ZED2i camera. It can be deployed on a Unitree Go1 quadruped, but nothing about this package is specific to this robot. A seperate repo was originally maintained under [https://github.com/katie-hughes/brne_social_nav](https://github.com/katie-hughes/brne_social_nav).

Code authors:

- [Katie Hughes](https://katie-hughes.github.io/) developed the ROS infrastructure and a C++ implementation of the ROS-integrated BRNE algorithm.  
- [Max Muchen Sun](https://msun.fyi/) implemented the ROS-integrated algorithm in Python (Numba) and PyTorch (deployed on an Nvidia Jetson AGX Orin).


## Clone Packages
To set up the necessary packages:
```
mkdir -p ${ws}/src
cd ${ws}/src
git clone https://github.com/MurpheyLab/brne.git
cd ..
vcs import --recursive < src/brne/brne_nav/repos/crowdnav.repos
```

## Build Packages
To build all packages:
```
colcon build
```
If you want to build this project to only test in simulation, or for use on an external computer that doesn't have/need the ZED SDK, you can build the project with the following command instead:
```
colcon build --packages-select crowd_nav brnelib crowd_nav_interfaces pedestrian_tracking brne_py zed_interfaces
```
Finally, if you are building only the perception related portions of this project (for example, to deploy on an Orin Nano), you can build with the following command:
```
colcon build --packages-select crowd_nav_interfaces zed_components zed_ros2 zed_wrapper zed_interfaces
```

## Languages

This project contains both Python (Numba or PyTorch) and C++ implementations of the BRNE algorithm code. The behavior should be identical. To specify which nodes get run, use the launch argument `lang:=C++` or `lang:=PYTHON` or `lang:=TORCH` (C++ is default). If you are interested in the time it takes for each iteration of the algorithm to execute, you can run with the launch argument `debug_level:=debug`.

## Parameters
The parameters of the BRNE algorithm, such as the number of samples and the cost parameters, can be tuned by editing [brne.yaml](crowd_nav/config/brne.yaml).

# Running On a Unitree Go1

## Connecting to the robot
The Unitree Go1 has an onboard WLAN hotspot that is bridged to the ethernet connections between the boards. When the robot is powered on, check on your compter to see if the `UnitreeWifi` network is visible. In the event that it doesn't come up after a few minutes, the robot will need to be rebooted.

Once you connect to the wifi, you will need to configure a static IP on your own computer that matches the subnet `192.168.123.xxx`, which is what the boards on the robot are also set up with. (I usually set my IP to `192.168.123.100`). Then, you should be able to directly ssh into and see ros topics from all boards inside the dog.

If at any point you stop being able to ping these boards or your ssh session freezes, the hotspot has probably died. The only way around this issue that I have found is to turn the robot off for a few minutes, then power it back on. 

## Go1 Xavier Setup
The result of the BRNE algorithm is a motion plan represented as a series of `cmd_vel` commands. In order to process these commands, the [`unitree_nav`](https://github.com/ngmor/unitree_nav) package should be installed and running onboard the Go1 Xavier. To process the `cmd_vel`, run:
```
ros2 launch unitree_nav control.launch.py use_rviz:=false
```

## Using an onboard laptop
If using a laptop strapped to the back of the robot, this computer will be responsible for pedestrian tracking with the ZED, odometry updates from the ZED, and the BRNE algorithm itself. To set all of this up, run 

```
ros2 launch crowd_nav onboard.launch.xml
```
As in simulation, you can choose the C++ or Python implementation of the BRNE nodes with `lang:=C++` or `lang:=PYTHON` (C++ is default), and you can see the time for each algorithm iteration to execute with the launch argument `debug_level:=debug`.

Then, on an external computer, you will need to launch rviz to see the visualizations as well as to set the goal location. This can be done via
```
ros2 launch crowd_nav visualizations.launch.xml
```

## Using an onboard Jetson 

If using a Jetson mounted to the back of the robot, this board will be connected to the ZED and only responsible for the pedestrian tracking portion of the project. On this device, run:
```
ros2 launch pedestrian_tracking orin.launch.xml
```

The external computer will be responsible for the BRNE nodes and visualization, as the Orin Nano does not have enough CPU to handle this algorithm. On this device, run:
```
ros2 launch crowd_nav algorithm.launch.xml
```
Once again, you can choose the C++ or Python implementation of the BRNE nodes with `lang:=C++` or `lang:=PYTHON` (C++ is default), and you can see the time for each algorithm iteration to execute with the launch argument `debug_level:=debug`.

# Dependencies

This project depends on a recent version of armadillo for the C++ library. If you get errors while building, you will have to install the newest version from source. At the time of writing this, the stable version of armadillo is version 12.6.6, which is available at [Armadillo's download center](https://arma.sourceforge.net/download.html) ([download link here](https://sourceforge.net/projects/arma/files/armadillo-12.6.6.tar.xz)). Download this file, then:
```
tar xf armadillo-12.6.6.tar.xz
cd armadillo-12.6.6
mkdir build
cd build
cmake ..
make
sudo make install
```

The C++ implementation of the BRNE algorithm also depends on Catch2 for unit testing purposes. If this package is not found, you will also have to build it from source.
```
git clone https://github.com/catchorg/Catch2.git
cd Catch2
mkdir build
cd build
cmake ..
make
sudo make install
```


# ROS 2 Packages
- [brne_py](brne_py) - A Python implemntation of the BNRE algorithm using Numba.
- [brne_torch](brne_torch) - A PyTorch implemntation of the BNRE algorithm.
- [crowd_nav_interfaces](crowd_nav_interfaces) - Defines some custom message types relevant to crowd navigation.
- [crowd_nav](crowd_nav) - A ROS 2, C++ implementation of the BRNE algorithm, which also contains the core launchfiles.
- [pedestrian_tracking](pedestrian_tracking) - Tracks, displays, and can simulate pedestrian locations.

# Libraries
- [brnelib](brnelib) - Implementation of the BRNE algorithm in C++
