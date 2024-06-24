# BRNE: Fast Mixed Strategy Nash Equilibrium Algorithm for Social Navigation

![](media/brne_atrium_test.gif)

This repository contains relevant code for BRNE (Bayes' Rule Nash Equilibrum), an algorithm that computes mixed strategy Nash equilibrium for real-time human-aware navigation in dense crowds. It contains PyTorch-based tutorial and a fully ROS integrated social navigation framework powered by BRNE, implemented in both PyTorch and C++. 

A journal manuscript is currently under review, an arXiv preprint can be found [here](https://arxiv.org/abs/2403.01537).

Video demonstrations of the algorithm deployed on an Unitree Go1 quadruped can be found at the [project website](https://sites.google.com/view/brne-crowdnav) and Katie's [portfolio post](https://katie-hughes.github.io/crowdnav/). You can also find more details on the project from our invited presentation at [NVIDIA GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-se63278/).

Code authors:

- [Muchen Sun](https://muchen-sun.com/) proposed the algorithm and implemented the core algorithm in Python (Numba) and PyTorch (deployed on an Nvidia Jetson AGX Orin).
- [Katie Hughes](https://katie-hughes.github.io/) developed the ROS social navigation framework (specified for the Unitree Go1 quadruped robot) and a C++ implementation of the core BRNE algorithm.  

To cite this work:

```
@article{sun2024mixed,
  title={Mixed-Strategy Nash Equilibrium for Crowd Navigation},
  author={Sun, Muchen and Baldini, Francesca and Hughes, Katie and Trautman, Peter and Murphey, Todd},
  journal={arXiv preprint arXiv:2403.01537},
  year={2024}
}
```

The structure of the repo is as follow: 

- [`tutorials`](tutorials) contains Python tutorials to illustrate the principle of the BRNE algorithm and intuitive PyTorch implementation for simple hallway navigation scenarios.
- [`brne_nav`](brne_nav) contains the fully integrated ROS social navigation framework. If you are interested in deploying BRNE, you should copy this directory as a ROS package.

*Contact: Please contact Muchen Sun* ([*muchen@u.northwestern.edu*](mailto:muchen@u.northwestern.edu)) *if you have any question*.
