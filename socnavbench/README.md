# BRNE implementation for [SocNavBench](https://github.com/CMU-TBD/SocNavBench)

*Author: Max Muchen Sun*

Social Navigation Benchmark utility (SocNavBench) is a codebase for benchmarking robot planning algorithms against various episodes of containing multi-agent environments, developed at Carneige Mellon University. More information regarding the benchmark framework can be found [here](https://github.com/CMU-TBD/SocNavBench). 

Below are the implementations of BRNE for the SocNavBench environment and how to install them. The implementations include the BRNE algorihtm (`brne`) and the constant velocity baseline (`cvm`). 

To install: 

 1. Install SocNavBench following the [instructions](https://github.com/CMU-TBD/SocNavBench/blob/master/docs/install.md).
 2. Copy the file [`joystick_client.py` ](joystick_client.py) into the directory `(root dir of SovNavBench)/joystick`.
 3. Copy the files [`brne.py`](brne.py), [`joystick_brne.py`](joystick_brne.py), and [`joystick_cvm.py`](joystick_cvm.py) into the directory `(root dir of SovNavBench)/joystick/joystick_py`.

To use:

 1. Follow the usage [instructions](https://github.com/CMU-TBD/SocNavBench/blob/master/docs/usage.md) of SovNavBench.
 2. Use the argument `--algo "brne"` or `--algo "cvm"` to run BRNE or the constant velocity baseline. 
