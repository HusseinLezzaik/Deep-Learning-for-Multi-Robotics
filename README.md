# Multi Agent Path Planning using GNN & DRL

This repository contains the code and models necessary to replicate the results of our work:

The main idea of our work is to develope a machine learning model powered by Graph Neural Networks and merged with Deep Reinforcement Learning algorithms 
to build a multi-agent path planning algorithm that generalizes to different network topologies, while mainting fast communication and efficient convergence.

## Overview of the Repository

The major content of our repo are:

* GNN Model contains the code for our experiments and training.
* Data Collection contains the code for collecting data from our experiments.

## Getting started
* Our code relies on using [CoppeliaSim](https://www.coppeliarobotics.com/)  for Simulating our experiments on robots, and [ROS2 Foxy](https://docs.ros.org/en/foxy/index.html) for publishing commands to our robots. *

1.  Clone our repo: `git clone https://github.com/HusseinLezzaik/Multi-agent-path-planning.git`

2.  Install dependencies:
    ```
    conda create -n multi-agent python=3.7
    conda activate multi-agent
    pip install -r requirements.txt
    ```
    
3. Install [CoppeliaSim edu](https://www.coppeliarobotics.com/downloads).

4. Install [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation.html) and make sure that the [ROS2 Interface](https://www.coppeliarobotics.com/helpFiles/en/ros2Interface.htm) works.

5. Make sure the [Python Remote API](https://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm) works.


## Simulation in CoppeliaSim 
We test our algorithms on two bubblerob's from CoppeliaSim, however our work applies to all kinds of mobile robots that just need some initial parameter tuning.
We first run a vanilla consensus algorithm, and start collecting data of relative poses for robot i w.r.t robot j and it's corresponding local input for each robot.

## Training data
For our first experiment, we collected two dataset's for each corresponding robot. We later will use this data to train our first graph neural networks architecture on, 
which takes the relative position of each robot i w.r.t robot j in the local transformation frame and it's corresponding control input Uj as an output. Value's are saved within two seperate 
CSV files for each robot. 

### Data Collection 
Using Python's remote API for CoppeliaSim, we initialized the positions randomly for each scene and ran the consensus algorithm to collect new data. Our dataset size is about 700 samples for each robot per scene,
and we stop collecting data for d=0.2 ie when they meet.

## Acknowledgement
We would like to thank Claudio Pacchierotti for comments and discussions.

## Contact
If you have any question, or if anything of the above is not working, don't hestitate to contact us! We are more than happy to help!
* Hussein Lezzaik (hussein dot lezzaik at irisa dot fr)
* Gennaro Notomista (gennaro dot notomista at irisa dot fr)
