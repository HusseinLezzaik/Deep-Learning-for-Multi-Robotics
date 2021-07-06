# Graph Neural Nets for Two Robots
Graph Neural Net for two robots to perform consensus.

<p float="center">
  <img src="consensus_graph1.PNG" width="370" hspace="20"/>
  <img src="consensus_graph2.PNG" width="370" /> 
</p>

### Table of Content

- [Consensus Algorithm](#Consensus%20Algorithm)
- [Data Collection](#Data%20Collection)
- [GNN Model](#GNN%20Model)
- [Pose Transformer](#Pose%20Transformer)
- [ROS2 Speed Publisher](#ROS2%20Speed%20Publisher)

For running our same robotic scene check `Consensus Algorithm/ros2_control_single_mobile_robot.ttt`

## Pose Transformer
Code for transforming control inputs of a Uni-Cycle to a two wheeled differential drive.

## ROS Speed Publisher
Code for creating a ROS node to publish speeds to CoppeliaSim scene.

## Consensus Algorithm

* `simConst.py`, `sim.py`, `simpleTest.py`, `remoteApi.so`: file dependency for Python remote API.
* `robot1.csv`, `robot2.csv`: csv files for storing relative poses and control inputs.
* `main_consensus_algorithm.py`: consensus algorithm and data saving.


We first run a vanilla consensus algorithm, and start collecting data of relative poses for robot i w.r.t robot j and it's corresponding local input for each robot.

## Data Collection
For our first experiment, we collected two dataset's for each corresponding robot. We later will use this data to train our first graph neural networks architecture on, 
which takes the relative position of each robot i w.r.t robot j in the local transformation frame and it's corresponding control input Uj as an output. Value's are saved within two seperate 
CSV files for each robot. 

## GNN Model

