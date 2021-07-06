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

For running our same robotic scene use `Consensus Algorithm/ros2_control_single_mobile_robot.ttt`

## Pose Transformer
Code for transforming control inputs of a unicycle to a two wheeled differential drive.

## ROS Speed Publisher
Code for creating a ROS node to publish speeds to CoppeliaSim scene.

## Consensus Algorithm

* `simConst.py`, `sim.py`, `simpleTest.py`, `remoteApi.so`: file dependencies for Python remote API.
* `robot1.csv`, `robot2.csv`: csv files for storing relative poses and control inputs.
* `main_consensus_algorithm.py`: consensus algorithm and data saving.

We first run a vanilla consensus algorithm, and start collecting data of relative poses for robot i w.r.t robot j and it's corresponding local control or speed input.

## Data Collection
We collected labelled datasets for the following two models: 
1) Differential Drive
2) UniCycle.

* `initial_positions.py`: test code for changing initial positions in each scene.
* `main_consensus.py`: consensus algorithm for two robots with data saving.
* `main_differential_drive_api.py`: consensus algorithm API for saving relative poses and local speed inputs for differential drive.
* `main_unicycle_api.py`: consensus algorithm API for saving relative poses and local control inputs for unicycle.

Note: read comments within each module for more details.

## GNN Model
We developed two different GNN models for each case:
1) UniCycle:

* `dataset_robot1.csv`: cleaned labelled dataset for training GNN model.
* `load_model.py`: code for loading trained model using PyTorch_state_dict.
* `MLP_Model.py`: code for building and training GNN model in PyTorch.
* `Main_MLP.py`: code for running trained model in real-time in simulator. 


2) Differential Drive:
