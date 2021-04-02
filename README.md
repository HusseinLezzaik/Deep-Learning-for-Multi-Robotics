# Multi agent Path Planning
This code contains my 6-month work as a Machine Learning intern at France's National Research Institute in Computer Science @Inria.

The main idea of my work is to develope a machine learning model powered by Graph Neural Networks and merged with Deep Reinforcement Learning algorithms 
to build a multi-agent path planning algorithm that generalizes to different network topologies, while mainting fast communication and efficient convergence.

We use CoppeliaSim as our simulator to see the performance of our algorithms on mobile robots.

All code is written in Python3, and we use Ros2-Interface to communicate with CoppeliaSim.

 
## Pre-requisites 
Please make sure to have the following installed before using the main.py code:
* NumPy 
* Pandas
* PyTorch
* ROS2 
* ROS2-Interface
* CoppeliaSim 
* Remote Python API


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

