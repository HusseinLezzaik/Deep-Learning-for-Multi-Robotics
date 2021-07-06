# Graph Neural Nets for Two Robots
Graph Neural Net for two robots for consensus.

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

## Consensus Algorithm
For our first experiment, we collected two dataset's for each corresponding robot. We later will use this data to train our first graph neural networks architecture on, 
which takes the relative position of each robot i w.r.t robot j in the local transformation frame and it's corresponding control input Uj as an output. Value's are saved within two seperate 
CSV files for each robot. 

We first run a vanilla consensus algorithm, and start collecting data of relative poses for robot i w.r.t robot j and it's corresponding local input for each robot.