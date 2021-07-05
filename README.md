# Graph NNs and RL for Multi-Robot Motion Planning

This repository contains the code and models necessary to replicate the results of our work:

<p>
<img src="consensus_graph.PNG" width="1000" >
</p>

The main idea of our work is to develop a machine learning model powered by Graph Neural Networks and Reinforcement Learning to build a multi-agent path planning algorithm that generalizes to different network topologies, while mainting fast communication and efficient convergence.

## Overview of the Repository

The major content of our repo are:

* `Two Robots Topology Graph`: code for building our GNN model for two robots.
* `Real Topology Graph`: code for building our scalable robust GNN model.
* `Reinforcement Learning`: code for building our custom gym environment & DQN model powered by GNN.

Note: please check the `README` of each repository to dive deeper into the code and be able to replicate our results.

## Getting Started
Our code relies on using [CoppeliaSim](https://www.coppeliarobotics.com/)  for Simulating our experiments on robots, and [ROS2 Foxy](https://docs.ros.org/en/foxy/index.html) for publishing commands to our robots. 

1.  Clone our repo: `git clone https://github.com/HusseinLezzaik/Multi-agent-path-planning.git`

2.  Install dependencies:
    ```
    conda create -n multi-agent python=3.8
    conda activate multi-agent
    pip install -r requirements.txt
    ```
    
3. Install [CoppeliaSim edu](https://www.coppeliarobotics.com/downloads).

4. Install [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation.html) and make sure that the [ROS2 Interface](https://www.coppeliarobotics.com/helpFiles/en/ros2Interface.htm) works.

5. Make sure the [Python Remote API](https://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm) works.

6. Install [Gym Environment](https://gym.openai.com/docs/) from OpenAI in order to use our custom environment for RL training.


## Simulation in V-Rep 
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
We would like to thank [Claudio Pacchierotti](https://team.inria.fr/rainbow/team/claudio-pacchierotti/) for his constructive comments and discussions.

## Contact
If you have any question, or if anything of the above is not working, don't hestitate to contact us! We are more than happy to help!
* [Hussein Lezzaik](https://www.husseinlezzaik.com/) (hussein dot lezzaik at irisa dot fr)
* [Gennaro Notomista](https://www.gnotomista.com/) (gennaro dot notomista at irisa dot fr)
