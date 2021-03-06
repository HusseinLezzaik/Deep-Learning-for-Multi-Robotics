# Custom Environment of gym for Vrep
This is an implementation of the Vrep Simulator as a gym environment. It can be used to make robots learn interacting with the physical environment.

<p>
<img src="consensus_graph1.PNG" width="1000" >
</p>

We built our gym environment from scratch by integrating it with V-Rep. And we used it to train our RL agent.

### Table of Content

- [Reinforcement Learning Setup](#Reinforcement%20Learning%20Setup)
- [CoppeliaSim Scene](#CoppeliaSim%20Scene)
- [Gym Environment](#Gym%20Environment)
- [Deep Q-Learning](#Deep%20Q-Learning)

## Reinforcement Learning Setup
Since our deep learning model is decentralized, it's sufficient to build 1 model and train it for one robot and use it for the rest. We used Robot1 as our RL agent,
and ran Deep Q-Learning to improve our GNN model. We controlled the rest of the robots using a vanilla consensus algorithm.

## CoppeliaSim Scene
In order to run our scene, please load `examples/scenes/Scene_of_Six_Robots.ttt`

## Gym Environment
To use our custom environment, check `examples/scenes/mobile_robot_env _gym.py`

## Deep Q-Learning
We used Deep Q-Learning algorithm to train the agent, and integrated the GNN model developed before as the deep neural network used in the architecture of  DQN. 
Run `DQN_MLP.py` for training the agent after installing the custom environment and all the dependencies.

## Acknowledgement

We used [Vrep-Env](https://github.com/ycps/vrep-env#vrepcartpole-v0) for building the skeleton of our custom environment.
