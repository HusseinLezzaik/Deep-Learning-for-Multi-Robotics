"""

Code for training DQN using MLP as the NN, integrated with ROS2 and V-Rep as environment. Custom environment for training is "mobile_robot_env_gym.py"

"""
# Source code from the Q-Learning documentation page on PyTorch

import sys 
import os
sys.path.append(os.path.abspath("/home/hussein/Desktop/Deep-Learning-for-Multi-Robotics/Reinforcement Learning"))
sys.path.append(os.path.abspath("/home/hussein/Desktop/Deep-Learning-for-Multi-Robotics/Reinforcement Learning/vrep_env"))
sys.path.append(os.path.abspath("/home/hussein/Desktop/Deep-Learning-for-Multi-Robotics/Reinforcement Learning/examples"))
sys.path.append(os.path.abspath("/home/hussein/Desktop/Deep-Learning-for-Multi-Robotics/Reinforcement Learning/examples/envs"))

import rclpy
rclpy.init()

import gym
import examples
from envs.mobile_robot_env_gym_multirobots import MobileRobotVrepEnv
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from decimal import Decimal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn import Module
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn import ReLU

env = MobileRobotVrepEnv()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(Module):
    # define model elements
    def __init__(self):
        super(DQN, self).__init__()
                
        " Model A_1 of Mxy "
        # Inputs to hidden layer linear transformation
        self.inputA_1 = Linear(2, 12) # 24 inputs, 3 hidden units
        xavier_uniform_(self.inputA_1.weight)
        self.actA1_1 = ReLU()
        # Define Hidden Layer
        self.hiddenA_1 = Linear(12, 12)
        xavier_uniform_(self.hiddenA_1.weight)
        self.actA2_1 = ReLU() 
        # Output Layer 3 to 2 units
        self.outputA_1 = Linear(12, 2)
        xavier_uniform_(self.outputA_1.weight)        
        
        " Model B_1 " 
        # Inputs to hidden layer linear transformation
        self.inputB_1 = Linear(2, 12) # 2 inputs, 3 hidden units
        xavier_uniform_(self.inputB_1.weight)
        self.actB1_1 = ReLU()
        # Define Hidden Layer
        self.hiddenB_1 = Linear(12, 12)
        xavier_uniform_(self.hiddenB_1.weight)
        self.actB2_1 = ReLU() 
        # Output layer 3 to 2 units
        self.outputB_1 = Linear(12, 2)
        xavier_uniform_(self.outputB_1.weight)        
        
        " Model E_1 Merged "        
        # Define 4x3 hidden unit
        self.inputE_1 = Linear(4,3)
        xavier_uniform_(self.inputE_1.weight)
        self.actE1_1 = ReLU()
        # Define Output 3x12 unit        
        self.outputE_1 = Linear(3,2)
        xavier_uniform_(self.outputE_1.weight)

    # forward propagate input
    def forward(self, inputs):
        
        M = inputs[:2]
        Phi = inputs[2:4]
        
        " Model A "
        # Input to first hidden layer
        X1_1 = self.inputA_1(M)
        X1_1 = self.actA1_1(X1_1)
        # Second hidden layer
        X1_1 = self.hiddenA_1(X1_1)
        X1_1 = self.actA2_1(X1_1)
        # Final hidden layer and Output
        X1_1 = self.outputA_1(X1_1)        

        " Model B "
        # Input to first hidden layer
        X2_1 = self.inputB_1(Phi)
        X2_1 = self.actB1_1(X2_1)
        # Second hidden layer
        X2_1 = self.hiddenB_1(X2_1)
        X2_1 = self.actB2_1(X2_1)
        # Final hidden layer and Output
        X2_1 = self.outputB_1(X2_1)        
        
        " Model E "
        # Combine Models
        X = torch.cat((X1_1, X2_1))
        # Define Hidden Layer
        X = self.inputE_1(X)
        X = self.actE1_1(X)
        # Output Layer
        X = self.outputE_1(X)
        
        if X[0]<0:
            X[0]=0.0
        else:
            X[0]=+1.0
    
        if X[1]<0:
            X[1]=0.0
        else:
            X[1]=+1.0   
                        
        return X

env.reset()


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 1

policy_net = DQN().to(device).double()
target_net = DQN().to(device).double()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

episode_durations = []

        
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    state_action_values = policy_net(state_batch).gather(0, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device).double()
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(0)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
num_episodes = 5
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.initialize_timer()
    state = env.reset()
    for t in count():
        # Select and perform an action
        action = policy_net(state.double())
        
        action_np = action.detach().numpy().astype(np.int64)
        next_state, reward, done, _ = env.step(action_np)
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory        
        action = torch.tensor(action, device=device, dtype=torch.long)
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(list(target_net.parameters()))
        # for param in target_net.parameters():
        #     print(param)

FILE = "model.pth"
torch.save(target_net.state_dict(), FILE)
print('Training for Localization is Complete')
env.render()
env.close()
plt.ioff()
plt.show()