"""

Defining Class of custom environment for V-Rep

@author: hussein
"""

# import vrep_env
# from vrep_env import vrep
import sys 
import os
sys.path.append(os.path.abspath("/home/hussein/Desktop/Multi-agent-path-planning/Reinforcement Learning"))
sys.path.append(os.path.abspath("/home/hussein/Desktop/Multi-agent-path-planning/Reinforcement Learning/vrep_env"))

import os
# vrep_scenes_path = os.environ['/home/hussein/Desktop/Multi-agent-path-planning/Reinforcement Learning/examples/scenes']

import torch
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float32
from vrep_env import sim
from data_collection_v3 import MinimalPublisher

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time

L = 0.0975 # Pioneer Robot Parameter
d = 0.109561 # Pioneer Robot Parameter

#L = 1 # Parameter of BubbleRob
#d = 0.5 # Parameter of BubbleRob
A = np.ones(6) - np.identity(6) # Adjancency Matrix fully connected case 6x6

ux = np.zeros((6,1)) # 6x1
uy = np.zeros((6,1)) # 6x1

" Connecting to V-Rep "
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19997,True,True,-500000,5) # Connect to CoppeliaSim
N_SCENES = 1500
scenes = np.hstack(( np.random.uniform(-2,2,size=(N_SCENES,2)), np.random.uniform(0,np.pi,size=(N_SCENES,1)), np.random.uniform(-2,2,(N_SCENES,2)), np.random.uniform(0,np.pi,size=(N_SCENES,1)) ))

def euler_from_quaternion(x, y, z, w):
        
     t3 = +2.0 * (w * z + x * y)
     t4 = +1.0 - 2.0 * (y * y + z * z)
     yaw_z = math.atan2(t3, t4)
     
     return yaw_z # in radians

def state_dot_from_state(current_state, current_action):
    state_dot = np.array([[0], [0], [0]])
    state_dot[0] = current_action[0] * math.cos(current_state[2])
    state_dot[1] = current_action[0] * math.sin(current_state[2])
    state_dot[2] = current_action[1]
    return state_dot
     
"""

Description:
    Consensus environment of 6 robots, where each episode they converge towards each other. DQN applied to robot 1 and rest are controlled with the consensus algorithm.
    
Source:
    This environment corresponds to V-Rep simulator, integrated with ROS to publish actions & subscribe to observations.

Observation:
    Type: Box(4) 
    Num     Observation               Min                     Max
    0       Mx                        -4.8                    4.8
    1       My                        -4.8                    4.8
    2       Phix                      -4.8                    4.8
    3       Phiy                      -4.8                    4.8

Actions: 
    Type: Discrete(4) 
    Num   Action 
    0     Move the robot upwards
    1     Move the robot downwards
    2     Move the robot to the left
    3     Move the robot to the right   
    
"""

class MinimalPublisherGym(MinimalPublisher):
    def __init__(self):
        #vrep_env.VrepEnv.__init__(self, server_addr, server_port, scene_path)
        super().__init__()
        self.publisher_l1 = self.create_publisher(Float32, '/leftMotorSpeedrobot1', 0) #Change according to topic in child script,String to Float32
        self.publisher_r1 = self.create_publisher(Float32, '/rightMotorSpeedrobot1',0) #Change according to topic in child script,String to Float32
        self.publisher_l2 = self.create_publisher(Float32, '/leftMotorSpeedrobot2', 0) #Change according to topic in child script,String to Float32
        self.publisher_r2 = self.create_publisher(Float32, '/rightMotorSpeedrobot2',0) #Change according to topic in child script,String to Float32
        self.publisher_l3 = self.create_publisher(Float32, '/leftMotorSpeedrobot3', 0) #Change according to topic in child script,String to Float32
        self.publisher_r3 = self.create_publisher(Float32, '/rightMotorSpeedrobot3',0) #Change according to topic in child script,String to Float32
        self.publisher_l4 = self.create_publisher(Float32, '/leftMotorSpeedrobot4', 0) #Change according to topic in child script,String to Float32
        self.publisher_r4 = self.create_publisher(Float32, '/rightMotorSpeedrobot4',0) #Change according to topic in child script,String to Float32
        self.publisher_l5 = self.create_publisher(Float32, '/leftMotorSpeedrobot5', 0) #Change according to topic in child script,String to Float32
        self.publisher_r5 = self.create_publisher(Float32, '/rightMotorSpeedrobot5',0) #Change according to topic in child script,String to Float32
        self.publisher_l6 = self.create_publisher(Float32, '/leftMotorSpeedrobot6', 0) #Change according to topic in child script,String to Float32
        self.publisher_r6 = self.create_publisher(Float32, '/rightMotorSpeedrobot6',0) #Change according to topic in child script,String to Float32              
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            0)
        
        " Timer Callback "
        timer_period = 0.03  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.LAST_UPDATED_TIME = time.time()
        
        " Parameters "
        self.t = 0 # Just to intialized Phix's and Phiy's
        self.scene = 0 # Nb of scene iteration
        
        " Initialize Phi's "
        self.Phix1 = 0 # 1x1
        self.Phiy1 = 0 # 1x1
        self.Phix2 = 0 # 1x1
        self.Phiy2 = 0 # 1x1
        self.Phix3 = 0 # 1x1
        self.Phiy3 = 0 # 1x1
        self.Phix4 = 0 # 1x1
        self.Phiy4 = 0 # 1x1
        self.Phix5 = 0 # 1x1
        self.Phiy5 = 0 # 1x1
        self.Phix6 = 0 # 1x1
        self.Phiy6 = 0 # 1x1        
        
        " Mobile Robot 1 Parameters "
        self.x1 = 0
        self.y1 = 0
        self.Theta1 = 0
        self.v1 = 0
        self.w1 = 0
        self.vL1 = 0
        self.vR1 = 0
        
        " Mobile Robot 2 Parameters "
        self.x2 = 0
        self.y2 = 0
        self.Theta2 = 0
        self.v2 = 0
        self.w2 = 0
        self.vL2 = 0 
        self.vR2 = 0
        
        " Mobile Robot 3 Parameters "
        self.x3 = 0
        self.y3 = 0
        self.Theta3 = 0
        self.v3 = 0
        self.w3 = 0
        self.vL3 = 0
        self.vR3 = 0                
        
        " Mobile Robot 4 Parameters "
        self.x4 = 0
        self.y4 = 0
        self.Theta4 = 0
        self.v4 = 0
        self.w4 = 0
        self.vL4 = 0
        self.vR4 = 0        
        
        " Mobile Robot 5 Parameters "
        self.x5 = 0
        self.y5 = 0
        self.Theta5 = 0
        self.v5 = 0
        self.w5 = 0
        self.vL5 = 0
        self.vR5 = 0
        
        " Mobile Robot 6 Parameters "
        self.x6 = 0
        self.y6 = 0
        self.Theta6 = 0
        self.v6 = 0
        self.w6 = 0
        self.vL6 = 0
        self.vR6 = 0
                        
            
    def listener_callback(self, msg):
        
        if msg.transforms[0].child_frame_id == 'robot1' :  
            self.x1 = msg.transforms[0].transform.translation.x
            self.y1 = msg.transforms[0].transform.translation.y
            self.xr1 = msg.transforms[0].transform.rotation.x
            self.yr1 = msg.transforms[0].transform.rotation.y
            self.zr1 = msg.transforms[0].transform.rotation.z
            self.wr1 = msg.transforms[0].transform.rotation.w
            self.Theta1 = euler_from_quaternion(self.xr1,self.yr1,self.zr1,self.wr1)
    
        if  msg.transforms[0].child_frame_id == 'robot2' :
            self.x2 = msg.transforms[0].transform.translation.x
            self.y2 = msg.transforms[0].transform.translation.y
            self.xr2 = msg.transforms[0].transform.rotation.x
            self.yr2 = msg.transforms[0].transform.rotation.y
            self.zr2 = msg.transforms[0].transform.rotation.z
            self.wr2 = msg.transforms[0].transform.rotation.w
            self.Theta2 = euler_from_quaternion(self.xr2,self.yr2,self.zr2,self.wr2)
        
        if  msg.transforms[0].child_frame_id == 'robot3' :
            
            self.x3 = msg.transforms[0].transform.translation.x
            self.y3 = msg.transforms[0].transform.translation.y
            self.xr3 = msg.transforms[0].transform.rotation.x
            self.yr3 = msg.transforms[0].transform.rotation.y
            self.zr3 = msg.transforms[0].transform.rotation.z
            self.wr3 = msg.transforms[0].transform.rotation.w
            self.Theta3 = euler_from_quaternion(self.xr3,self.yr3,self.zr3,self.wr3)
    
        if  msg.transforms[0].child_frame_id == 'robot4' :
            
            self.x4 = msg.transforms[0].transform.translation.x
            self.y4 = msg.transforms[0].transform.translation.y
            self.xr4 = msg.transforms[0].transform.rotation.x
            self.yr4 = msg.transforms[0].transform.rotation.y
            self.zr4 = msg.transforms[0].transform.rotation.z
            self.wr4 = msg.transforms[0].transform.rotation.w
            self.Theta4 = euler_from_quaternion(self.xr4,self.yr4,self.zr4,self.wr4)        
            
        if  msg.transforms[0].child_frame_id == 'robot5' :
            
            self.x5 = msg.transforms[0].transform.translation.x
            self.y5 = msg.transforms[0].transform.translation.y
            self.xr5 = msg.transforms[0].transform.rotation.x
            self.yr5 = msg.transforms[0].transform.rotation.y
            self.zr5 = msg.transforms[0].transform.rotation.z
            self.wr5 = msg.transforms[0].transform.rotation.w
            self.Theta5 = euler_from_quaternion(self.xr5,self.yr5,self.zr5,self.wr5)
            
        if  msg.transforms[0].child_frame_id == 'robot6' :
            
            self.x6 = msg.transforms[0].transform.translation.x
            self.y6 = msg.transforms[0].transform.translation.y
            self.xr6 = msg.transforms[0].transform.rotation.x
            self.yr6 = msg.transforms[0].transform.rotation.y
            self.zr6 = msg.transforms[0].transform.rotation.z
            self.wr6 = msg.transforms[0].transform.rotation.w
            self.Theta6 = euler_from_quaternion(self.xr6,self.yr6,self.zr6,self.wr6)

    def timer_callback(self):
                       
        " Publish Speed Commands to Robot 1 "
    
        msgl1 = Float32()    
        msgr1 = Float32()
        msgl1.data = self.VL1
        msgr1.data = self.VR1
        self.publisher_l1.publish(msgl1)
        self.publisher_r1.publish(msgr1)

        " Publish Speed Commands to Robot 2 "
        
        msgl2 = Float32()
        msgr2 = Float32()
        msgl2.data = self.VL2
        msgr2.data = self.VR2
        self.publisher_l2.publish(msgl2)
        self.publisher_r2.publish(msgr2)

        " Publish Speed Commands to Robot 3 "
        
        msgl3 = Float32()
        msgr3 = Float32()
        msgl3.data = self.VL3
        msgr3.data = self.VR3
        self.publisher_l3.publish(msgl3)
        self.publisher_r3.publish(msgr3)
        
        " Publish Speed Commands to Robot 4 "
    
        msgl4 = Float32()
        msgr4 = Float32()
        msgl4.data = self.VL4
        msgr4.data = self.VR4
        self.publisher_l4.publish(msgl4)
        self.publisher_r4.publish(msgr4)        
        
        " Publish Speed Commands to Robot 5 "
        
        msgl5 = Float32()
        msgr5 = Float32()
        msgl5.data = self.VL5
        msgr5.data = self.VR5
        self.publisher_l5.publish(msgl5)
        self.publisher_r5.publish(msgr5)        


        " Publish Speed Commands to Robot 6 "
        
        msgl6 = Float32()
        msgr6 = Float32()
        msgl6.data = self.VL6
        msgr6.data = self.VR6
        self.publisher_l6.publish(msgl6)
        self.publisher_r6.publish(msgr6)        

    def spin_once_gym(self):
        rclpy.spin_once(self)


class MobileRobotVrepEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
        }
    def __init__(self):
        self.mpg = MinimalPublisherGym()
        
        self.scene = 0 # Nb of scene iteration

        self.LAST_UPDATED_TIME = time.time()
        
        " Distance at which to fail the episode "
        self.distance_threshold = 2.2
                
        " Observation & Action Space "
        # Define Action Space
            
        self.action_space = spaces.MultiDiscrete([2,2])               
        
        # Define Observation Space
        high_observation = np.array([4.8,
                                     4.8,
                                     4.8,
                                     4.8],
                                    dtype=np.float32)
        
        self.observation_space = spaces.Box(-high_observation, -high_observation, dtype=np.float32)        
        
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        " Distance Threshold "
        self.distance = abs(self.mpg.x1 - self.mpg.x2) + abs(self.mpg.y1 - self.mpg.y2) + abs(self.mpg.x1 - self.mpg.x3) + abs(self.mpg.y1 - self.mpg.y3) + abs(self.mpg.x1 - self.mpg.x4) + abs(self.mpg.y1 - self.mpg.y4) + abs(self.mpg.x1 - self.mpg.x5) + abs(self.mpg.y1 - self.mpg.y5) + abs(self.mpg.x1 - self.mpg.x6) + abs(self.mpg.y1 - self.mpg.y6)
        #print(self.distance)
        #print(self.mpg.x1)
        
        " Use Adjacency Matrix to find Mxy and Phi's "                
        
        A = np.ones(6) - np.identity(6) # Adjancency Matrix

        self.X = np.array([ [self.mpg.x1], [self.mpg.x2], [self.mpg.x3], [self.mpg.x4], [self.mpg.x5], [self.mpg.x6]  ]) #6x1
        self.Y = np.array([ [self.mpg.y1], [self.mpg.y2], [self.mpg.y3], [self.mpg.y4], [self.mpg.y5], [self.mpg.y6]  ]) #6x1

        ux = np.zeros((6,1)) # 6x1
        uy = np.zeros((6,1)) # 6x1

            
        for i in range(1,7):
            for j in range(1,7):
                ux[i-1] += -(A[i-1][j-1])*(self.X[i-1]-self.X[j-1]) # 1x1 each
                uy[i-1] += -(A[i-1][j-1])*(self.Y[i-1]-self.Y[j-1]) # 1x1 each
    
        # Manage 4 directions (Up/Down/Left/Right)
        if action[0] == 0.0:
            self.mpg.v1 = -1.0
        elif action[0] == 1.0:
            self.mpg.v1 = 1.0
        elif action[1] == 0.0:
            self.mpg.w1 = -1.0
        elif action[1] == 1.0:
            self.mpg.w1 = 1.0


        print("---------- Action of Robot 1 -------------------", action)

        number_steps = 10
        delta_time = 0.1
        
        robot1_current_state = np.array([ [self.mpg.x1], [self.mpg.y1], [self.mpg.Theta1] ])
        robot2_current_state = np.array([ [self.mpg.x2], [self.mpg.y2], [self.mpg.Theta2] ])
        robot3_current_state = np.array([ [self.mpg.x3], [self.mpg.y3], [self.mpg.Theta3] ])
        robot4_current_state = np.array([ [self.mpg.x4], [self.mpg.y4], [self.mpg.Theta4] ])
        robot5_current_state = np.array([ [self.mpg.x5], [self.mpg.y5], [self.mpg.Theta5] ])
        robot6_current_state = np.array([ [self.mpg.x6], [self.mpg.y6], [self.mpg.Theta6] ])
        
        robot1_current_action = np.array([ [self.mpg.v1], [self.mpg.w1] ])
        robot2_current_action = np.array([[self.mpg.v2], [self.mpg.w2]])
        robot3_current_action = np.array([[self.mpg.v3], [self.mpg.w3]])
        robot4_current_action = np.array([[self.mpg.v4], [self.mpg.w4]])
        robot5_current_action = np.array([[self.mpg.v5], [self.mpg.w5]])
        robot6_current_action = np.array([[self.mpg.v6], [self.mpg.w6]])
        
        multi_robot_sensitivity_matrix = np.identity(18) 
        A_sensitibity_matrix = np.identity(18)
        dphi = np.identity(18)
        observability_gramian = np.zeros([18,18]) # 18x18 of 0's
        
        for i in range(1, number_steps):
            # best_action = something_computed_from Q-learning
            
            # Euler Integrator Implementation for Each Robot:
                
            robot1_new_state = robot1_current_state + delta_time*state_dot_from_state(robot1_current_state, robot1_current_action) 
            robot2_new_state = robot2_current_state + delta_time*state_dot_from_state(robot2_current_state, robot2_current_action)
            robot3_new_state = robot3_current_state + delta_time*state_dot_from_state(robot3_current_state, robot3_current_action)
            robot4_new_state = robot4_current_state + delta_time*state_dot_from_state(robot4_current_state, robot4_current_action)
            robot5_new_state = robot5_current_state + delta_time*state_dot_from_state(robot5_current_state, robot5_current_action)
            robot6_new_state = robot6_current_state + delta_time*state_dot_from_state(robot6_current_state, robot6_current_action)
            
            # Create the Gramiam:
            # A = [0 0 -us(1)*sin(qs(3)); 0 0 us(1)*cos(qs(3)); 0 0 0]
            
            A_robot1 = np.array([ [ 0, 0, -robot1_current_action[0]*math.sin(robot1_current_state[2]) ], [ 0, 0, robot1_current_action[0]*math.cos(robot1_current_state[2])], [ 0, 0, 0] ]) #3x3
            A_sensitibity_matrix[:3 , :3] = A_robot1 
            
            A_robot2 = np.array([ [ 0, 0, -robot2_current_action[0]*math.sin(robot2_current_state[2]) ], [ 0, 0, robot2_current_action[0]*math.cos(robot2_current_state[2])], [ 0, 0, 0] ]) #3x3
            A_sensitibity_matrix[3:6 , 3:6] = A_robot2 

            A_robot3 = np.array([ [ 0, 0, -robot3_current_action[0]*math.sin(robot3_current_state[2]) ], [ 0, 0, robot3_current_action[0]*math.cos(robot3_current_state[2])], [ 0, 0, 0] ]) #3x3
            A_sensitibity_matrix[6:9 , 6:9] = A_robot3 

            A_robot4 = np.array([ [ 0, 0, -robot4_current_action[0]*math.sin(robot4_current_state[2]) ], [ 0, 0, robot4_current_action[0]*math.cos(robot4_current_state[2])], [ 0, 0, 0] ]) #3x3
            A_sensitibity_matrix[9:12 , 9:12] = A_robot4 

            A_robot5 = np.array([ [ 0, 0, -robot5_current_action[0]*math.sin(robot5_current_state[2]) ], [ 0, 0, robot5_current_action[0]*math.cos(robot5_current_state[2])], [ 0, 0, 0] ]) #3x3
            A_sensitibity_matrix[12:15 , 12:15] = A_robot5 

            A_robot6 = np.array([ [ 0, 0, -robot6_current_action[0]*math.sin(robot6_current_state[2]) ], [ 0, 0, robot6_current_action[0]*math.cos(robot6_current_state[2])], [ 0, 0, 0] ]) #3x3
            A_sensitibity_matrix[15: , 15:] = A_robot6             
            
            dphi = A_sensitibity_matrix*multi_robot_sensitivity_matrix
            
            # Phi_new is the new sensitivity matrix
            phi_new = multi_robot_sensitivity_matrix + delta_time*dphi
            
            # dGo = Phi'*H'*inv(R)*H*Phi;
            # H = 1.0 # derivative of measurement model - to be assigned later row: nb of measuremants, column: 
            
            multi_robot_sensitivity_matrix = phi_new
             
            # H Matrix 15x18:
            H = np.array([ [(self.mpg.x1-self.mpg.x2), (self.mpg.y1-self.mpg.y2), 0, -(self.mpg.x1-self.mpg.x2), -(self.mpg.y1-self.mpg.y2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
                           [(self.mpg.x1-self.mpg.x3), (self.mpg.y1-self.mpg.y3), 0, 0, 0, 0, -(self.mpg.x1-self.mpg.x3), -(self.mpg.y1-self.mpg.y3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                           [(self.mpg.x1-self.mpg.x4), (self.mpg.y1-self.mpg.y4), 0, 0, 0, 0, 0, 0, 0, -(self.mpg.x1-self.mpg.x4), -(self.mpg.y1-self.mpg.y4), 0, 0, 0, 0, 0, 0, 0 ], 
                           [(self.mpg.x1-self.mpg.x5), (self.mpg.y1-self.mpg.y5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -(self.mpg.x1-self.mpg.x5), -(self.mpg.y1-self.mpg.y5), 0, 0, 0, 0 ],
                           [(self.mpg.x1-self.mpg.x6), (self.mpg.y1-self.mpg.y6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0                                                   ], 
                           [0, 0, 0, (self.mpg.x2-self.mpg.x3), (self.mpg.y2-self.mpg.y3), 0, -(self.mpg.x2-self.mpg.x3), -(self.mpg.y2-self.mpg.y3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                           [0, 0, 0, (self.mpg.x2-self.mpg.x4), (self.mpg.y2-self.mpg.y4), 0, 0, 0, 0, -(self.mpg.x2-self.mpg.x4), -(self.mpg.y2-self.mpg.y4), 0, 0, 0, 0, 0, 0, 0 ],                           
                           [0, 0, 0, (self.mpg.x2-self.mpg.x5), (self.mpg.y2-self.mpg.y5), 0, 0, 0, 0, 0, 0, 0, -(self.mpg.x2-self.mpg.x5), -(self.mpg.y2-self.mpg.y5), 0, 0, 0, 0 ],                           
                           [0, 0, 0, (self.mpg.x2-self.mpg.x6), (self.mpg.y2-self.mpg.y6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -(self.mpg.x2-self.mpg.x6), -(self.mpg.y2-self.mpg.y6), 0 ],
                           [0, 0, 0, 0, 0, 0, (self.mpg.x3-self.mpg.x4), (self.mpg.y3-self.mpg.y4), 0, -(self.mpg.x3-self.mpg.x4), -(self.mpg.y3-self.mpg.y4), 0, 0, 0, 0, 0, 0, 0 ],                           
                           [0, 0, 0, 0, 0, 0, (self.mpg.x3-self.mpg.x5), (self.mpg.y3-self.mpg.y5), 0, 0, 0, 0, -(self.mpg.x3-self.mpg.x5), -(self.mpg.y3-self.mpg.y5), 0, 0, 0, 0 ],
                           [0, 0, 0, 0, 0, 0, (self.mpg.x3-self.mpg.x6), (self.mpg.y3-self.mpg.y6), 0, 0, 0, 0, 0, 0, 0, -(self.mpg.x3-self.mpg.x6), -(self.mpg.y3-self.mpg.y6), 0 ],                           
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, (self.mpg.x4-self.mpg.x5), (self.mpg.y4-self.mpg.y5), 0, -(self.mpg.x4-self.mpg.x5), -(self.mpg.y4-self.mpg.y5), 0, 0, 0, 0 ],                           
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, (self.mpg.x4-self.mpg.x6), (self.mpg.y4-self.mpg.y6), 0, 0, 0, 0, -(self.mpg.x4-self.mpg.x6), -(self.mpg.y4-self.mpg.y6), 0 ],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (self.mpg.x5-self.mpg.x6), (self.mpg.y5-self.mpg.y6), 0, -(self.mpg.x5-self.mpg.x6), -(self.mpg.y5-self.mpg.y6), 0 ] ]) #18x18

            dGi = np.dot(multi_robot_sensitivity_matrix.transpose(), H.transpose())
            dGj = np.dot(H, multi_robot_sensitivity_matrix)

            dGo = np.dot(dGi, dGj)
            observability_gramian = observability_gramian + delta_time*dGo
            
            # Current state is assigned into new States:
            robot1_current_state = robot1_new_state
            robot2_current_state = robot2_new_state
            robot3_current_state = robot3_new_state
            robot4_current_state = robot4_new_state
            robot5_current_state = robot5_new_state
            robot6_current_state = robot6_new_state



        #print(" ---- ACTION ZERO ---:")
        #print(action[0])
        #print(" -------- V1 VALUE ------")
        #print(self.mpg.v1)

    
        #print(" ---- ACTION ONE ---:")
        #print(action[1])
        #print(" -------- W1 Value ------")
        #print(self.mpg.w1)
    
        u2 = np.array([ [float(ux[1])], [float(uy[1])] ]) # 2x1
        u3 = np.array([ [float(ux[2])], [float(uy[2])] ]) # 2x1
        u4 = np.array([ [float(ux[3])], [float(uy[3])] ]) # 2x1
        u5 = np.array([ [float(ux[4])], [float(uy[4])] ]) # 2x1
        u6 = np.array([ [float(ux[5])], [float(uy[5])] ]) # 2x1        
        
        
        " Calculate V1/W1, V2/W2, V3/W3, V4/W4, V5/W5, V6/W6 "
        
        
        S1 = np.array([[self.mpg.v1], [self.mpg.w1]]) #2x1
        # G1 = np.array([[1,0], [0,1/L]]) #2x2
        # R1 = np.array([[math.cos(self.Theta1),math.sin(self.Theta1)],[-math.sin(self.Theta1),math.cos(self.Theta1)]]) #2x2
        # S1 = np.dot(np.dot(G1, R1), u1) #2x1

        S2 = np.array([[self.mpg.v2], [self.mpg.w2]]) #2x1
        G2 = np.array([[1,0], [0,1/L]]) #2x2
        R2 = np.array([[math.cos(self.mpg.Theta2),math.sin(self.mpg.Theta2)],[-math.sin(self.mpg.Theta2),math.cos(self.mpg.Theta2)]]) #2x2
        S2 = np.dot(np.dot(G2, R2), u2) # 2x1

        S3 = np.array([[self.mpg.v3], [self.mpg.w3]]) #2x1
        G3 = np.array([[1,0], [0,1/L]]) #2x2
        R3 = np.array([[math.cos(self.mpg.Theta3),math.sin(self.mpg.Theta3)],[-math.sin(self.mpg.Theta3),math.cos(self.mpg.Theta3)]]) #2x2
        S3 = np.dot(np.dot(G3, R3), u3) #2x1        

        S4 = np.array([[self.mpg.v4], [self.mpg.w4]]) #2x1
        G4 = np.array([[1,0], [0,1/L]]) #2x2
        R4 = np.array([[math.cos(self.mpg.Theta4),math.sin(self.mpg.Theta4)],[-math.sin(self.mpg.Theta4),math.cos(self.mpg.Theta4)]]) #2x2
        S4 = np.dot(np.dot(G4, R4), u4) #2x1        

        S5 = np.array([[self.mpg.v5], [self.mpg.w5]]) #2x1
        G5 = np.array([[1,0], [0,1/L]]) #2x2
        R5 = np.array([[math.cos(self.mpg.Theta5),math.sin(self.mpg.Theta5)],[-math.sin(self.mpg.Theta5),math.cos(self.mpg.Theta5)]]) #2x2
        S5 = np.dot(np.dot(G5, R5), u5) #2x1

        S6 = np.array([[self.mpg.v6], [self.mpg.w6]]) #2x1
        G6 = np.array([[1,0], [0,1/L]]) #2x2
        R6 = np.array([[math.cos(self.mpg.Theta6),math.sin(self.mpg.Theta6)],[-math.sin(self.mpg.Theta6),math.cos(self.mpg.Theta6)]]) #2x2
        S6 = np.dot(np.dot(G6, R6), u6) #2x1        
                
        
        " Calculate VL1/VR1, VL2/VR2, VL3/VR3, VL4/VR4, VL5/VR5, VL6/VR6 "
    
        D = np.array([[1/2,1/2],[-1/(2*d),1/(2*d)]]) #2x2
        Di = np.linalg.inv(D) #2x2

        Speed_L1 = np.array([[self.mpg.vL1], [self.mpg.vR1]]) # Vector 2x1 for Speed of Robot 1
        Speed_L2 = np.array([[self.mpg.vL2], [self.mpg.vR2]]) # Vector 2x1 for Speed of Robot 2 
        Speed_L3 = np.array([[self.mpg.vL3], [self.mpg.vR3]]) # Vector 2x1 for Speed of Robot 3
        Speed_L4 = np.array([[self.mpg.vL4], [self.mpg.vR4]]) # Vector 2x1 for Speed of Robot 4
        Speed_L5 = np.array([[self.mpg.vL5], [self.mpg.vR5]]) # Vector 2x1 for Speed of Robot 5
        Speed_L6 = np.array([[self.mpg.vL6], [self.mpg.vR6]]) # Vector 2x1 for Speed of Robot 6

        M1 = np.array([[S1[0]],[S1[1]]]).reshape(2,1) #2x1
        M2 = np.array([[S2[0]],[S2[1]]]).reshape(2,1) #2x1
        M3 = np.array([[S3[0]],[S3[1]]]).reshape(2,1) #2x1
        M4 = np.array([[S4[0]],[S4[1]]]).reshape(2,1) #2x1
        M5 = np.array([[S5[0]],[S5[1]]]).reshape(2,1) #2x1
        M6 = np.array([[S6[0]],[S6[1]]]).reshape(2,1) #2x1

        Speed_L1 = np.dot(Di, M1) # 2x1 (VL1, VR1)
        Speed_L2 = np.dot(Di, M2) # 2x1 (VL2, VR2)
        Speed_L3 = np.dot(Di, M3) # 2x1 (VL3, VR3)
        Speed_L4 = np.dot(Di, M4) # 2x1 (VL4, VR4)
        Speed_L5 = np.dot(Di, M5) # 2x1 (VL5, VR5)
        Speed_L6 = np.dot(Di, M6) # 2x1 (VL6, VR6)
    
        self.mpg.VL1 = float(Speed_L1[0])
        self.mpg.VR1 = float(Speed_L1[1])
        self.mpg.VL2 = float(Speed_L2[0])
        self.mpg.VR2 = float(Speed_L2[1])
        self.mpg.VL3 = float(Speed_L3[0])
        self.mpg.VR3 = float(Speed_L3[1])
        self.mpg.VL4 = float(Speed_L4[0])
        self.mpg.VR4 = float(Speed_L4[1])
        self.mpg.VL5 = float(Speed_L5[0])
        self.mpg.VR5 = float(Speed_L5[1])        
        self.mpg.VL6 = float(Speed_L6[0])
        self.mpg.VR6 = float(Speed_L6[1])        
        
        
        Mx = np.zeros((6,1)) # 6x1
        My = np.zeros((6,1)) # 6x1
                
        for i in range(1,7):
            for j in range(1,7):
                Mx[i-1] += (A[i-1][j-1])*(self.X[j-1] - self.X[i-1]) # 1x1 each
                My[i-1] += (A[i-1][j-1])*(self.Y[j-1] - self.Y[i-1]) # 1x1 each
    
            
        Mx1 = float(Mx[0]) / 5 # 1x1
        My1 = float(My[0]) / 5 # 1x1
        
        Mx2 = float(Mx[1]) / 5 # 1x1
        My2 = float(My[1]) / 5 # 1x1        
    
        Mx3 = float(Mx[2]) / 5 # 1x1
        My3 = float(My[2]) / 5 # 1x1
        
        Mx4 = float(Mx[3]) / 5 # 1x1
        My4 = float(My[3]) / 5 # 1x1
        
        Mx5 = float(Mx[4]) / 5 # 1x1
        My5 = float(My[4]) / 5 # 1x1
        
        Mx6 = float(Mx[5]) / 5 # 1x1
        My6 = float(My[5]) / 5 # 1x1         
                
        self.mpg.Phix1 = ( Mx2 + Mx3 + Mx4 + Mx5 + Mx6 ) / 5 # 1x1
        self.mpg.Phiy1 = ( My2 + My3 + My4 + My5 + My6 ) / 5 # 1x1
        
        # self.Phix2 = ( Mx1 + Mx3 + Mx4 + Mx5 + Mx6 ) / 5 # 1x1
        # self.Phiy2 = ( My1 + My3 + My4 + My5 + My6 ) / 5 # 1x1
        
        # self.Phix3 = ( Mx1 + Mx2 + Mx4 + Mx5 + Mx6 ) / 5 # 1x1
        # self.Phiy3 = ( My1 + My2 + My4 + My5 + My6 ) / 5 # 1x1
        
        # self.Phix4 = ( Mx1 + Mx2 + Mx3 + Mx5 + Mx6 ) / 5 # 1x1
        # self.Phiy4 = ( My1 + My2 + My3 + My5 + My6 ) / 5 # 1x1
        
        # self.Phix5 = ( Mx1 + Mx2 + Mx3 + Mx4 + Mx6 ) / 5 # 1x1
        # self.Phiy5 = ( My1 + My2 + My3 + My4 + My6 ) / 5 # 1x1
        
        # self.Phix6 = ( Mx1 + Mx2 + Mx3 + Mx4 + Mx5 ) / 5 # 1x1
        # self.Phiy6 = ( My1 + My2 + My3 + My4 + My5 ) / 5 # 1x1         
        
        observation_DQN = torch.tensor(np.array([Mx1, My1, self.mpg.Phix1, self.mpg.Phiy1], dtype=np.double))
        
        CURRENT_TIME = time.time()
        DIFFERENT_TIME = CURRENT_TIME - self.LAST_UPDATED_TIME
        done = self.distance < self.distance_threshold or DIFFERENT_TIME>20
        done = bool(done)
        reward = np.trace(observability_gramian)
        print("Reward Value:", reward)
        self.mpg.spin_once_gym()
        
        return observation_DQN, reward, done, {}
    
    def initialize_timer(self):
       self.LAST_UPDATED_TIME = time.time()
       
    def reset(self):
        observation_DQN = np.array([0, 0, 0, 0])

        print(" ----------------- Episode ------------------------- ", self.scene)
        
        # Stop Simulation
        print("Stop Simulation")
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)  

        # Retrieve some handles:
            
        ErrLocM1,LocM1 =sim.simxGetObjectHandle(clientID, 'robot1', sim.simx_opmode_oneshot_wait)
    
        if (not ErrLocM1==sim.simx_return_ok):
            pass
        
        ErrLocM2,LocM2 =sim.simxGetObjectHandle(clientID, 'robot2#0', sim.simx_opmode_oneshot_wait)
    
        if (not ErrLocM2==sim.simx_return_ok):
            pass           

        ErrLoc1,Loc1 =sim.simxGetObjectPosition(clientID, LocM1, -1, sim.simx_opmode_oneshot_wait)
    
        if (not ErrLoc1==sim.simx_return_ok):
            pass            
    
        ErrLoc2,Loc2 =sim.simxGetObjectPosition(clientID, LocM2, -1, sim.simx_opmode_oneshot_wait)

        if (not ErrLoc2==sim.simx_return_ok):
            pass     

        ErrLocO1,OriRobo1 =sim.simxGetObjectOrientation(clientID,LocM1, -1, sim.simx_opmode_oneshot_wait)
    
        if (not ErrLocO1==sim.simx_return_ok):
            pass             
    
        ErrLocO2,OriRobo2 =sim.simxGetObjectOrientation(clientID,LocM2, -1, sim.simx_opmode_oneshot_wait)

        if (not ErrLocO2==sim.simx_return_ok):
            pass 
        
        # Take handles of the other robotos
        ErrLocM3,LocM3 =sim.simxGetObjectHandle(clientID, 'robot3#1', sim.simx_opmode_oneshot_wait)
        if (not ErrLocM3==sim.simx_return_ok):
            pass
        
        ErrLocM4,LocM4 =sim.simxGetObjectHandle(clientID, 'robot4#2', sim.simx_opmode_oneshot_wait)
        if (not ErrLocM4==sim.simx_return_ok):
            pass
        
        ErrLocM5,LocM5 = sim.simxGetObjectHandle(clientID, 'robot5#3', sim.simx_opmode_oneshot_wait)
        if (not ErrLocM5==sim.simx_return_ok):
            pass
        
        ErrLocM6,LocM6 = sim.simxGetObjectHandle(clientID, 'robot6#4', sim.simx_opmode_oneshot_wait)
        if (not ErrLocM6==sim.simx_return_ok):
            pass
        
        
        OriRobo1[2] = scenes[self.scene][2]
        OriRobo2[2] = scenes[self.scene][5]


        # Set Robot Orientation

        sim.simxSetObjectOrientation(clientID, LocM1, -1, OriRobo1, sim.simx_opmode_oneshot_wait) 
        sim.simxSetObjectOrientation(clientID, LocM2, -1, OriRobo2, sim.simx_opmode_oneshot_wait)


        Loc1[0] = scenes[self.scene][0]
        Loc2[0] = scenes[self.scene][3]


        Loc1[1] = scenes[self.scene][1]
        Loc2[1] = scenes[self.scene][4]
        
        # Check if robot1 is close to any other robot (except robot2)
        #for i in range(1,4):
        too_close_dist = 0.55
        ErrLoc3,Loc3 =sim.simxGetObjectPosition(clientID, LocM3, -1, sim.simx_opmode_oneshot_wait)  
        if (not ErrLocM3==sim.simx_return_ok):
            pass 
        ErrLoc4,Loc4 =sim.simxGetObjectPosition(clientID, LocM4, -1, sim.simx_opmode_oneshot_wait)  
        if (not ErrLocM4==sim.simx_return_ok):
            pass 
        ErrLoc5,Loc5 =sim.simxGetObjectPosition(clientID, LocM5, -1, sim.simx_opmode_oneshot_wait)  
        if (not ErrLocM5==sim.simx_return_ok):
            pass 
        ErrLoc6,Loc6 =sim.simxGetObjectPosition(clientID, LocM6, -1, sim.simx_opmode_oneshot_wait)  
        if (not ErrLocM6==sim.simx_return_ok):
            pass 
        exit_cond = 0
        #print(" -------- ELEMENT 1 ------------")
        #print(Loc1)
        #print(" --------- ELEMENT 2 -----------")
        #print(Loc3)
        while exit_cond == 0:
            #print(Loc1)
            distr13 = np.sqrt(pow(Loc1[0] - Loc3[0],2) + pow(Loc1[1] - Loc3[1],2))
            distr14 = np.sqrt(pow(Loc1[0] - Loc4[0],2) + pow(Loc1[1] - Loc4[1],2))
            distr15 = np.sqrt(pow(Loc1[0] - Loc5[0],2) + pow(Loc1[1] - Loc5[1],2))
            distr16 = np.sqrt(pow(Loc1[0] - Loc6[0],2) + pow(Loc1[1] - Loc6[1],2))
            if (distr13 < too_close_dist) or (distr14 < too_close_dist) or (distr15 < too_close_dist) or (distr16 <too_close_dist):
                # TOO close. Select another starting point for robot 1
                Loc1z = Loc1[2]
                Loc1 = np.random.uniform(-2,2,size=(1,3))[0]
                Loc1[2] = Loc1z
                print(" Robot 1 is close!")
            else:
                exit_cond = 1
                
        # In Loc1 there is here a valid inizialization of robot1
        scenes[self.scene][0] = Loc1[0]
        scenes[self.scene][1] = Loc1[1]
        
        # Check if the robot2 is too close to other robots
        exit_cond = 0
        while exit_cond == 0:
            dist21 = np.sqrt(pow(Loc2[0] - Loc1[0],2) + pow(Loc2[1] - Loc1[1],2))
            dist23 = np.sqrt(pow(Loc2[0] - Loc3[0],2) + pow(Loc2[1] - Loc3[1],2))
            dist24 = np.sqrt(pow(Loc2[0] - Loc4[0],2) + pow(Loc2[1] - Loc4[1],2))
            dist25 = np.sqrt(pow(Loc2[0] - Loc5[0],2) + pow(Loc2[1] - Loc5[1],2))
            dist26 = np.sqrt(pow(Loc2[0] - Loc6[0],2) + pow(Loc2[1] - Loc6[1],2))
            
            if dist21 < too_close_dist or dist23 < too_close_dist or dist24 < too_close_dist or dist25 < too_close_dist or dist26 < too_close_dist:
                # TOO close. Select another starting point for robot 2
                Loc2z = Loc2[2]
                Loc2 = np.random.uniform(-2,2,size=(1,3))[0]
                Loc2[2] = Loc2z
                print(" Robot 2 is close!")
            else:
                exit_cond = 1
                
        # In Loc2 there is here a valid inizialization of robot2
        scenes[self.scene][3] = Loc2[0]
        scenes[self.scene][4] = Loc2[1]
        
        # Set Robot Position

        sim.simxSetObjectPosition(clientID, LocM1, -1, Loc1, sim.simx_opmode_oneshot)
        sim.simxSetObjectPosition(clientID, LocM2, -1, Loc2, sim.simx_opmode_oneshot)

        # Print Positions and Orientation
        
        #print("Robot1 Position:", Loc1)
        #print("Robot2 Position:", Loc2)
    
        #print("Robot1 Orientation:", OriRobo1)
        #print("Robot2 Orientation:", OriRobo2)
                
        # Nb of Scene Counter
        self.scene += 1
            
        # Start Simulation
        print("Start New Simulation:")
        sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)            
        
        " Use Adjacency Matrix to find Mxy and Phi's "                
        
        A = np.ones(6) - np.identity(6) # Adjancency Matrix
    
        self.X = np.array([ [self.mpg.x1], [self.mpg.x2], [self.mpg.x3], [self.mpg.x4], [self.mpg.x5], [self.mpg.x6]  ]) #6x1
        self.Y = np.array([ [self.mpg.y1], [self.mpg.y2], [self.mpg.y3], [self.mpg.y4], [self.mpg.y5], [self.mpg.y6]  ]) #6x1        
        
        Mx = np.zeros((6,1)) # 6x1
        My = np.zeros((6,1)) # 6x1
                
        for i in range(1,7):
            for j in range(1,7):
                Mx[i-1] += (A[i-1][j-1])*(self.X[j-1] - self.X[i-1]) # 1x1 each
                My[i-1] += (A[i-1][j-1])*(self.Y[j-1] - self.Y[i-1]) # 1x1 each
    
        Mx1 = float(Mx[0]) / 5 # 1x1
        My1 = float(My[0]) / 5 # 1x1
        
        Mx2 = float(Mx[1]) / 5 # 1x1
        My2 = float(My[1]) / 5 # 1x1        
    
        Mx3 = float(Mx[2]) / 5 # 1x1
        My3 = float(My[2]) / 5 # 1x1
        
        Mx4 = float(Mx[3]) / 5 # 1x1
        My4 = float(My[3]) / 5 # 1x1
        
        Mx5 = float(Mx[4]) / 5 # 1x1
        My5 = float(My[4]) / 5 # 1x1
        
        Mx6 = float(Mx[5]) / 5 # 1x1
        My6 = float(My[5]) / 5 # 1x1         
        
        
        self.mpg.Phix1 = ( Mx2 + Mx3 + Mx4 + Mx5 + Mx6 ) / 5 # 1x1
        self.mpg.Phiy1 = ( My2 + My3 + My4 + My5 + My6 ) / 5 # 1x1
        
        # self.Phix2 = ( Mx1 + Mx3 + Mx4 + Mx5 + Mx6 ) / 5 # 1x1
        # self.Phiy2 = ( My1 + My3 + My4 + My5 + My6 ) / 5 # 1x1
        
        # self.Phix3 = ( Mx1 + Mx2 + Mx4 + Mx5 + Mx6 ) / 5 # 1x1
        # self.Phiy3 = ( My1 + My2 + My4 + My5 + My6 ) / 5 # 1x1
        
        # self.Phix4 = ( Mx1 + Mx2 + Mx3 + Mx5 + Mx6 ) / 5 # 1x1
        # self.Phiy4 = ( My1 + My2 + My3 + My5 + My6 ) / 5 # 1x1
        
        # self.Phix5 = ( Mx1 + Mx2 + Mx3 + Mx4 + Mx6 ) / 5 # 1x1
        # self.Phiy5 = ( My1 + My2 + My3 + My4 + My6 ) / 5 # 1x1
        
        # self.Phix6 = ( Mx1 + Mx2 + Mx3 + Mx4 + Mx5 ) / 5 # 1x1
        # self.Phiy6 = ( My1 + My2 + My3 + My4 + My5 ) / 5 # 1x1          
        
        observation_DQN = torch.tensor(np.array([Mx1, My1, self.mpg.Phix1, self.mpg.Phiy1], dtype=np.double))
                                    
        return observation_DQN
    
    def render(self):
        pass
    
    
# def main(args=None):
#     rclpy.init(args=args)
#     minimal_publisher = MinimalPublisherGym()
#     time.sleep(5)
#     rclpy.spin(minimal_publisher)
#     minimal_publisher.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()    