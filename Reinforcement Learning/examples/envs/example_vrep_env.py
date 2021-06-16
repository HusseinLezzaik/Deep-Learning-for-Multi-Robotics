
# This file is a template for V-rep environments
#     all names in this file are just examples
# Search for '#modify' and replace accordingly

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent

import os
vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces

import numpy as np

# #modify: the env class name
class ExampleVrepEnv(vrep_env.VrepEnv):
	metadata = {
		'render.modes': [],
	}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		# #modify: the filename of your v-rep scene
		scene_path=vrep_scenes_path+'/example.ttt'
	):
		
		vrep_env.VrepEnv.__init__(self,server_addr,server_port,scene_path)
		# #modify: the name of the joints to be used in action space
		joint_names = [
			'example_joint_0',
			'example_left_joint_0','example_right_joint_0',
			'example_joint_1',
			'example_left_joint_1','example_right_joint_1',
		]
		# #modify: the name of the shapes to be used in observation space
		shape_names = [
			'example_head',
			'example_left_arm','example_right_arm',
			'example_torso',
			'example_left_leg','example_right_leg',
		]
		
		# Getting object handles
		
		# we will store V-rep object handles (oh = object handle)
		
		# Meta
		# #modify: if you want additional object handles
		self.camera = self.get_object_handle('camera')
		
		# Actuators
		self.oh_joint = list(map(self.get_object_handle, joint_names))
		# Shapes
		self.oh_shape = list(map(self.get_object_handle, shape_names))
		
		
		# #modify: if size of action space is different than number of joints
		# Example: One action per joint
		num_act = len(self.oh_joint)
		
		# #modify: if size of observation space is different than number of joints
		# Example: 3 dimensions of linear and angular (2) velocities + 6 additional dimension
		num_obs = (len(self.oh_shape)*3*2) + 3*2
		
		# #modify: action_space and observation_space to suit your needs
		self.joints_max_velocity = 3.0
		act = np.array( [self.joints_max_velocity] * num_act )
		obs = np.array(          [np.inf]          * num_obs )
		
		self.action_space      = spaces.Box(-act,act)
		self.observation_space = spaces.Box(-obs,obs)
		
		# #modify: optional message
		print('ExampleVrepEnv: initialized')
	
	def _make_observation(self):
		"""Query V-rep to make observation.
		   The observation is stored in self.observation
		"""
		# start with empty list
		lst_o = [];
		
		# #modify: optionally include positions or velocities
		pos               = self.obj_get_position(self.oh_shape[0])
		lin_vel , ang_vel = self.obj_get_velocity(self.oh_shape[0])
		lst_o += pos
		lst_o += lin_vel
		
		# #modify
		# example: include position, linear and angular velocities of all shapes
		for i_oh in self.oh_shape:
			rel_pos = self.obj_get_position(i_oh, relative_to=vrep.sim_handle_parent)
			lst_o += rel_pos ;
			lin_vel , ang_vel = self.obj_get_velocity(i_oh)
			lst_o += ang_vel ;
			lst_o += lin_vel ;
		
		self.observation = np.array(lst_o).astype('float32');
	
	def _make_action(self, a):
		"""Query V-rep to make action.
		   no return value
		"""
		# #modify
		# example: set a velocity for each joint
		for i_oh, i_a in zip(self.oh_joint, a):
			self.obj_set_velocity(i_oh, i_a)
	
	def step(self, action):
		"""Gym environment 'step'
		"""
		# #modify Either clip the actions outside the space or assert the space contains them
		# actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
		assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
		
		# Actuate
		self._make_action(action)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		# #modify the reward computation
		# example: possible variables used in reward
		head_pos_x = self.observation[0] # front/back
		head_pos_y = self.observation[1] # left/right
		head_pos_z = self.observation[2] # up/down
		nrm_action  = np.linalg.norm(actions)
		r_regul     = -(nrm_action**2)
		r_alive = 1.0
		# example: different weights in reward
		reward = (8.0)*(r_alive) +(4.0)*(head_pos_x) +(1.0)*(head_pos_z)
		
		# Early stop
		# #modify if the episode should end earlier
		tolerable_threshold = 0.20
		done = (head_pos_z < tolerable_threshold)
		#done = False
		
		return self.observation, reward, done, {}
	
	def reset(self):
		"""Gym environment 'reset'
		"""
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		self._make_observation()
		return self.observation
	
	def render(self, mode='human', close=False):
		"""Gym environment 'render'
		"""
		pass
	
	def seed(self, seed=None):
		"""Gym environment 'seed'
		"""
		return []
	
def main(args):
	"""main function used as test and example.
	   Agent does random actions with 'action_space.sample()'
	"""
	# #modify: the env class name
	env = ExampleVrepEnv()
	for i_episode in range(8):
		observation = env.reset()
		total_reward = 0
		for t in range(256):
			action = env.action_space.sample()
			observation, reward, done, _ = env.step(action)
			total_reward += reward
			if done:
				break
		print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
	env.close()
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
