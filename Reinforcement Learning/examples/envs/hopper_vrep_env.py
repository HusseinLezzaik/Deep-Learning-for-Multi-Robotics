
from vrep_env import vrep_env

import os
vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class HopperVrepEnv(vrep_env.VrepEnv):
	metadata = {'render.modes': [],}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=-19997,
		scene_path=vrep_scenes_path+'/hopper.ttt',
	):
		vrep_env.VrepEnv.__init__(
			self,
			server_addr,
			server_port,
			scene_path,
		)
		
		# Settings
		self.random_start = False
		
		# All joints
		joint_names = ['thigh_joint','leg_joint','foot_joint']
		# All shapes
		shape_names = ['torso','thigh','leg','foot']
		
		# Getting object handles
		
		# Meta
		self.camera = self.get_object_handle('camera')
		# Actuators
		self.oh_joint = list(map(self.get_object_handle, joint_names))
		# Shapes
		self.oh_shape = list(map(self.get_object_handle, shape_names))
		
		# One action per joint
		dim_act = len(self.oh_joint)
		# Multiple dimensions per shape
		dim_obs = (len(self.oh_shape)*3*2)+1
		
		high_act =        np.ones([dim_act])
		high_obs = np.inf*np.ones([dim_obs])
		
		self.action_space      = gym.spaces.Box(-high_act, high_act)
		self.observation_space = gym.spaces.Box(-high_obs, high_obs)
		
		# Parameters
		self.joints_max_velocity = 8.0
		#self.power = 0.75
		self.power = 3.75
		
		self.seed()
		
		print('HopperVrepEnv: initialized')
	
	def _make_observation(self):
		"""Get observation from v-rep and stores in self.observation
		"""
		lst_o = []
		
		# Include z position in observation
		torso_pos = self.obj_get_position(self.oh_shape[0])
		lst_o += [torso_pos[2]]
		
		# Include shapes relative velocities in observation
		for i_oh in self.oh_shape:
			lin_vel , ang_vel = self.obj_get_velocity(i_oh)
			lst_o += ang_vel
			lst_o += lin_vel
		
		self.observation = np.array(lst_o).astype('float32')
	
	def _make_action(self, a):
		"""Send action to v-rep
		"""
		for i_oh, i_a in zip(self.oh_joint, a):
			#self.obj_set_velocity(i_oh, i_a)
			self.obj_set_velocity(i_oh, self.power*float(np.clip(i_a,-1,+1)))
	
	def step(self, action):
		# Clip xor Assert
		#actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		
		# Actuate
		self._make_action(action)
		#self._make_action(action*self.joints_max_velocity)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		torso_pos_z  = self.observation[0] # up/down
		torso_lvel_x = self.observation[4]
		r_alive = 1.0
		
		reward = (16.0)*(r_alive) +(8.0)*(torso_lvel_x)
		
		# Early stop
		stand_threshold = 0.10
		done = (torso_pos_z < stand_threshold)
		#done = False
		
		return self.observation, reward, done, {}
	
	def reset(self):
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		
		# First action is random: emulate random initialization
		if self.random_start:
			factor = self.np_random.uniform(low=0, high=0.02, size=(1,))[0]
			action = self.action_space.sample()*factor
			self._make_action(action)
			self.step_simulation()
		
		self._make_observation()
		return self.observation
	
	def render(self, mode='human', close=False):
		pass
	
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
def main(args):
	env = HopperVrepEnv()
	for i_episode in range(4):
		observation = env.reset()
		print(observation)
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
