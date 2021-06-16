"""
Adaptation from:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
Which is based in:
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

from vrep_env import vrep_env
from vrep_env import vrep

import os
vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class CartPoleContinuousVrepEnv(vrep_env.VrepEnv):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		scene_path=vrep_scenes_path+'/gym_cartpole.ttt',
	):
		vrep_env.VrepEnv.__init__(
			self,
			server_addr,
			server_port,
			scene_path,
		)
		
		# getting object handles
		self.action   = self.get_object_handle('action')
		self.cart     = self.get_object_handle('cart')
		self.pole     = self.get_object_handle('pole')
		self.viewer   = self.get_object_handle('viewer')
		
		# adjusting parameters
		self.tau = 0.02  # seconds between state updates
		self.gravity = 9.8
		#self.force_mag = 10.0
		self.force_mag = 100.0
		
		self.set_float_parameter(vrep.sim_floatparam_simulation_time_step, self.tau)
		self.set_array_parameter(vrep.sim_arrayparam_gravity,[0,0,-self.gravity])
		self.obj_set_force(self.action,self.force_mag)
		
		# Angle at which to fail the episode
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		self.x_threshold = 2.4
		# Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
		high = np.array([
			self.x_threshold * 2,             np.finfo(np.float32).max,
			self.theta_threshold_radians * 2, np.finfo(np.float32).max])
		
		self.min_action = -1.0
		self.max_action =  1.0
		
		self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
		self.observation_space = spaces.Box(-high, high)
		
		self.seed()
		self.viewer = None
		self.state = None
		self.steps_beyond_done = None
	
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
	def _make_observation(self):
		# discard y and z values
		[  x   ,_,_ ]         = self.obj_get_position(self.cart)
		[x_dot ,_,_ ] , _     = self.obj_get_velocity(self.cart)
		
		[_, theta ,_]         = self.obj_get_orientation(self.pole)
		_ , [_, theta_dot ,_] = self.obj_get_velocity(self.pole)
		
		self.state = (x,x_dot,theta,theta_dot)
	
	def _make_action(self, a):
		self.obj_set_velocity(self.action,a*2.0)
	
	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		
		# Actuate
		self._make_action(action)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		(x,x_dot,theta,theta_dot) = self.state
		
		done = x < -self.x_threshold or theta < -self.theta_threshold_radians \
			or x >  self.x_threshold or theta >  self.theta_threshold_radians
		done = bool(done)
		
		if not done:
			reward = 1.0
		elif self.steps_beyond_done is None:
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0
		
		return np.array(self.state), reward, done, {}
	
	def reset(self):
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		self.steps_beyond_done = None
		
		v = self.np_random.uniform(low=-0.04, high=0.04, size=(1,))
		self.obj_set_velocity(self.action,v)
		self.step_simulation()
		
		self._make_observation()
		return np.array(self.state)
	
	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400
		
		world_width = self.x_threshold*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * 1.0
		cartwidth = 50.0
		cartheight = 30.0
		
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)
		
		if self.state is None: return None
		
		x = self.state
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')
	
	def close(self):
		if self.viewer: self.viewer.close()
		vrep_env.VrepEnv.close(self)

def main(args):
	env = CartPoleContinuousVrepEnv()
	for i_episode in range(8):
		observation = env.reset()
		total_reward = 0
		for t in range(200):
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
