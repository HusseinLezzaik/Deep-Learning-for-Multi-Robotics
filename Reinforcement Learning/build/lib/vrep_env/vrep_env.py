import vrep

import gym
import time
import numpy as np

class VrepEnv(gym.Env):
	"""Superclass for V-REP environments.
	"""
	def __init__(self,server_addr,server_port,scene_path=None):
		# Parameters
		self.server_addr = server_addr
		self.server_port = server_port
		self.scene_path  = scene_path
		
		self.opM_get = vrep.simx_opmode_blocking
		#self.opM_get = vrep.simx_opmode_oneshot
		self.opM_set = vrep.simx_opmode_oneshot
		
		# Status
		self.cID = -1
		self.connected = False
		self.scene_loaded = (scene_path == None)
		self.sim_running = False
		
		# Remote API function meaningful return codes
		self.str_simx_return = [
			'simx_return_ok',
			'simx_return_novalue_flag',
			'simx_return_timeout_flag',
			'simx_return_illegal_opmode_flag',
			'simx_return_remote_error_flag',
			'simx_return_split_progress_flag',
			'simx_return_local_error_flag',
			'simx_return_initialize_error_flag']
		
		self.connect(server_addr,server_port)
		if not self.scene_loaded:
			self.load_scene(scene_path)
	 
	# internal methods
	
	# Remote API call wrapper
	#def RAPI_rc(self, ret_tuple, tolerance=vrep.simx_return_ok):
	def RAPI_rc(self, ret_tuple, tolerance=vrep.simx_return_novalue_flag):
		istuple = isinstance(ret_tuple, tuple)
		ret = ret_tuple[0] if istuple else ret_tuple
		if (ret != vrep.simx_return_ok) and (ret != tolerance):
			raise RuntimeError('Remote API return code: ('+str(ret)+': '+self.str_simx_return[ret.bit_length()]+')')
		
		return ret_tuple[1:] if istuple else None
	
	def connect(self, server_addr, server_port):
		if self.connected:
			raise RuntimeError('Client is already connected.')
		attempts = 0
		max_attempts = 64
		while True:
			self.cID = vrep.simxStart(
				connectionAddress              = server_addr,
				connectionPort                 = server_port,
				waitUntilConnected             = True,
				doNotReconnectOnceDisconnected = True,
				timeOutInMs                    = 1000,
				commThreadCycleInMs            = 0)
			attempts += 1
			if self.cID != -1:
				self.connected = True
				break
			elif attempts < max_attempts:
				print('Unable to connect to V-REP at ',server_addr,':',server_port,'. Retrying...')
				time.sleep(4)
			else:
				raise RuntimeError('Unable to connect to V-REP.')
		
		# Setting up debug signal
		self.set_integer_signal('sig_debug',1337)
		
		# Getting useful parameter values
		self.is_headless = self.get_boolean_parameter(vrep.sim_boolparam_headless)
		
		# If not headless, remove GUI clutter
		if not self.is_headless:
			self.set_boolean_parameter(vrep.sim_boolparam_browser_visible  ,False)
			self.set_boolean_parameter(vrep.sim_boolparam_hierarchy_visible,False)
			#self.set_boolean_parameter(vrep.sim_boolparam_display_enabled  ,False)
			# Remove GUI controls
			#self.set_boolean_parameter(vrep.sim_boolparam_play_toolbarbutton_enabled  ,False)
			#self.set_boolean_parameter(vrep.sim_boolparam_pause_toolbarbutton_enabled ,False)
			#self.set_boolean_parameter(vrep.sim_boolparam_stop_toolbarbutton_enabled  ,False)
			self.set_boolean_parameter(vrep.sim_boolparam_console_visible  ,False)
		
		# Optionally override real-time mode
		self.set_boolean_parameter(vrep.sim_boolparam_realtime_simulation, False)
	
	def disconnect(self):
		if not self.connected:
			raise RuntimeError('Client is not even connected.')
		# Clearing debug signal
		vrep.simxClearIntegerSignal(self.cID,'sig_debug', vrep.simx_opmode_blocking)
		vrep.simxFinish(self.cID)
		self.connected = False
	
	def load_scene(self, scene_path):
		if self.scene_loaded:
			raise RuntimeError('Scene is already loaded.')
		self.RAPI_rc(vrep.simxLoadScene(self.cID,scene_path,0, vrep.simx_opmode_blocking))
		self.scene_loaded = True
	
	def close_scene(self):
		if not self.scene_loaded:
			raise RuntimeError('Scene is not loaded.')
		self.RAPI_rc(vrep.simxCloseScene(self.cID, vrep.simx_opmode_blocking))
		self.scene_loaded = False
	
	def start_simulation(self):
		if self.sim_running:
			raise RuntimeError('Simulation is already running.')
		
		# Optionally override physics engine ( 0=Bullet, 1=ODE, 2=Vortex, 3=Newton )
		#self.set_integer_parameter(vrep.sim_intparam_dynamic_engine, 0) # 0=Bullet
		
		# Optionally override delta time
		#self.set_float_parameter(vrep.sim_floatparam_simulation_time_step, 25)
		
		self.RAPI_rc(vrep.simxSynchronous(self.cID,True))
		self.RAPI_rc(vrep.simxStartSimulation(self.cID, vrep.simx_opmode_blocking))
		
		# Enable Threaded Rendering for faster simulation
		if not self.is_headless:
			self.set_boolean_parameter(vrep.sim_boolparam_threaded_rendering_enabled,True)
		
		self.sim_running = True
	
	def stop_simulation(self):
		if not self.sim_running:
			raise RuntimeError('Simulation is not running.')
		
		self.RAPI_rc(vrep.simxStopSimulation(self.cID, vrep.simx_opmode_blocking))
		
		# Checking if the server really stopped
		try:
			while True:
				self.RAPI_rc(vrep.simxGetIntegerSignal(self.cID,'sig_debug',vrep.simx_opmode_blocking))
				e = vrep.simxGetInMessageInfo(self.cID,vrep.simx_headeroffset_server_state)
				still_running = e[1] & 1
				if not still_running:
					break
		except: pass
		self.sim_running = False
	
	def step_simulation(self):
		self.RAPI_rc(vrep.simxSynchronousTrigger(self.cID))
	
	# Below are all wrapped methods unrelated to connection/scene
	
	# misc methods
	
	def add_statusbar_message(self, message):
		self.RAPI_rc(vrep.simxAddStatusbarMessage(self.cID, message, vrep.simx_opmode_blocking))
	
	# object methods
	
	def get_object_handle(self, name):
		handle, = self.RAPI_rc(vrep.simxGetObjectHandle(self.cID, name, vrep.simx_opmode_blocking))
		return handle
	
	# "getters"
	
	def obj_get_position(self, handle, relative_to=None):
		position, = self.RAPI_rc(vrep.simxGetObjectPosition( self.cID,handle,
			-1 if relative_to is None else relative_to,
			self.opM_get))
		return position
	def obj_get_orientation(self, handle, relative_to=None):
		eulerAngles, = self.RAPI_rc(vrep.simxGetObjectOrientation( self.cID,handle,
			-1 if relative_to is None else relative_to,
			self.opM_get))
		return eulerAngles
	def obj_get_orientation_continuous(self, handle, relative_to=None):
		ea = self.obj_get_orientation(handle,relative_to)
		return [
			np.sin(ea[0]),np.cos(ea[0]),
			np.sin(ea[1]),np.cos(ea[1]),
			np.sin(ea[2]),np.cos(ea[2])]
	
	# (linearVel, angularVel)
	def obj_get_velocity(self, handle):
		return self.RAPI_rc(vrep.simxGetObjectVelocity( self.cID,handle,
			self.opM_get))
	def obj_get_joint_angle(self, handle):
		angle, = self.RAPI_rc(vrep.simxGetJointPosition( self.cID,handle,
				self.opM_get))
		#return -np.rad2deg(angle[0])
		return angle
	def obj_get_joint_angle_continuous(self, handle):
		rad = self.obj_get_joint_angle(handle)
		return [np.sin(rad),np.cos(rad)]
	def obj_get_joint_force(self, handle):
		force = self.RAPI_rc(vrep.simxGetJointForce( self.cID,handle,
				self.opM_get))
		return force
	def obj_read_force_sensor(self, handle):
		state, forceVector, torqueVector = self.RAPI_rc(vrep.simxReadForceSensor( self.cID,handle,
			self.opM_get))
		if   state & 1 != 1: # bit 0 not set
			return None # sensor data not (yet) available
		elif state & 2 == 1: # bit 1 set
			return 0 # force sensor is broken
		else:
			return forceVector, torqueVector
	def obj_get_vision_image(self, handle):
		resolution, image = self.RAPI_rc(vrep.simxGetVisionSensorImage( self.cID,handle,
			0, # assume RGB
			self.opM_get,))
		dim, im = resolution, image
		nim = np.array(im, dtype='uint8')
		nim = np.reshape(nim, (dim[1], dim[0], 3))
		nim = np.flip(nim, 0)  # horizontal flip
		#nim = np.flip(nim, 2)  # RGB -> BGR
		return nim
	
	# "setters"
	
	def obj_set_position_target(self, handle, angle):
		return self.RAPI_rc(vrep.simxSetJointTargetPosition( self.cID,handle,
			-np.deg2rad(angle),
			self.opM_set))
	def obj_set_velocity(self, handle, v):
		return self.RAPI_rc(vrep.simxSetJointTargetVelocity( self.cID,handle,
			v,
			self.opM_set))
	def obj_set_force(self, handle, f):
		return self.RAPI_rc(vrep.simxSetJointForce( self.cID,handle,
			f,
			self.opM_set))
	def obj_set_position(self, handle, pos, relative_to=None):
		return self.RAPI_rc(vrep.simxSetObjectPosition( self.cID,handle,
			-1 if relative_to is None else relative_to,
			pos,
			self.opM_set))
	def obj_set_orientation(self, handle, eulerAngles, relative_to=None):
		return self.RAPI_rc(vrep.simxSetObjectOrientation( self.cID,handle,
			-1 if relative_to is None else relative_to,
			eulerAngles,
			self.opM_set))
	# collisions
	
	def get_collision_handle(self, name):
		handle, = self.RAPI_rc(vrep.simxGetCollisionHandle(self.cID, name, vrep.simx_opmode_blocking))
		return handle
	def read_collision(self, handle):
		collisionState, = self.RAPI_rc(vrep.simxReadCollision( self.cID,handle,
				self.opM_get))
		return collisionState
	
	# signals
	
	def set_integer_signal(self, sig_name, sig_val):
		return self.RAPI_rc(vrep.simxSetIntegerSignal( self.cID,
			sig_name, sig_val,
			self.opM_set))
	def set_float_signal(self, sig_name, sig_val):
		return self.RAPI_rc(vrep.SetFloatSignal( self.cID,
			sig_name, sig_val,
			self.opM_set))
	def set_string_signal(self, sig_name, sig_val):
		return self.RAPI_rc(vrep.SetStringSignal( self.cID,
			sig_name, sig_val,
			self.opM_set))
	
	def get_integer_signal(self, sig_name):
		return self.RAPI_rc(vrep.simxGetIntegerSignal( self.cID,
			sig_name,
			self.opM_get))
	def get_float_signal(self, sig_name):
		return self.RAPI_rc(vrep.simxGetFloatSignal( self.cID,
			sig_name,
			self.opM_get))
	def get_string_signal(self, sig_name):
		return self.RAPI_rc(vrep.simxGetStringSignal( self.cID,
			sig_name,
			self.opM_get))
	
	# parameters
	
	def set_boolean_parameter(self, param_id, param_val):
		return self.RAPI_rc(vrep.simxSetBooleanParameter( self.cID,
			param_id, param_val,
			vrep.simx_opmode_blocking))
	def set_integer_parameter(self, param_id, param_val):
		return self.RAPI_rc(vrep.simxSetIntegerParameter( self.cID,
			param_id, param_val,
			vrep.simx_opmode_blocking))
	def set_float_parameter(self, param_id, param_val):
		return self.RAPI_rc(vrep.simxSetFloatingParameter( self.cID,
			param_id, param_val,
			vrep.simx_opmode_blocking))
	def set_array_parameter(self, param_id, param_val):
		return self.RAPI_rc(vrep.simxSetArrayParameter( self.cID,
			param_id, param_val,
			vrep.simx_opmode_blocking))
	
	def get_boolean_parameter(self, param_id):
		return self.RAPI_rc(vrep.simxGetBooleanParameter( self.cID,
			param_id,
			vrep.simx_opmode_blocking))[0]
	def get_integer_parameter(self, param_id):
		return self.RAPI_rc(vrep.simxGetIntegerParameter( self.cID,
			param_id,
			vrep.simx_opmode_blocking))[0]
	def get_float_parameter(self, param_id):
		return self.RAPI_rc(vrep.simxGetFloatingParameter( self.cID,
			param_id,
			vrep.simx_opmode_blocking))[0]
	def get_array_parameter(self, param_id):
		return self.RAPI_rc(vrep.simxGetArrayParameter( self.cID,
			param_id,
			vrep.simx_opmode_blocking))[0]
	
	# scripts
	# child scripts
	def call_childscript_function(self,obj_name,func_name,in_tuple):
		return self.RAPI_rc(vrep.simxCallScriptFunction(self.cID,
			obj_name,vrep.sim_scripttype_childscript,func_name,
			in_tuple[0],in_tuple[1],in_tuple[2],in_tuple[3],
			vrep.simx_opmode_blocking))
	
	# openai/gym
	
	# Set this in SOME subclasses
	#metadata = {'render.modes': []}
	#reward_range = (-np.inf, np.inf)

	# Override in SOME subclasses
	#def _close(self): pass
	
	# Set these in ALL subclasses
	#action_space = None
	#observation_space = None
	
	# Override in ALL subclasses
	#def _step(self, action): raise NotImplementedError
	#def _reset(self): raise NotImplementedError
	#def _render(self, mode='human', close=False): return
	#def _seed(self, seed=None): return []
	
	#def _close(self):
	def close(self):
		if self.sim_running:
			self.stop_simulation()
		# Closing the scene is unnecessary
		#if self.scene_loaded:
		#	self.close_scene()
		if self.connected:
			self.disconnect()
	
