"""
Created on Fri Jun 11 15:30:43 2021

@author: hussein
"""

from gym.envs.registration import register

register(
    id='vrep-v0',
    entry_point='gym_vrep.envs:VrepEnv',
)

