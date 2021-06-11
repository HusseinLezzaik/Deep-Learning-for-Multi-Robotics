"""
Init file

@author: hussein
"""

from gym.envs.registration import register

register(
    id='vrep-v0',
    entry_point='gym_vrep.envs:VrepEnv',
    max_episode_steps=2000,
)

