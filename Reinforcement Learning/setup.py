"""
Setup file for gym and v-rep

@author: hussein
"""
# What you enter as the name variable of the setup is what you will use to import your environment(for eg. here, import gym_vrep).

from setuptools import setup

setup(name='gym_vrep',
      version='0.0.1',
      install_requires=['gym']#And any other dependencies required
)