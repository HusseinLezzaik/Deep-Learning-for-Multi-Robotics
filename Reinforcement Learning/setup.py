"""
Setup file for gym and v-rep

@author: hussein
"""
# What you enter as the name variable of the setup is what you will use to import your environment(for eg. here, import gym_vrep).

# from setuptools import setup

# setup(name='gym_vrep',
#       version='0.0.1',
#       install_requires=['gym']#And any other dependencies required
# )

from setuptools import setup, find_packages

install_requires=[
    'gym',
    'numpy'
]

setup(name='vrep_env',
      version='0.0.2',
      description='V-REP integrated with OpenAI Gym',
      url='https://github.com/ycps/vrep-env',
      packages=[package for package in find_packages() if package.startswith('vrep_env')],
      install_requires=install_requires,
      package_data={'': ['remoteApi.so']},
      include_package_data=True
)