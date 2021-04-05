#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control Initial Position of two robots using Python Remote API 

@author: hussein
"""

import sim
import time 
import math
import numpy as np
print("Program Started")

" Control to V-Rep "

sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim

iter = 1
x_disp = np.array([0], [1])
y_disp =  np.array([0], [1])
z_rot =  np.array([0], [90])

if clientID!=-1:
    print ('Connected to remote API server')
    for z in z_rot:
        for x in x_disp:
            for y in y_disp:
                print(" Simulation iter")
                # Set Camera_robo Positon
                
                
                
                
                
                # Start the Simulation
                
                
                

else:
    
