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
x_disp = np.array([[0], [1]])
y_disp =  np.array([[0], [1]])
z_rot =  np.array([[0], [90]])

if clientID!=-1:
    print ('Connected to remote API server')
    for z in z_rot:
        for x in x_disp:
            for y in y_disp:
                print(" Simulation ", iter)
                # Set Camera_robo Positon
                ErrLocM, LocM = sim.simxGetObjectHandle(clientID, 'CoppeliaSim_Two_Robots_Scenes', sim.simx_opmode_oneshot_wait)
                ErrLoc, Loc = sim.simxGetObjectPosition(clientID, LocM, -1, sim.simx_opmode_oneshot_wait)
                ErrLocO, OriRobo = sim.simxGetObjectOrientation(clientID,LocM, -1, sim.simx_opmode_oneshot_wait)
                OriRobo[0] = ((z*math.pi)/100)
                sim.simxSetObjectOrientation(clientID, LocM, -1, OriRobo, sim.simx_opmode_oneshot_wait) # Set Robot Orientation
                Loc[0] = x
                Loc[1] = y
                sim.simxSetObjectPosition(clientID, LocM, -1, Loc, sim.simx_opmode_oneshot) # Set Robot Position
                # Start the Simulation:
                print("bubbleRob Position:", Loc)
                print("bubbleRob Orientation:", OriRobo)
                print("Simulation Running ...")
                sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)
                time.sleep(5)
                sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
                iter += 1

else:
    print("Failed connecting to remote API server") 
print("Program Ended")

# End Connection to V-Rep
sim.simxFinish(clientID)
