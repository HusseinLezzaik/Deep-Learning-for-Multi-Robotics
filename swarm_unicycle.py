"""

Control of a Unicycle Model of a Mobile Robot

"""

import math
import numpy as np


def getPose(obj):
    q = np.array([[obj.x], [obj.y], [obj.th]])
    return q
 
 
def setPose(obj, q):
    obj.x = q[0]
    obj.y = q[1]
    obj.th = q[2]
 
 
def moveUnicycle(obj, vOmega):
     obj.x = obj.x + (vOmega[0]*math.cos(obj.th))*obj.DT
     obj.y = obj.y + (vOmega[0]*math.sin(obj.th))*obj.DT
     obj.th = obj.th + vOmega[1]*obj.DT
     obj.th = math.atan2(math.sin(obj.th), math.cos(obj.th))
     
def moveSingleIntegrator(obj,v):
    vOmega = np.array([[1,0], [0, 1/obj.L]])*np.array([[math.cos(obj.th), math.sin(obj.th)], [-math.sin(obj.th), math.cos(obj.th)]])*v
    obj.moveUnicycle(vOmega)
    
    
def differentialSpeed(v,w,d):
    D = np.array([[ 1/2, 1/2], [-1/(2*d), 1/(2*d) ]])
    Di = np.linalg.inv(D)
    vL=0 
    vR=0
    Vd = np.array([[vL],[vR]])
    V = np.array([[v],[w]])
    Vd = Di * V
    return Vd
    
    

