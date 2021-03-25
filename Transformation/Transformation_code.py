import math
import numpy as np

k = 1
L = 1
d = 1
v1 = 0
w1 = 0
v2 = 0
w2 = 0
vL1 = 0
vR1 = 0
vL2 = 0 
vR2 = 0

" Calculate Control inputs u1 and u2 "
u1 = np.array([[ k*(x2-x1) ], [k*(y2-y1)]])
u2 = np.array([[ k*(x1-x2) ], [k*(y1-y2)]])

" Calculate V1/W1 and V2/W2 "
Speed_G1 = np.array([[v1], [w1]])
Speed_G2 = np.array([[v2], [w2]])
Speed_G1 = np.array([[1,0], [0, 1/L]])*np.array([[math.cos(Theta1), math.sin(Theta1)], [-math.sin(Theta1), math.cos(Theta1)]])*u1
Speed_G2 = np.array([[1,0], [0, 1/L]])*np.array([[math.cos(Theta2), math.sin(Theta2)], [-math.sin(Theta2), math.cos(Theta2)]])*u2

" Calculate VL1/VR1 and VL2/VR2 "
D = np.array([[ 1/2, 1/2], [-1/(2*d), 1/(2*d) ]])
Di = np.linalg.inv(D)
Speed_L1 = np.array([[vL1], [vR1]]) 
Speed_L2 = np.array([[vL2], [vR2]])
Speed_L1 = Di*np.array([[v1], [w1]])
Speed_L2 = Di*np.array([[v2], [w2]])
