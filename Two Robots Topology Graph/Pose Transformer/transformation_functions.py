import math
import numpy as np


" Calculate Control inputs u1 and u2 "
            
u1 = np.array([[ k*(self.x2-self.x1)],[k*(self.y2-self.y1)]]) # 2x1 
u2 = np.array([[ k*(self.x1-self.x2)],[k*(self.y1-self.y2)]]) # 2x1
        
" Calculate V1/W1 and V2/W2 "
            
S1 = np.array([[self.v1], [self.w1]]) #2x1
G1 = np.array([[1,0], [0,1/L]]) #2x2
F1 = np.array([[math.cos(self.Theta1),math.sin(self.Theta1)],[-math.sin(self.Theta1),math.cos(self.Theta1)]]) #2x2
S1 = np.dot(np.dot(G1, F1), u1) #2x1
        
           
S2 = np.array([[self.v2], [self.w2]]) #2x1
G2 = np.array([[1,0], [0,1/L]]) #2x2
F2 = np.array([[math.cos(self.Theta2),math.sin(self.Theta2)],[-math.sin(self.Theta2),math.cos(self.Theta2)]]) #2x2
S2 = np.dot(np.dot(G2, F2), u2) # 2x1
            
" Calculate VL1/VR1 and VL2/VR2 "
            
D = np.array([[1/2,1/2],[-1/(2*d),1/(2*d)]]) #2x2
Di = np.linalg.inv(D) #2x2
        
    
    
Speed_L1 = np.array([[self.vL1], [self.vR1]]) # Vector 2x1 for Speed of Robot 1
Speed_L2 = np.array([[self.vL2], [self.vR2]]) # Vector 2x1 for Speed of Robot 2 
M1 = np.array([[S1[0]],[S1[1]]]).reshape(2,1) #2x1
M2 = np.array([[S2[0]], [S2[1]]]).reshape(2,1) #2x1
Speed_L1 = np.dot(Di, M1) # 2x1 (VL1, VR1)
Speed_L2 = np.dot(Di, M2) # 2x1 (VL2, VR2)
            
VL1 = float(Speed_L1[0])
VR1 = float(Speed_L1[1])
VL2 = float(Speed_L2[0])
VR2 = float(Speed_L2[1])
