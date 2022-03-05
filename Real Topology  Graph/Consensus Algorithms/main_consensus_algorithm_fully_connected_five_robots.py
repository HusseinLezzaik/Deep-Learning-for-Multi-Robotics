"""

Consensus Algorithm for Six mobile robots Fully Connected Adjancency Matrix 

"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float32
# importing the required module
import matplotlib.pyplot as plt
import time


k = 1 # Control Gain
L = 0.0975 # Pioneer Robot Parameters
d = 0.109561 # Pioneer Robot Parameters
A = np.ones(6) - np.identity(6) # Adjancency Matrix fully connected case 6x6

ux = np.zeros((6,1)) # 6x1
uy = np.zeros((6,1)) # 6x1


def euler_from_quaternion(x, y, z, w):
        
     t3 = +2.0 * (w * z + x * y)
     t4 = +1.0 - 2.0 * (y * y + z * z)
     yaw_z = math.atan2(t3, t4)
     
     return yaw_z # in radians
 
    

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher1')
        self.publisher_l1 = self.create_publisher(Float32, '/leftMotorSpeedrobot1', 10) #Change according to topic in child script,String to Float32
        self.publisher_r1 = self.create_publisher(Float32, '/rightMotorSpeedrobot1', 10) #Change according to topic in child script,String to Float32
        self.publisher_l2 = self.create_publisher(Float32, '/leftMotorSpeedrobot2', 10) #Change according to topic in child script,String to Float32
        self.publisher_r2 = self.create_publisher(Float32, '/rightMotorSpeedrobot2', 10) #Change according to topic in child script,String to Float32
        self.publisher_l3 = self.create_publisher(Float32, '/leftMotorSpeedrobot3', 10) #Change according to topic in child script,String to Float32
        self.publisher_r3 = self.create_publisher(Float32, '/rightMotorSpeedrobot3', 10) #Change according to topic in child script,String to Float32
        self.publisher_l4 = self.create_publisher(Float32, '/leftMotorSpeedrobot4', 10) #Change according to topic in child script,String to Float32
        self.publisher_r4 = self.create_publisher(Float32, '/rightMotorSpeedrobot4', 10) #Change according to topic in child script,String to Float32
        self.publisher_l5 = self.create_publisher(Float32, '/leftMotorSpeedrobot5', 10) #Change according to topic in child script,String to Float32
        self.publisher_r5 = self.create_publisher(Float32, '/rightMotorSpeedrobot5', 10) #Change according to topic in child script,String to Float32
        self.publisher_l6 = self.create_publisher(Float32, '/leftMotorSpeedrobot6', 10) #Change according to topic in child script,String to Float32
        self.publisher_r6 = self.create_publisher(Float32, '/rightMotorSpeedrobot6', 10) #Change according to topic in child script,String to Float32          
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            10)
        
        " Mobile Robot 1 Parameters "
        self.x1 = 0
        self.y1 = 0
        self.Theta1 = 0
        self.v1 = 0
        self.w1 = 0
        self.vL1 = 0
        self.vR1 = 0
        
        " Mobile Robot 1 Parameters "
        self.x2 = 0
        self.y2 = 0
        self.Theta2 = 0
        self.v2 = 0
        self.w2 = 0
        self.vL2 = 0 
        self.vR2 = 0
        
        " Mobile Robot 3 Parameters "
        self.x3 = 0
        self.y3 = 0
        self.Theta3 = 0
        self.v3 = 0
        self.w3 = 0
        self.vL3 = 0
        self.vR3 = 0                
        
        " Mobile Robot 4 Parameters "
        self.x4 = 0
        self.y4 = 0
        self.Theta4 = 0
        self.v4 = 0
        self.w4 = 0
        self.vL4 = 0
        self.vR4 = 0       
        
        " Mobile Robot 5 Parameters "
        self.x5 = 0
        self.y5 = 0
        self.Theta5 = 0
        self.v5 = 0
        self.w5 = 0
        self.vL5 = 0
        self.vR5 = 0

        " Mobile Robot 6 Parameters "
        self.x6 = 0
        self.y6 = 0
        self.Theta6 = 0
        self.v6 = 0
        self.w6 = 0
        self.vL6 = 0
        self.vR6 = 0            
                
        
        " Time Ellapsed Variable"
        self.counter = 0
        
    def listener_callback(self, msg):

        
        if msg.transforms[0].child_frame_id == 'robot1' :  
            self.x1 = msg.transforms[0].transform.translation.x
            self.y1 = msg.transforms[0].transform.translation.y
            self.xr1 = msg.transforms[0].transform.rotation.x
            self.yr1 = msg.transforms[0].transform.rotation.y
            self.zr1 = msg.transforms[0].transform.rotation.z
            self.wr1 = msg.transforms[0].transform.rotation.w
            self.Theta1 = euler_from_quaternion(self.xr1,self.yr1,self.zr1,self.wr1)
                       
            
        if  msg.transforms[0].child_frame_id == 'robot2' :
            
            self.x2 = msg.transforms[0].transform.translation.x
            self.y2 = msg.transforms[0].transform.translation.y
            self.xr2 = msg.transforms[0].transform.rotation.x
            self.yr2 = msg.transforms[0].transform.rotation.y
            self.zr2 = msg.transforms[0].transform.rotation.z
            self.wr2 = msg.transforms[0].transform.rotation.w
            self.Theta2 = euler_from_quaternion(self.xr2,self.yr2,self.zr2,self.wr2)
            
        if  msg.transforms[0].child_frame_id == 'robot3' :
            
            self.x3 = msg.transforms[0].transform.translation.x
            self.y3 = msg.transforms[0].transform.translation.y
            self.xr3 = msg.transforms[0].transform.rotation.x
            self.yr3 = msg.transforms[0].transform.rotation.y
            self.zr3 = msg.transforms[0].transform.rotation.z
            self.wr3 = msg.transforms[0].transform.rotation.w
            self.Theta3 = euler_from_quaternion(self.xr3,self.yr3,self.zr3,self.wr3)

        if  msg.transforms[0].child_frame_id == 'robot4' :
            
            self.x4 = msg.transforms[0].transform.translation.x
            self.y4 = msg.transforms[0].transform.translation.y
            self.xr4 = msg.transforms[0].transform.rotation.x
            self.yr4 = msg.transforms[0].transform.rotation.y
            self.zr4 = msg.transforms[0].transform.rotation.z
            self.wr4 = msg.transforms[0].transform.rotation.w
            self.Theta4 = euler_from_quaternion(self.xr4,self.yr4,self.zr4,self.wr4)        
            
        if  msg.transforms[0].child_frame_id == 'robot5' :
            
            self.x5 = msg.transforms[0].transform.translation.x
            self.y5 = msg.transforms[0].transform.translation.y
            self.xr5 = msg.transforms[0].transform.rotation.x
            self.yr5 = msg.transforms[0].transform.rotation.y
            self.zr5 = msg.transforms[0].transform.rotation.z
            self.wr5 = msg.transforms[0].transform.rotation.w
            self.Theta5 = euler_from_quaternion(self.xr5,self.yr5,self.zr5,self.wr5)               
            
        if  msg.transforms[0].child_frame_id == 'robot6' :
            
            self.x6 = msg.transforms[0].transform.translation.x
            self.y6 = msg.transforms[0].transform.translation.y
            self.xr6 = msg.transforms[0].transform.rotation.x
            self.yr6 = msg.transforms[0].transform.rotation.y
            self.zr6 = msg.transforms[0].transform.rotation.z
            self.wr6 = msg.transforms[0].transform.rotation.w
            self.Theta6 = euler_from_quaternion(self.xr6,self.yr6,self.zr6,self.wr6)             
           
        
        " Calculate Control inputs u1, u2, u3, u4, u5, u6 "
        
        
        # self.distance = abs(self.x2 - self.x3) + abs(self.y2 - self.y3) + abs(self.x1 - self.x3) + abs(self.y1 - self.y3) + abs(self.x1 - self.x4) + abs(self.y1 - self.y4) + abs(self.x1 - self.x5) + abs(self.y1 - self.y5) + abs(self.x1 - self.x6) + abs(self.y1 - self.y6)     
    
        
        A = np.ones(5) - np.identity(5) # Adjancency Matrix
        
        self.X = np.array([ [self.x1], [self.x3], [self.x4], [self.x5], [self.x6] ]) #6x1
        self.Y = np.array([ [self.y1], [self.y3], [self.y4], [self.y5], [self.y6] ]) #6x1
        
        ux = np.zeros((5,1)) # 6x1
        uy = np.zeros((5,1)) # 6x1
        
                    
        for i in range(1,6):
            for j in range(1,6):
                ux[i-1] += -(A[i-1][j-1])*(self.X[i-1]-self.X[j-1]) # 1x1 each
                uy[i-1] += -(A[i-1][j-1])*(self.Y[i-1]-self.Y[j-1]) # 1x1 each
            
        
        u1 = np.array([ [float(ux[0])], [float(uy[0])] ]) # 2x1
        # u2 = np.array([ [float(ux[0])], [float(uy[0])] ]) # 2x1
        u3 = np.array([ [float(ux[1])], [float(uy[1])] ]) # 2x1
        u4 = np.array([ [float(ux[2])], [float(uy[2])] ]) # 2x1
        u5 = np.array([ [float(ux[3])], [float(uy[3])] ]) # 2x1
        u6 = np.array([ [float(ux[4])], [float(uy[4])] ]) # 2x1
        
        " Calculate V1/W1, V2/W2, V3/W3, V4/W4, V5/W5, V6/W6 "
            
        S1 = np.array([[self.v1], [self.w1]]) #2x1
        G1 = np.array([[1,0], [0,1/L]]) #2x2
        R1 = np.array([[math.cos(self.Theta1),math.sin(self.Theta1)],[-math.sin(self.Theta1),math.cos(self.Theta1)]]) #2x2
        S1 = np.dot(np.dot(G1, R1), u1) #2x1
        
        # S2 = np.array([[self.v2], [self.w2]]) #2x1
        # G2 = np.array([[1,0], [0,1/L]]) #2x2
        # R2 = np.array([[math.cos(self.Theta2),math.sin(self.Theta2)],[-math.sin(self.Theta2),math.cos(self.Theta2)]]) #2x2
        # S2 = np.dot(np.dot(G2, R2), u2) # 2x1
        
        S3 = np.array([[self.v3], [self.w3]]) #2x1
        G3 = np.array([[1,0], [0,1/L]]) #2x2
        R3 = np.array([[math.cos(self.Theta3),math.sin(self.Theta3)],[-math.sin(self.Theta3),math.cos(self.Theta3)]]) #2x2
        S3 = np.dot(np.dot(G3, R3), u3) #2x1        
        
        S4 = np.array([[self.v4], [self.w4]]) #2x1
        G4 = np.array([[1,0], [0,1/L]]) #2x2
        R4 = np.array([[math.cos(self.Theta4),math.sin(self.Theta4)],[-math.sin(self.Theta4),math.cos(self.Theta4)]]) #2x2
        S4 = np.dot(np.dot(G4, R4), u4) #2x1        
        
        S5 = np.array([[self.v5], [self.w5]]) #2x1
        G5 = np.array([[1,0], [0,1/L]]) #2x2
        R5 = np.array([[math.cos(self.Theta5),math.sin(self.Theta5)],[-math.sin(self.Theta5),math.cos(self.Theta5)]]) #2x2
        S5 = np.dot(np.dot(G5, R5), u5) #2x1
        
        S6 = np.array([[self.v6], [self.w6]]) #2x1
        G6 = np.array([[1,0], [0,1/L]]) #2x2
        R6 = np.array([[math.cos(self.Theta6),math.sin(self.Theta6)],[-math.sin(self.Theta6),math.cos(self.Theta6)]]) #2x2
        S6 = np.dot(np.dot(G6, R6), u6) #2x1        
        
        
        " Calculate VL1/VR1, VL2/VR2, VL3/VR3, VL4/VR4, VL5/VR5, VL6/VR6 "
            
        D = np.array([[1/2,1/2],[-1/(2*d),1/(2*d)]]) #2x2
        Di = np.linalg.inv(D) #2x2
        
        Speed_L1 = np.array([[self.vL1], [self.vR1]]) # Vector 2x1 for Speed of Robot 1
        #Speed_L2 = np.array([[self.vL2], [self.vR2]]) # Vector 2x1 for Speed of Robot 2 
        Speed_L3 = np.array([[self.vL3], [self.vR3]]) # Vector 2x1 for Speed of Robot 3
        Speed_L4 = np.array([[self.vL4], [self.vR4]]) # Vector 2x1 for Speed of Robot 4
        Speed_L5 = np.array([[self.vL5], [self.vR5]]) # Vector 2x1 for Speed of Robot 5
        Speed_L6 = np.array([[self.vL6], [self.vR6]]) # Vector 2x1 for Speed of Robot 6
        
        
        M1 = np.array([[S1[0]],[S1[1]]]).reshape(2,1) #2x1
        #M2 = np.array([[S2[0]],[S2[1]]]).reshape(2,1) #2x1
        M3 = np.array([[S3[0]],[S3[1]]]).reshape(2,1) #2x1
        M4 = np.array([[S4[0]],[S4[1]]]).reshape(2,1) #2x1
        M5 = np.array([[S5[0]],[S5[1]]]).reshape(2,1) #2x1
        M6 = np.array([[S6[0]],[S6[1]]]).reshape(2,1) #2x1
        
        Speed_L1 = np.dot(Di, M1) # 2x1 (VL1, VR1)
        #Speed_L2 = np.dot(Di, M2) # 2x1 (VL2, VR2)
        Speed_L3 = np.dot(Di, M3) # 2x1 (VL3, VR3)
        Speed_L4 = np.dot(Di, M4) # 2x1 (VL4, VR4)
        Speed_L5 = np.dot(Di, M5) # 2x1 (VL5, VR5)
        Speed_L6 = np.dot(Di, M6) # 2x1 (VL6, VR6)
        print(" Velocity of Robot 6", Speed_L6)
        
        
        VL1 = float(Speed_L1[0])
        VR1 = float(Speed_L1[1])
        #VL2 = float(Speed_L2[0])
        #VR2 = float(Speed_L2[1])
        VL3 = float(Speed_L3[0])
        VR3 = float(Speed_L3[1])
        VL4 = float(Speed_L4[0])
        VR4 = float(Speed_L4[1])
        VL5 = float(Speed_L5[0])
        VR5 = float(Speed_L5[1])        
        VL6 = 0.6*float(Speed_L6[0])
        VR6 = 0.6*float(Speed_L6[1])
        
        # " Publish Speed Commands to Robot 1 "
        
        msgl1 = Float32()    
        msgr1 = Float32()
        msgl1.data = VL1
        msgr1.data = VR1
        self.publisher_l1.publish(msgl1)
        self.publisher_r1.publish(msgr1)
        #self.get_logger().info('Publishing R1: "%s"' % msgr1.data)
        
        
        " Publish Speed Commands to Robot 2 "
        
        # msgl2 = Float32()
        # msgr2 = Float32()
        # msgl2.data = VL2
        # msgr2.data = VR2
        # self.publisher_l2.publish(msgl2)
        # self.publisher_r2.publish(msgr2)

        " Publish Speed Commands to Robot 3 "
        
        msgl3 = Float32()
        msgr3 = Float32()
        msgl3.data = VL3
        msgr3.data = VR3
        self.publisher_l3.publish(msgl3)
        self.publisher_r3.publish(msgr3)
        
        " Publish Speed Commands to Robot 4 "
        
        msgl4 = Float32()
        msgr4 = Float32()
        msgl4.data = VL4
        msgr4.data = VR4
        self.publisher_l4.publish(msgl4)
        self.publisher_r4.publish(msgr4)        
        
        
        " Publish Speed Commands to Robot 5 "
        
        msgl5 = Float32()
        msgr5 = Float32()
        msgl5.data = VL5
        msgr5.data = VR5
        self.publisher_l5.publish(msgl5)
        self.publisher_r5.publish(msgr5)        
        
        
        # " Publish Speed Commands to Robot 6 "
        
        msgl6 = Float32()
        msgr6 = Float32()
        msgl6.data = VL6
        msgr6.data = VR6
        self.publisher_l6.publish(msgl6)
        self.publisher_r6.publish(msgr6)        
        
        

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    # # x axis values
    # x = [1,2,3]
    # # corresponding y axis values
    # y = [2,4,1]
      
    # # plotting the points 
    # plt.plot(x, y)
      
    # # naming the x axis
    # plt.xlabel('time - axis')
    # # naming the y axis
    # plt.ylabel('disk distance - axis')
      
    # # giving a title to my graph
    # plt.title('Disk Distance vs Time')
      
    # # function to show the plot
    # plt.show()       
    
    #print(minimal_publisher.distance) 
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
