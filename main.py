"""

Publisher node speed of a single Mobile Robot Wheel

"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float32
import csv
import pandas as pd

k = 1 # Control Gain
L = 1
d = 0.5
 
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
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            10)
        " Mobile Robot 1 Parameters "
        self.x1 = 20
        self.y1 = 20
        self.Theta1 = 0
        self.v1 = 1
        self.w1 = 1
        self.vL1 = 2
        self.vR1 = 2
        " Mobile Robot 1 Parameters "
        self.x2 = 20
        self.y2 = 20
        self.Theta2 = 0
        self.v2 = 2
        self.w2 = 2
        self.vL2 = 2 
        self.vR2 = 2
        self.i1 = 0
        self.i2 = 0
        
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
        
        
        " Calculate the Pose of Robot 2 w.r.t Robot 1 and Control input U1 "
        
        self.X1 = self.x2 - self.x1 # 1x1
        self.Y1 = self.y2 -self.y1 # 1x1
        self.U1 = u1 # 2x1
        
        with open('robot1.csv', 'a', newline='') as f:
            fieldnames = ['Data_X', 'Data_Y', 'Label_X', 'Label_Y']
            thewriter = csv.DictWriter(f, fieldnames=fieldnames)
            
            if self.i1 == 0:
                thewriter.writeheader()
                self.i1 = 1
            
            thewriter.writerow({'Data_X' : self.X1, 'Data_Y' : self.Y1, 'Label_X' : self.U1[0], 'Label_Y' : self.U1[1]})
                
        
        " Calculate the Pose of Robot 1 w.r.t Robot 2 and Control input U2 "
        
        self.X2 = self.x1 - self.x2 # 1x1
        self.Y2 = self.y1 -self.y2 # 1x1
        self.U2 = u2 # 2x1
        
        with open('robot2.csv', 'a', newline='') as f:
            fieldnames = ['Data_X', 'Data_Y', 'Label_X', 'Label_Y']
            thewriter = csv.DictWriter(f, fieldnames=fieldnames)
            
            if self.i2 == 0:
                thewriter.writeheader()
                self.i2 = 1
            
            thewriter.writerow({'Data_X' : self.X2, 'Data_Y' : self.Y2, 'Label_X' : self.U2[0], 'Label_Y' : self.U2[1]})
        
        " Speed Commands to Robot 1"
        
        msgl1 = Float32()    
        msgr1 = Float32()
        msgl1.data = VL1
        msgr1.data = VR1
        self.publisher_l1.publish(msgl1)
        self.publisher_r1.publish(msgr1)
        #self.get_logger().info('Publishing R1: "%s"' % msgr1.data)
         
        
        " Speed Commands to Robot 2"
        
        msgl2 = Float32()
        msgr2 = Float32()
        msgl2.data = VL2
        msgr2.data = VR2
        self.publisher_l2.publish(msgl2)
        self.publisher_r2.publish(msgr2)
            
            
       
        
       
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
