"""

Consensus Algorithm for 6 Mobile robots using MLP Model Fully Connected Graph Implementation

Inputs: Mx, My
Outputs: Ux, Uy

"""
import torch
import MLP_Model
import math
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float32

L = 1
d = 0.5
#distance = 2

A = np.ones(6) - np.identity(6) # Adjancency Matrix fully connected case 6x6

ux = np.zeros((6,1)) # 6x1 controller vector
uy = np.zeros((6,1)) # 6x1 controller vector

# load model using dict
FILE = "model.pth"
loaded_model = MLP_Model.MLP()
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()


def euler_from_quaternion(x, y, z, w):
        
     t3 = +2.0 * (w * z + x * y)
     t4 = +1.0 - 2.0 * (y * y + z * z)
     yaw_z = math.atan2(t3, t4)
     
     return yaw_z # in radians


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher1')
        self.publisher_l1 = self.create_publisher(Float32, '/leftMotorSpeedrobot1', 0) #Change according to topic in child script,String to Float32
        self.publisher_r1 = self.create_publisher(Float32, '/rightMotorSpeedrobot1', 0) #Change according to topic in child script,String to Float32
        self.publisher_l2 = self.create_publisher(Float32, '/leftMotorSpeedrobot2', 0) #Change according to topic in child script,String to Float32
        self.publisher_r2 = self.create_publisher(Float32, '/rightMotorSpeedrobot2', 0) #Change according to topic in child script,String to Float32
        self.publisher_l3 = self.create_publisher(Float32, '/leftMotorSpeedrobot3', 0) #Change according to topic in child script,String to Float32
        self.publisher_r3 = self.create_publisher(Float32, '/rightMotorSpeedrobot3', 0) #Change according to topic in child script,String to Float32
        self.publisher_l4 = self.create_publisher(Float32, '/leftMotorSpeedrobot4', 0) #Change according to topic in child script,String to Float32
        self.publisher_r4 = self.create_publisher(Float32, '/rightMotorSpeedrobot4', 0) #Change according to topic in child script,String to Float32
        self.publisher_l5 = self.create_publisher(Float32, '/leftMotorSpeedrobot5', 0) #Change according to topic in child script,String to Float32
        self.publisher_r5 = self.create_publisher(Float32, '/rightMotorSpeedrobot5', 0) #Change according to topic in child script,String to Float32
        self.publisher_l6 = self.create_publisher(Float32, '/leftMotorSpeedrobot6', 0) #Change according to topic in child script,String to Float32
        self.publisher_r6 = self.create_publisher(Float32, '/rightMotorSpeedrobot6', 0) #Change according to topic in child script,String to Float32              
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            0)

        "Parameters "
        self.k = 1 # Control Gain
        self.scene = 0 # Nb of scene iteration
        
        " Mobile Robot 1 Parameters "
        self.x1 = 10
        self.y1 = 12
        self.Theta1 = 0
        self.v1 = 1
        self.w1 = 1
        self.vL1 = 2
        self.vR1 = 2
        
        " Mobile Robot 1 Parameters "
        self.x2 = 5
        self.y2 = 7
        self.Theta2 = 0
        self.v2 = 2
        self.w2 = 2
        self.vL2 = 2 
        self.vR2 = 2
        
        " Mobile Robot 3 Parameters "
        self.x3 = 13
        self.y3 = 14
        self.Theta3 = 0
        self.v3 = 3
        self.w3 = 3
        self.vL3 = 2
        self.vR3 = 2                
        
        " Mobile Robot 4 Parameters "
        self.x4 = 14
        self.y4 = 15
        self.Theta4 = 0
        self.v4 = 4
        self.w4 = 4
        self.vL4 = 2
        self.vR4 = 2        
        
        " Mobile Robot 5 Parameters "
        self.x5 = 15
        self.y5 = 16
        self.Theta5 = 0
        self.v5 = 5
        self.w5 = 5
        self.vL5 = 2
        self.vR5 = 2

        " Mobile Robot 6 Parameters "
        self.x6 = 16
        self.y6 = 17
        self.Theta6 = 0
        self.v6 = 6
        self.w6 = 6
        self.vL6 = 2
        self.vR6 = 2
        
        " Mx, My Initialization "
        self.Mx1 = 0.1
        self.My1 = 0.1
        self.Mx2 = 0.1
        self.My2 = 0.1
        self.Mx3 = 0.1
        self.My3 = 0.1
        self.Mx4 = 0.1
        self.My4 = 0.1
        self.Mx5 = 0.1
        self.My5 = 0.1
        self.Mx6 = 0.1
        self.My6 = 0.1        
        
        

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
        
                
        " Calculate Mx1, My1, ...... Mx6, My6 "
        
        #Phix = ( u1[0][0] + u3[0][0] )/2 # 1x1
        #Phiy = ( u1[1][0] + u3[1][0] )/2 # 1x1
        
        self.Mx1 = ( ( self.x2 - self.x1 ) + ( self.x6 - self.x1 ) ) / 2 # 1x1
        self.My1 = ( ( self.y2 - self.y1 ) + ( self.y6 - self.y1 ) ) / 2 # 1x1
        
        self.Mx2 = ( ( self.x1 - self.x2 ) + ( self.x3 - self.x2 ) ) / 2 # 1x1
        self.My2 = ( ( self.y1 - self.y2 ) + ( self.y3 - self.y2 ) ) / 2 # 1x1            

        self.Mx3 = ( ( self.x2 - self.x3 ) + ( self.x4 - self.x3 ) ) / 2 # 1x1
        self.My3 = ( ( self.y2 - self.y3 ) + ( self.y4 - self.y3 ) ) / 2 # 1x1               
        
        self.Mx4 = ( ( self.x3 - self.x4 ) + ( self.x5 - self.x4 ) ) / 2 # 1x1
        self.My4 = ( ( self.y4 - self.y4 ) + ( self.y5 - self.y4 ) ) / 2 # 1x1               

        self.Mx5 = ( ( self.x4 - self.x5 ) + ( self.x6 - self.x5 ) ) / 2 # 1x1
        self.My5 = ( ( self.y4 - self.y5 ) + ( self.y6 - self.y5 ) ) / 2 # 1x1   
        
        self.Mx6 = ( ( self.x5 - self.x6 ) + ( self.x1 - self.x6 ) ) / 2 # 1x1
        self.My6 = ( ( self.y5 - self.y6 ) + ( self.y1 - self.y6 ) ) / 2 # 1x1   

        " Use MLP to Predict control inputs "
        
        relative_pose_1 = [ self.Mx1, self.My1 ] # tensor data for MLP model
        relative_pose_2 = [ self.Mx2, self.My2 ] # tensor data for MLP model
        relative_pose_3 = [ self.Mx3, self.My3 ] # tensor data for MLP model
        relative_pose_4 = [ self.Mx4, self.My4 ] # tensor data for MLP model
        relative_pose_5 = [ self.Mx5, self.My5 ] # tensor data for MLP model
        relative_pose_6 = [ self.Mx6, self.My6 ] # tensor data for MLP model
        
        u1_predicted = MLP_Model.predict(relative_pose_1, loaded_model) # predict control input u1, tensor
        u2_predicted = MLP_Model.predict(relative_pose_2, loaded_model) # predict control input u2, tensor
        u3_predicted = MLP_Model.predict(relative_pose_3, loaded_model) # predict control input u3, tensor
        u4_predicted = MLP_Model.predict(relative_pose_4, loaded_model) # predict control input u4, tensor
        u5_predicted = MLP_Model.predict(relative_pose_5, loaded_model) # predict control input u5, tensor
        u6_predicted = MLP_Model.predict(relative_pose_6, loaded_model) # predict control input u6, tensor
        
        u1_predicted_np = np.array([[ u1_predicted[0][0] ], [ u1_predicted[0][1] ]]) # from tensor to numpy array for calculation
        u2_predicted_np = np.array([[ u2_predicted[0][0] ], [ u2_predicted[0][1] ]]) # from tensor to numpy array for calculation
        u3_predicted_np = np.array([[ u3_predicted[0][0] ], [ u3_predicted[0][1] ]]) # from tensor to numpy array for calculation
        u4_predicted_np = np.array([[ u4_predicted[0][0] ], [ u4_predicted[0][1] ]]) # from tensor to numpy array for calculation    
        u5_predicted_np = np.array([[ u5_predicted[0][0] ], [ u5_predicted[0][1] ]]) # from tensor to numpy array for calculation
        u6_predicted_np = np.array([[ u6_predicted[0][0] ], [ u6_predicted[0][1] ]]) # from tensor to numpy array for calculation
            
                              
        " Calculate V1/W1, V2/W2, V3/W3, V4/W4, V5/W5, V6/W6 "
        
        S1 = np.array([[self.v1], [self.w1]]) #2x1
        G1 = np.array([[1,0], [0,1/L]]) #2x2
        R1 = np.array([[math.cos(self.Theta1),math.sin(self.Theta1)],[-math.sin(self.Theta1),math.cos(self.Theta1)]]) #2x2
        S1 = np.dot(np.dot(G1, R1), u1_predicted_np) #2x1
    
        S2 = np.array([[self.v2], [self.w2]]) #2x1
        G2 = np.array([[1,0], [0,1/L]]) #2x2
        R2 = np.array([[math.cos(self.Theta2),math.sin(self.Theta2)],[-math.sin(self.Theta2),math.cos(self.Theta2)]]) #2x2
        S2 = np.dot(np.dot(G2, R2), u2_predicted_np) # 2x1

        S3 = np.array([[self.v3], [self.w3]]) #2x1
        G3 = np.array([[1,0], [0,1/L]]) #2x2
        R3 = np.array([[math.cos(self.Theta3),math.sin(self.Theta3)],[-math.sin(self.Theta3),math.cos(self.Theta3)]]) #2x2
        S3 = np.dot(np.dot(G3, R3), u3_predicted_np) #2x1        

        S4 = np.array([[self.v4], [self.w4]]) #2x1
        G4 = np.array([[1,0], [0,1/L]]) #2x2
        R4 = np.array([[math.cos(self.Theta4),math.sin(self.Theta4)],[-math.sin(self.Theta4),math.cos(self.Theta4)]]) #2x2
        S4 = np.dot(np.dot(G4, R4), u4_predicted_np) #2x1        

        S5 = np.array([[self.v5], [self.w5]]) #2x1
        G5 = np.array([[1,0], [0,1/L]]) #2x2
        R5 = np.array([[math.cos(self.Theta5),math.sin(self.Theta5)],[-math.sin(self.Theta5),math.cos(self.Theta5)]]) #2x2
        S5 = np.dot(np.dot(G5, R5), u5_predicted_np) #2x1

        S6 = np.array([[self.v6], [self.w6]]) #2x1
        G6 = np.array([[1,0], [0,1/L]]) #2x2
        R6 = np.array([[math.cos(self.Theta6),math.sin(self.Theta6)],[-math.sin(self.Theta6),math.cos(self.Theta6)]]) #2x2
        S6 = np.dot(np.dot(G6, R6), u6_predicted_np) #2x1        
                
        
        " Calculate VL1/VR1, VL2/VR2, VL3/VR3, VL4/VR4, VL5/VR5, VL6/VR6 "
    
        D = np.array([[1/2,1/2],[-1/(2*d),1/(2*d)]]) #2x2
        Di = np.linalg.inv(D) #2x2

        Speed_L1 = np.array([[self.vL1], [self.vR1]]) # Vector 2x1 for Speed of Robot 1
        Speed_L2 = np.array([[self.vL2], [self.vR2]]) # Vector 2x1 for Speed of Robot 2 
        Speed_L3 = np.array([[self.vL3], [self.vR3]]) # Vector 2x1 for Speed of Robot 3
        Speed_L4 = np.array([[self.vL4], [self.vR4]]) # Vector 2x1 for Speed of Robot 4
        Speed_L5 = np.array([[self.vL5], [self.vR5]]) # Vector 2x1 for Speed of Robot 5
        Speed_L6 = np.array([[self.vL6], [self.vR6]]) # Vector 2x1 for Speed of Robot 6


        M1 = np.array([[S1[0]],[S1[1]]]).reshape(2,1) #2x1
        M2 = np.array([[S2[0]],[S2[1]]]).reshape(2,1) #2x1
        M3 = np.array([[S3[0]],[S3[1]]]).reshape(2,1) #2x1
        M4 = np.array([[S4[0]],[S4[1]]]).reshape(2,1) #2x1
        M5 = np.array([[S5[0]],[S5[1]]]).reshape(2,1) #2x1
        M6 = np.array([[S6[0]],[S6[1]]]).reshape(2,1) #2x1
    
        Speed_L1 = np.dot(Di, M1) # 2x1 (VL1, VR1)
        Speed_L2 = np.dot(Di, M2) # 2x1 (VL2, VR2)
        Speed_L3 = np.dot(Di, M3) # 2x1 (VL3, VR3)
        Speed_L4 = np.dot(Di, M4) # 2x1 (VL4, VR4)
        Speed_L5 = np.dot(Di, M5) # 2x1 (VL5, VR5)
        Speed_L6 = np.dot(Di, M6) # 2x1 (VL6, VR6)


        VL1 = float(Speed_L1[0])
        VR1 = float(Speed_L1[1])
        VL2 = float(Speed_L2[0])
        VR2 = float(Speed_L2[1])
        VL3 = float(Speed_L3[0])
        VR3 = float(Speed_L3[1])
        VL4 = float(Speed_L4[0])
        VR4 = float(Speed_L4[1])
        VL5 = float(Speed_L5[0])
        VR5 = float(Speed_L5[1])        
        VL6 = float(Speed_L6[0])
        VR6 = float(Speed_L6[1])
                

        
        " Publish Speed Commands to Robot 1 "
    
        msgl1 = Float32()    
        msgr1 = Float32()
        msgl1.data = VL1
        msgr1.data = VR1
        self.publisher_l1.publish(msgl1)
        self.publisher_r1.publish(msgr1)
        #self.get_logger().info('Publishing R1: "%s"' % msgr1.data)


        " Publish Speed Commands to Robot 2 "
        
        msgl2 = Float32()
        msgr2 = Float32()
        msgl2.data = VL2
        msgr2.data = VR2
        self.publisher_l2.publish(msgl2)
        self.publisher_r2.publish(msgr2)

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


        " Publish Speed Commands to Robot 6 "
        
        msgl6 = Float32()
        msgr6 = Float32()
        msgl6.data = VL6
        msgr6.data = VR6
        self.publisher_l6.publish(msgl6)
        self.publisher_r6.publish(msgr6)
        
    
def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
