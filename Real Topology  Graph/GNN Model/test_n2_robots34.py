"""

Consensus for 2 Robots, using MLP Model

Scene: Robot 3, Robot 4

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
import time

L = 1
d = 0.5

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
        self.publisher_l4 = self.create_publisher(Float32, '/leftMotorSpeedrobot4', 0) #Change according to topic in child script,String to Float32
        self.publisher_r4 = self.create_publisher(Float32, '/rightMotorSpeedrobot4', 0) #Change according to topic in child script,String to Float32
        self.publisher_l3 = self.create_publisher(Float32, '/leftMotorSpeedrobot3', 0) #Change according to topic in child script,String to Float32
        self.publisher_r3 = self.create_publisher(Float32, '/rightMotorSpeedrobot3',0) #Change according to topic in child script,String to Float32          
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            0)
        
        " Mobile Robot 4 Parameters "
        self.x4 = 14
        self.y4 = 15
        self.Theta4 = 0
        self.v4 = 4
        self.w4 = 4
        self.vL4 = 2
        self.vR4 = 2    
        
        " Mobile Robot 3 Parameters "
        self.x3 = 5
        self.y3 = 7
        self.Theta3 = 0
        self.v3 = 2
        self.w3 = 2
        self.vL3 = 2 
        self.vR3 = 2
        
    def listener_callback(self, msg):
        
        if  msg.transforms[0].child_frame_id == 'robot4' :
            
            self.x4 = msg.transforms[0].transform.translation.x
            self.y4 = msg.transforms[0].transform.translation.y
            self.xr4 = msg.transforms[0].transform.rotation.x
            self.yr4 = msg.transforms[0].transform.rotation.y
            self.zr4 = msg.transforms[0].transform.rotation.z
            self.wr4 = msg.transforms[0].transform.rotation.w
            self.Theta4 = euler_from_quaternion(self.xr4,self.yr4,self.zr4,self.wr4) 
            
        if msg.transforms[0].child_frame_id == 'robot3' :  
            self.x3 = msg.transforms[0].transform.translation.x
            self.y3 = msg.transforms[0].transform.translation.y
            self.xr3 = msg.transforms[0].transform.rotation.x
            self.yr3 = msg.transforms[0].transform.rotation.y
            self.zr3 = msg.transforms[0].transform.rotation.z
            self.wr3 = msg.transforms[0].transform.rotation.w
            self.Theta3 = euler_from_quaternion(self.xr3,self.yr3,self.zr3,self.wr3)             
   
           
            " Calculate Mx1, My1, ...... Mx6, My6 "
            
            Mx3 = self.x4 - self.x3
            My3 = self.y4 - self.y3
        
            Mx4 = self.x3 - self.x4
            My4 = self.y3 - self.y4             
           
            " Use MLP to Predict control inputs "
            
            relative_pose_3 = [ Mx3, My3 ] # tensor data for MLP model
            relative_pose_4 = [ Mx4, My4 ] # tensor data for MLP model
    
            u3_predicted = MLP_Model.predict(relative_pose_3, loaded_model) # predict control input u1, tensor
            u4_predicted = MLP_Model.predict(relative_pose_4, loaded_model) # predict control input u2, tensor
            
            u3_predicted_np = np.array([[ u3_predicted[0][0] ], [ u3_predicted[0][1] ]]) # from tensor to numpy array for calculation
            u4_predicted_np = np.array([[ u4_predicted[0][0] ], [ u4_predicted[0][1] ]]) # from tensor to numpy array for calculation
    
            " Calculate V1/W1, V2/W2, V3/W3, V4/W4, V5/W5, V6/W6 "
            
            S3 = np.array([[self.v3], [self.w3]]) #2x1
            G3 = np.array([[1,0], [0,1/L]]) #2x2
            R3 = np.array([[math.cos(self.Theta3),math.sin(self.Theta3)],[-math.sin(self.Theta3),math.cos(self.Theta3)]]) #2x2
            S3 = np.dot(np.dot(G3, R3), u3_predicted_np) #2x1        
    
            S4 = np.array([[self.v4], [self.w4]]) #2x1
            G4 = np.array([[1,0], [0,1/L]]) #2x2
            R4 = np.array([[math.cos(self.Theta4),math.sin(self.Theta4)],[-math.sin(self.Theta4),math.cos(self.Theta4)]]) #2x2
            S4 = np.dot(np.dot(G4, R4), u4_predicted_np) #2x1  
                            
            " Calculate VL1/VR1, VL2/VR2, VL3/VR3, VL4/VR4, VL5/VR5, VL6/VR6 "
        
            D = np.array([[1/2,1/2],[-1/(2*d),1/(2*d)]]) #2x2
            Di = np.linalg.inv(D) #2x2
    
            Speed_L3 = np.array([[self.vL3], [self.vR3]]) # Vector 2x1 for Speed of Robot 3
            Speed_L4 = np.array([[self.vL4], [self.vR4]]) # Vector 2x1 for Speed of Robot 4
      
    
            M3 = np.array([[S3[0]],[S3[1]]]).reshape(2,1) #2x1
            M4 = np.array([[S4[0]],[S4[1]]]).reshape(2,1) #2x1
    
            Speed_L3 = np.dot(Di, M3) # 2x1 (VL3, VR3)
            Speed_L4 = np.dot(Di, M4) # 2x1 (VL4, VR4)
    
            VL3 = float(Speed_L3[0])
            VR3 = float(Speed_L3[1])
            VL4 = float(Speed_L4[0])
            VR4 = float(Speed_L4[1])
           
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
     
                    
        
def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    time.sleep(5)
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
            

