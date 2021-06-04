"""

Consensus Algorithm for 2 Robots using MLP Model

Scene: Robot 1, Robot 2

Inputs: Mx, My, Phix, Phiy
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
loaded_model = MLP_Model.ModelE()
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
        self.publisher_r1 = self.create_publisher(Float32, '/rightMotorSpeedrobot1',0) #Change according to topic in child script,String to Float32
        self.publisher_l2 = self.create_publisher(Float32, '/leftMotorSpeedrobot2', 0) #Change according to topic in child script,String to Float32
        self.publisher_r2 = self.create_publisher(Float32, '/rightMotorSpeedrobot2',0) #Change according to topic in child script,String to Float32          
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            0)
       
        " Timer Callback " 
        timer_period = 0.03  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0          

        " Parameters "
        self.t = 0 # Just to intialized Phix's and Phiy's
        
        " Initialize Phi's"
        self.Phix1 = 0 # 1x1
        self.Phiy1 = 0 # 1x1
        self.Phix2 = 0 # 1x1
        self.Phiy2 = 0 # 1x1        
        
        " Mobile Robot 1 Parameters "
        self.x1 = 0
        self.y1 = 0
        self.Theta1 = 0
        self.v1 = 0
        self.w1 = 0
        self.vL1 = 0
        self.vR1 = 0
        
        " Mobile Robot 2 Parameters "
        self.x2 = 0
        self.y2 = 0
        self.Theta2 = 0
        self.v2 = 0
        self.w2 = 0
        self.vL2 = 0 
        self.vR2 = 0
 
    def timer_callback(self):  
        
        " Calculate Mx1, My1, ...... Mx6, My6 "
        #Initialize Phi's
        if self.t ==0:
            self.Phix1 = 0 # 1x1
            self.Phiy1 = 0 # 1x1
            self.Phix2 = 0 # 1x1
            self.Phiy2 = 0 # 1x1
            self.t += 1          
        
        Mx1 = self.x2 - self.x1
        My1 = self.y2 - self.y1
    
        Mx2 = self.x1 - self.x2
        My2 = self.y1 - self.y2             
       
        " Use MLP to Predict control inputs "
        
        relative_pose_1 = [ Mx1, My1, self.Phix1, self.Phiy1 ] # tensor data for MLP model
        relative_pose_2 = [ Mx2, My2, self.Phix2, self.Phiy2 ] # tensor data for MLP model

        u1_predicted = MLP_Model.predict(relative_pose_1, loaded_model) # predict control input u1, tensor
        u2_predicted = MLP_Model.predict(relative_pose_2, loaded_model) # predict control input u2, tensor
        
        print(u1_predicted)
        
        self.Phix1 = 0.3* u2_predicted[0][0] # 1x1
        self.Phiy1 = 0.3* u2_predicted[0][1] # 1x1

        self.Phix2 = 0.3* u1_predicted[0][0] # 1x1
        self.Phiy2 = 0.3* u1_predicted[0][1] # 1x1          
        
        u1_predicted_np = 0.3* np.array([[u1_predicted[0][0] ], [ u1_predicted[0][1] ]]) # from tensor to numpy array for calculation
        u2_predicted_np = 0.3* np.array([[ u2_predicted[0][0] ], [ u2_predicted[0][1] ]]) # from tensor to numpy array for calculation

        " Calculate V1/W1, V2/W2, V3/W3, V4/W4, V5/W5, V6/W6 "
        
        S1 = np.array([[self.v1], [self.w1]]) #2x1
        G1 = np.array([[1,0], [0,1/L]]) #2x2
        R1 = np.array([[math.cos(self.Theta1),math.sin(self.Theta1)],[-math.sin(self.Theta1),math.cos(self.Theta1)]]) #2x2
        S1 = np.dot(np.dot(G1, R1), u1_predicted_np) #2x1
    
        S2 = np.array([[self.v2], [self.w2]]) #2x1
        G2 = np.array([[1,0], [0,1/L]]) #2x2
        R2 = np.array([[math.cos(self.Theta2),math.sin(self.Theta2)],[-math.sin(self.Theta2),math.cos(self.Theta2)]]) #2x2
        S2 = np.dot(np.dot(G2, R2), u2_predicted_np) # 2x1
                        
        " Calculate VL1/VR1, VL2/VR2, VL3/VR3, VL4/VR4, VL5/VR5, VL6/VR6 "
    
        D = np.array([[1/2,1/2],[-1/(2*d),1/(2*d)]]) #2x2
        Di = np.linalg.inv(D) #2x2

        Speed_L1 = np.array([[self.vL1], [self.vR1]]) # Vector 2x1 for Speed of Robot 1
        Speed_L2 = np.array([[self.vL2], [self.vR2]]) # Vector 2x1 for Speed of Robot 2
  

        M1 = np.array([[S1[0]],[S1[1]]]).reshape(2,1) #2x1
        M2 = np.array([[S2[0]],[S2[1]]]).reshape(2,1) #2x1

        Speed_L1 = np.dot(Di, M1) # 2x1 (VL1, VR1)
        Speed_L2 = np.dot(Di, M2) # 2x1 (VL2, VR2)

        VL1 = float(Speed_L1[0])
        VR1 = float(Speed_L1[1])
        VL2 = float(Speed_L2[0])
        VR2 = float(Speed_L2[1])
       
        " Publish Speed Commands to Robot 1 "
    
        msgl1 = Float32()    
        msgr1 = Float32()
        msgl1.data = VL1
        msgr1.data = VR1
        self.publisher_l1.publish(msgl1)
        self.publisher_r1.publish(msgr1)


        " Publish Speed Commands to Robot 2 "
        
        msgl2 = Float32()
        msgr2 = Float32()
        msgl2.data = VL2
        msgr2.data = VR2
        self.publisher_l2.publish(msgl2)
        self.publisher_r2.publish(msgr2)  
        
        self.i += 1

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
                               
        
def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    time.sleep(5)
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
            

