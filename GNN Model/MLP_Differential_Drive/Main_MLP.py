"""
Consensus for Two Mobile Robots using Differential Drive Model, using MLP model that transforms relative pose of each robot
to control speed
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
distance = 6

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
        self.publisher_l1 = self.create_publisher(Float32, '/leftMotorSpeedrobot1', 10) #Change according to topic in child script,String to Float32
        self.publisher_r1 = self.create_publisher(Float32, '/rightMotorSpeedrobot1', 10) #Change according to topic in child script,String to Float32
        self.publisher_l2 = self.create_publisher(Float32, '/leftMotorSpeedrobot2', 10) #Change according to topic in child script,String to Float32
        self.publisher_r2 = self.create_publisher(Float32, '/rightMotorSpeedrobot2', 10) #Change according to topic in child script,String to Float32
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            0)

        " Counter Variables "
        self.i1 = 0
        self.i2 = 0
        self.count = 2
        self.j1 = 0
        self.j2 = 0
        self.loop = 0        
        
        "Parameters "
        self.k = 1 # Control Gain
        
        " Mobile Robot 1 Parameters "
        self.x1 = 10
        self.y1 = 12
        self.Theta1 = 0
        self.v1 = 1
        self.w1 = 1
        self.vL1 = 2
        self.vR1 = 2
        
        " Mobile Robot 2 Parameters "
        self.x2 = 5
        self.y2 = 7
        self.Theta2 = 0
        self.v2 = 2
        self.w2 = 2
        self.vL2 = 2 
        self.vR2 = 2
        

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
        
        
        " Calculate the Pose of Robot 2 w.r.t Robot 1 and Control input U1 "

        self.X1 = self.x2 - self.x1 # Relative Pose of Robot 2 wrt Robot 1 in Global frame for X coordinate of dimension 1x1
        self.Y1 = self.y2 -self.y1 # Relative Pose of Robot 2 wrt Robot 1 in Global frame for Y coordinate of dimension 1x1
        #self.U1 = u1 # Control input U1 in Global frame for robot 1 of dimension 2x1
        

        " Calculate the Pose of Robot 1 w.r.t Robot 2 and Control input U2 "

        self.X2 = self.x1 - self.x2 # Relative Pose of Robot 1 wrt Robot 2 in Global frame for X coordinate of dimension 1x1
        self.Y2 = self.y1 -self.y2 # Relative Pose of Robot 1 wrt Robot 2 in Global frame for Y coordinate of dimension 1x1
        #self.U2 = u2 # Control input U2 in Global frame for robot 2 of dimension 2x1

        " Transform Relative Pose from Global to Local Reference Frame "
        
        R1 = np.array([[math.cos(self.Theta1),math.sin(self.Theta1)],[-math.sin(self.Theta1),math.cos(self.Theta1)]]) #2x2
        R2 = np.array([[math.cos(self.Theta2),math.sin(self.Theta2)],[-math.sin(self.Theta2),math.cos(self.Theta2)]]) #2x2
        
        PoseG1 = np.array([[self.X1],[self.Y1]]) # Relative Pose of Robot 2 wrt Robot 1 in Global Frame of dimension 2x1
        PoseL1 = np.dot(R1, PoseG1) # Relative Pose of Robot 2 wrt Robot 2 in Local Frame of dimension 2x1 
        PoseG2 = np.array([[self.X2],[self.Y2]]) # Relative Pose of Robot 1 wrt Robot 1 in Global Frame of dimension 2x1
        PoseL2 = np.dot(R2, PoseG2) # Relative Pose of Robot 1 wrt Robot 2 in Local Frame of dimension 2x1 

                    
        " Calculate Speed inputs V1/2 using MLP "
        
        relative_pose_1 = [ PoseL1[0][0], PoseL1[1][0] ] # tensor data for MLP model
        relative_pose_2 = [ PoseL2[0][0], PoseL2[1][0] ] # tensor data for MLP model
        
        V1_predicted = MLP_Model.predict(relative_pose_1, loaded_model) # predict local control input u1, tensor
        V2_predicted = MLP_Model.predict(relative_pose_2, loaded_model) # predict local control input u2, tensor
        
        
        V1_predicted_np = np.array([[ V1_predicted[0][0] ], [ V1_predicted[0][1] ]]) # from tensor to numpy array for calculation
        V2_predicted_np = np.array([[ V2_predicted[0][0] ], [ V2_predicted[0][1] ]]) # from tensor to numpy array for calculation
        
        print(V1_predicted_np)
        print(V2_predicted_np)
        

        VL1 = float(V1_predicted[0][0])
        VR1 = float(V1_predicted[0][1])
        VL2 = float(V2_predicted[0][0])
        VR2 = float(V2_predicted[0][1])


        " Publish Speed Commands to Robot 1"

        msgl1 = Float32()    
        msgr1 = Float32()
        msgl1.data = VL1
        msgr1.data = VR1
        self.publisher_l1.publish(msgl1)
        self.publisher_r1.publish(msgr1)
        #self.get_logger().info('Publishing R1: "%s"' % msgr1.data)
 

        " Publish Speed Commands to Robot 2"

        msgl2 = Float32()
        msgr2 = Float32()
        msgl2.data = VL2
        msgr2.data = VR2
        self.publisher_l2.publish(msgl2)
        self.publisher_r2.publish(msgr2)

        #distance = self.x2 - self.x1
        #print(distance)                               
    
def main(args=None):
    print("Program Started")
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()
    print("Program Ended")


if __name__ == '__main__':
    main()
