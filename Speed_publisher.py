"""

Publisher node speed of a single Mobile Robot Wheel

"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float32


k = 1
L = 0.1
d = 3

" Mobile Robot 1 Parameters "
x1 = 1
y1 = 1
Theta1 = 0
v1 = 1
w1 = 1
vL1 = 2
vR1 = 2
" Mobile Robot 1 Parameters "
x2 = 2
y2 = 2
Theta2 = 0
v2 = 2
w2 = 2
vL2 = 2 
vR2 = 2
 
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

    def listener_callback(self, msg):

        
        if msg.transforms[0].child_frame_id == 'robot1' or msg.transforms[0].child_frame_id == 'robot2' :  
            x1 = msg.transforms[0].transform.translation.x
            y1 = msg.transforms[0].transform.translation.y
            xr1 = msg.transforms[0].transform.rotation.x
            yr1 = msg.transforms[0].transform.rotation.y
            zr1 = msg.transforms[0].transform.rotation.z
            wr1 = msg.transforms[0].transform.rotation.w
            x2 = msg.transforms[0].transform.translation.x
            y2 = msg.transforms[0].transform.translation.y
            xr2 = msg.transforms[0].transform.rotation.x
            yr2 = msg.transforms[0].transform.rotation.y
            zr2 = msg.transforms[0].transform.rotation.z
            wr2 = msg.transforms[0].transform.rotation.w
            
            Theta1 = euler_from_quaternion(xr1,yr1,zr1,wr1)
            Theta2 = euler_from_quaternion(xr2,yr2,zr2,wr2)
            
            " Calculate Control inputs u1 and u2 "
            
            u1 = np.array([[ k*(x2-x1)],[k*(y2-y1)]]) # 2x1
            u2 = np.array([[ k*(x1-x2) ],[k*(y1-y2)]]) # 2x1
           
            " Calculate V1/W1 and V2/W2 "
            
            S1 = np.array([[v1], [w1]]) # 2x1
            G1 = np.array([[1,0], [0,1/L]])
            F1 = np.array([[math.cos(Theta1),math.sin(Theta1)],[-math.sin(Theta1),math.cos(Theta1)]])
            S1 = np.dot(np.dot(G1, F1), u1) # 2x1
           
            S2 = np.array([[v2], [w2]]) # 2x1
            G2 = np.array([[1,0], [0,1/L]])
            F2 = np.array([[math.cos(Theta2),math.sin(Theta2)],[-math.sin(Theta2),math.cos(Theta2)]])
            S2 = np.dot(np.dot(G2, F2), u2) # 2x1
            
            " Calculate VL1/VR1 and VL2/VR2 "
            
            D = np.array([[1/2,1/2],[-1/(2*d),1/(2*d)]])
            Di = np.linalg.inv(D)
    
            Speed_L1 = np.array([[vL1], [vR1]]) # Vector 2x1 for Speed of Robot 1
            Speed_L2 = np.array([[vL2], [vR2]]) # Vector 2x1 for Speed of Robot 2
            M1 = np.array([[v1], [w1]])
            M2 = np.array([[v2], [w2]])
            Speed_L1 = np.dot(Di, M1) # 2x1 (VL1, VR1)
            Speed_L2 = np.dot(Di, M2) # 2x1 (VL2, VR2)
            
            VL1 = Speed_L1[0]
            VR1 = Speed_L1[1]
            VL2 = Speed_L2[0]
            VR2 = Speed_L2[1]
            
            " Speed Commands to Robot 1"
            
            msgr1 = Float32()
            msgl1 = Float32()
            msgr1.data = 9.0
            msgl1.data = 9.0
            self.publisher_l1.publish(msgl1)
            self.publisher_r1.publish(msgr1)
            self.get_logger().info('Publishing R1: "%s"' % msgr1.data)
            
            " Speed Commands to Robot 2"
            msgr2 = Float32()
            msgl2 = Float32()
            msgr2.data = 9.0
            msgl2.data = 9.0
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
