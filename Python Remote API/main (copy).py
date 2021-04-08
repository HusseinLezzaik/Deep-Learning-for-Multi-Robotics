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
import sim
import time 


L = 1
d = 0.5
distance = 6

" Connecting to V-Rep "

sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19997,True,True,-500000,5) # Connect to CoppeliaSim


x_disp1 = np.array([[0], [1]])
y_disp1 = np.array([[0], [1]])
z_rot1 =  np.array([[0], [0]])


x_disp2 = np.array([[1], [2]])
y_disp2 = np.array([[1], [2]])
z_rot2 =  np.array([[180], [180]]) # Adjust angle for the robots orientation to face each other


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

        " Counter Variables "
        self.i1 = 0
        self.i2 = 0
        self.count = 2
        self.j1 = 0
        self.j2 = 0
        self.iter = 1        
        
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
            
        
        
        " Calculate Control inputs u1 and u2 "
            
        u1 = np.array([[ self.k*(self.x2-self.x1)],[self.k*(self.y2-self.y1)]]) # 2x1 
        u2 = np.array([[ self.k*(self.x1-self.x2)],[self.k*(self.y1-self.y2)]]) # 2x1
        
        " Calculate V1/W1 and V2/W2 "
            
        S1 = np.array([[self.v1], [self.w1]]) #2x1
        G1 = np.array([[1,0], [0,1/L]]) #2x2
        R1 = np.array([[math.cos(self.Theta1),math.sin(self.Theta1)],[-math.sin(self.Theta1),math.cos(self.Theta1)]]) #2x2
        S1 = np.dot(np.dot(G1, R1), u1) #2x1
        
           
        S2 = np.array([[self.v2], [self.w2]]) #2x1
        G2 = np.array([[1,0], [0,1/L]]) #2x2
        R2 = np.array([[math.cos(self.Theta2),math.sin(self.Theta2)],[-math.sin(self.Theta2),math.cos(self.Theta2)]]) #2x2
        S2 = np.dot(np.dot(G2, R2), u2) # 2x1
            
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
        
        
        distance = self.x2 - self.x1
        

        
        
        " Calculate the Pose of Robot 2 w.r.t Robot 1 and Control input U1 "
        
        self.X1 = self.x2 - self.x1 # Relative Pose of Robot 2 wrt Robot 1 in Global frame for X coordinate of dimension 1x1
        self.Y1 = self.y2 -self.y1 # Relative Pose of Robot 2 wrt Robot 1 in Global frame for Y coordinate of dimension 1x1
        self.U1 = u1 # Control input U1 in Global frame for robot 1 of dimension 2x1
        
        
        
        " Calculate the Pose of Robot 1 w.r.t Robot 2 and Control input U2 "
        
        self.X2 = self.x1 - self.x2 # Relative Pose of Robot 1 wrt Robot 2 in Global frame for X coordinate of dimension 1x1
        self.Y2 = self.y1 -self.y2 # Relative Pose of Robot 1 wrt Robot 2 in Global frame for Y coordinate of dimension 1x1
        self.U2 = u2 # Control input U2 in Global frame for robot 2 of dimension 2x1
        
        " Transform Control Input U1 from Global to Local Reference Frame "
        
        U1L = np.dot(R1, self.U1) # Control input of Robot 1 in Local Frame of dimension 2x1
        U2L = np.dot(R2, self.U2) # Control input of Robot 2 in Local Frame of dimension 2x1
        
        " Transform Relative Pose from Global to Local Reference Frame "
        
        PoseG1 = np.array([[self.X1],[self.Y1]]) # Relative Pose of Robot 2 wrt Robot 1 in Global Frame of dimension 2x1
        PoseL1 = np.dot(R1, PoseG1) # Relative Pose of Robot 2 wrt Robot 2 in Local Frame of dimension 2x1 
        PoseG2 = np.array([[self.X2],[self.Y2]]) # Relative Pose of Robot 1 wrt Robot 1 in Global Frame of dimension 2x1
        PoseL2 = np.dot(R2, PoseG2) # Relative Pose of Robot 1 wrt Robot 2 in Local Frame of dimension 2x1 
        
        " Publsih Speed Commands to Robot 1"
        
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
        
        
        " Write Values to CSV1 and CSV2 "
        
        if distance > 0.2:
            if self.count % 2 == 0:
                
                with open('robot1.csv', 'a', newline='') as f:
                    fieldnames = ['Data_X', 'Data_Y', 'Angle', 'Label_X', 'Label_Y']
                    thewriter = csv.DictWriter(f, fieldnames=fieldnames)
                
                    if self.i1 == 0:
                        thewriter.writeheader()
                        self.i1 = 1
                        
                    if self.j1 != 0:    
                        thewriter.writerow({'Data_X' : PoseL1[0][0], 'Data_Y' : PoseL1[1][0], 'Angle' : self.Theta1, 'Label_X' : U1L[0][0], 'Label_Y' : U1L[1][0]})
                        
                    if self.j1 == 0:
                        self.j1 = 1
                    
                with open('robot2.csv', 'a', newline='') as f:
                    fieldnames = ['Data_X', 'Data_Y', 'Angle', 'Label_X', 'Label_Y']
                    thewriter = csv.DictWriter(f, fieldnames=fieldnames)
            
                    if self.i2 == 0:
                        thewriter.writeheader()
                        self.i2 = 1
                    
                    if self.j2 != 0:
                        thewriter.writerow({'Data_X' : PoseL2[0][0], 'Data_Y' : PoseL2[1][0], 'Angle' : self.Theta2, 'Label_X' : U2L[0][0], 'Label_Y' : U2L[1][0]})
                    
                    if self.j2 == 0:
                        self.j2 = 1
            
        elif clientID!=-1:             
            print ('Connected to remote API server')
            for i in range(len(z_rot1)):
                for j in range(len(x_disp1)):
                    for k in range(len(y_disp1)):
                        print(" Simulation ", self.iter)
                        
                        # Stop Simulation
                        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)  
                
                        # Retrieve some handles:
                
                        ErrLocM1,LocM1 =sim.simxGetObjectHandle(clientID, 'robot1', sim.simx_opmode_oneshot_wait)
                        ErrLocM2,LocM2 =sim.simxGetObjectHandle(clientID, 'robot2#0', sim.simx_opmode_oneshot_wait)
                
                
                        ErrLoc1,Loc1 =sim.simxGetObjectPosition(clientID, LocM1, -1, sim.simx_opmode_oneshot_wait)
                        ErrLoc2,Loc2 =sim.simxGetObjectPosition(clientID, LocM2, -1, sim.simx_opmode_oneshot_wait)
                
                
                        ErrLocO1,OriRobo1 =sim.simxGetObjectOrientation(clientID,LocM1, -1, sim.simx_opmode_oneshot_wait)
                        ErrLocO2,OriRobo2 =sim.simxGetObjectOrientation(clientID,LocM2, -1, sim.simx_opmode_oneshot_wait)
                
                
                        OriRobo1[2] = ((z_rot1[i][0]*math.pi)/180)
                        OriRobo2[2] = ((z_rot2[i][0]*math.pi)/180)
                
                
                        # Set Robot Orientation
                
                        sim.simxSetObjectOrientation(clientID, LocM1, -1, OriRobo1, sim.simx_opmode_oneshot_wait) 
                        sim.simxSetObjectOrientation(clientID, LocM2, -1, OriRobo2, sim.simx_opmode_oneshot_wait)
                
                
                        Loc1[0] = x_disp1[j][0]
                        Loc2[0] = x_disp2[j][0]
                
                
                        Loc1[1] = y_disp1[k][0]
                        Loc2[1] = y_disp2[k][0]
                
                        # Set Robot Position
                
                        sim.simxSetObjectPosition(clientID, LocM1, -1, Loc1, sim.simx_opmode_oneshot)
                        sim.simxSetObjectPosition(clientID, LocM2, -1, Loc2, sim.simx_opmode_oneshot)
                
                        # Print Positions and Orientation
                
                        print("Robot1 Position:", Loc1)
                        print("Robot2 Position:", Loc2)
                
                        print("Robot1 Orientation:", OriRobo1)
                        print("Robot2 Orientation:", OriRobo2)
                        
                        # Start Simulation
                        sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)                        
                             
                
                        print("Simulation Running ...")
                        time.sleep(1)
                        
                        self.iter += 1 # Nb of Simulation Counter
                
                        # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
                        sim.simxGetPingTime(clientID)
 
            # Stop Simulation 
            sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
            # End Connection to V-Rep
            sim.simxFinish(clientID)

        else:
            print("Failed connecting to remote API server") 
        
        
        self.count += 1 # Counter to skip values while saving to csv file  
    
def main(args=None):
    print("Program Started")
    # Start the simulation:
    sim.simxStartSimulation(clientID,sim.simx_opmode_oneshot_wait)
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()
    print("Program Ended")


if __name__ == '__main__':
    main()