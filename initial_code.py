"""

Publisher node speed of a single Mobile Robot Wheel

"""


import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from std_msgs.msg import Float32


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher1')
        self.publisher1 = self.create_publisher(Float32, '/leftMotorSpeedrobot1', 10) #Change according to topic in child script,String to Float32
        self.publisher2 = self.create_publisher(Float32, '/rightMotorSpeedrobot1', 10) #Change according to topic in child script,String to Float32
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            10)
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0

    def listener_callback(self, msg):
        if msg.transforms[0].child_frame_id == 'robot2' :   
            self.get_logger().info('Subscribing: "%f"' % msg.transforms[0].transform.translation.z)
        msg = Float32()
        msg.data = 9.0  
        self.publisher1.publish(msg)
        self.publisher2.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        # self.i += 1
      


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
