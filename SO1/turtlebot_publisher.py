#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time
import json
import socket

class TurtleBotPublisher(Node):
    def __init__(self):
        super().__init__('turtlebot_publisher')
        
        # Publisher for actual TurtleBot movement
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publisher for timing data (for latency measurement)
        self.timing_publisher = self.create_publisher(String, '/timing_data', 10)
        
        # Timer for publishing commands (20Hz for good AUV simulation)
        self.timer = self.create_timer(0.05, self.publish_commands)
        
        self.count = 0
        self.start_time = time.time()
        
        # Get TurtleBot's IP for identification
        self.robot_ip = self.get_local_ip()
        
        # Log file on TurtleBot
        self.log_file = open('/tmp/turtlebot_publisher.csv', 'w')
        self.log_file.write('Message_ID,Timestamp_ns,Robot_IP,Linear_X,Angular_Z\n')
        
        self.get_logger().info(f'🤖 TurtleBot Publisher Started on {self.robot_ip}')
        
    def get_local_ip(self):
        """Get TurtleBot's IP address"""
        try:
            # Connect to remote address to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "unknown"
    
    def publish_commands(self):
        # Create movement command
        cmd_msg = Twist()
        
        # TurtleBot movement pattern (safe indoor movement)
        if self.count < 100:  # Move forward
            cmd_msg.linear.x = 0.2  # 0.2 m/s forward
            cmd_msg.angular.z = 0.0
        elif self.count < 150:  # Turn
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.5  # Turn in place
        elif self.count < 200:  # Move forward again
            cmd_msg.linear.x = 0.2
            cmd_msg.angular.z = 0.0
        else:  # Stop and reset
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            if self.count >= 250:
                self.count = 0  # Reset pattern
        
        # Timestamp for latency measurement
        timestamp_ns = time.time_ns()
        
        # Create timing data message for latency analysis
        timing_data = {
            'message_id': self.count,
            'publish_timestamp_ns': timestamp_ns,
            'robot_ip': self.robot_ip,
            'linear_x': cmd_msg.linear.x,
            'angular_z': cmd_msg.angular.z
        }
        
        timing_msg = String()
        timing_msg.data = json.dumps(timing_data)
        
        # Publish both messages
        self.cmd_publisher.publish(cmd_msg)
        self.timing_publisher.publish(timing_msg)
        
        # Log data locally on TurtleBot
        self.log_file.write(f'{self.count},{timestamp_ns},{self.robot_ip},'
                           f'{cmd_msg.linear.x},{cmd_msg.angular.z}\n')
        self.log_file.flush()
        
        # Console output every 20 messages (1 second)
        if self.count % 20 == 0:
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f'🤖 TurtleBot: {self.count} msgs | Rate: {self.count/elapsed:.1f} Hz | '
                f'Linear: {cmd_msg.linear.x:.2f} m/s | Angular: {cmd_msg.angular.z:.2f} rad/s'
            )
        
        self.count += 1

def main():
    rclpy.init()
    node = TurtleBotPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n🛑 TurtleBot Publisher stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
