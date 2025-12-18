#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time
import json
import statistics
import os
from collections import deque
import csv

class LaptopSubscriber(Node):
    def __init__(self):
        super().__init__('laptop_subscriber')
        
        # Subscribe to both command and timing data
        self.cmd_subscription = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)
        
        self.timing_subscription = self.create_subscription(
            String, '/timing_data', self.timing_callback, 10)
        
        # Data storage for analysis
        self.network_latencies = deque(maxlen=1000)  # Publisher to Subscriber latency
        self.processing_times = deque(maxlen=1000)   # Command processing time
        self.message_intervals = deque(maxlen=1000)  # Between messages
        
        self.last_cmd_time = None
        self.total_messages = 0
        self.robot_ip = "unknown"
        
        # CSV logging
        self.csv_file = open('laptop_analysis.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Message_ID', 'Robot_IP', 'Network_Latency_ms', 'Processing_Time_us',
            'Interval_ms', 'Linear_X', 'Angular_Z', 'Laptop_Timestamp'
        ])
        
        self.get_logger().info('💻 Laptop Subscriber Started - Monitoring TurtleBot')

    def timing_callback(self, msg):
        """Process timing data for latency calculation"""
        try:
            laptop_receive_time = time.time_ns()
            timing_data = json.loads(msg.data)
            
            # Calculate network latency (TurtleBot → Laptop)
            publish_time = timing_data['publish_timestamp_ns']
            network_latency_ms = (laptop_receive_time - publish_time) / 1e6
            
            self.network_latencies.append(network_latency_ms)
            self.robot_ip = timing_data.get('robot_ip', 'unknown')
            
            # Store for correlation with cmd_callback
            self.last_timing_data = {
                'latency': network_latency_ms,
                'message_id': timing_data['message_id'],
                'robot_ip': self.robot_ip
            }
            
        except Exception as e:
            self.get_logger().error(f'Timing processing error: {e}')

    def cmd_callback(self, msg):
        """Process command data and calculate processing time"""
        processing_start = time.time()
        laptop_timestamp = time.time_ns()
        
        # Simulate AUV command processing on laptop
        self.process_auv_command(msg)
        
        # Calculate processing time
        processing_end = time.time()
        processing_time_us = (processing_end - processing_start) * 1e6
        self.processing_times.append(processing_time_us)
        
        # Calculate message interval
        if self.last_cmd_time is not None:
            interval_ms = (laptop_timestamp - self.last_cmd_time) / 1e6
            self.message_intervals.append(interval_ms)
        
        self.total_messages += 1
        
        # Get network latency from timing data
        network_latency = 0
        message_id = 0
        if hasattr(self, 'last_timing_data'):
            network_latency = self.last_timing_data['latency']
            message_id = self.last_timing_data['message_id']
        
        # Log to CSV
        self.csv_writer.writerow([
            message_id, self.robot_ip, network_latency, processing_time_us,
            interval_ms if self.last_cmd_time else 0,
            msg.linear.x, msg.angular.z, laptop_timestamp
        ])
        self.csv_file.flush()
        
        # Real-time analysis display
        if self.total_messages % 20 == 0:
            self.display_network_analysis()
        
        self.last_cmd_time = laptop_timestamp

    def process_auv_command(self, msg):
        """Simulate AUV command processing on laptop"""
        # Simulate navigation calculations, path planning, etc.
        velocity_magnitude = (msg.linear.x**2 + msg.angular.z**2)**0.5
        
        # Simulate computational load
        time.sleep(0.0005)  # 0.5ms processing simulation

    def display_network_analysis(self):
        """Display real-time network and processing analysis"""
        if len(self.network_latencies) < 10:
            return
        
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Calculate statistics
        recent_latencies = list(self.network_latencies)[-50:]
        recent_processing = list(self.processing_times)[-50:]
        recent_intervals = list(self.message_intervals)[-50:]
        
        avg_latency = statistics.mean(recent_latencies)
        max_latency = max(recent_latencies)
        min_latency = min(recent_latencies)
        
        avg_processing = statistics.mean(recent_processing)
        avg_interval = statistics.mean(recent_intervals) if recent_intervals else 0
        frequency = 1000 / avg_interval if avg_interval > 0 else 0
        
        # Network performance assessment
        if avg_latency < 5:
            latency_status = "🟢 EXCELLENT"
        elif avg_latency < 20:
            latency_status = "🟡 GOOD"
        elif avg_latency < 50:
            latency_status = "🟠 ACCEPTABLE"
        else:
            latency_status = "🔴 POOR"
        
        # Display dashboard
        print("="*80)
        print("💻 LAPTOP ←→ 🤖 TURTLEBOT ROS NETWORK PERFORMANCE ANALYSIS")
        print("="*80)
        print(f"🤖 Robot IP: {self.robot_ip}")
        print(f"📊 Total Messages: {self.total_messages}")
        print(f"📡 Message Rate: {frequency:.1f} Hz")
        print()
        print("🌐 NETWORK LATENCY (TurtleBot → Laptop)")
        print("-"*60)
        print(f"  Average: {avg_latency:.2f} ms {latency_status}")
        print(f"  Minimum: {min_latency:.2f} ms")
        print(f"  Maximum: {max_latency:.2f} ms")
        print(f"  Jitter:  {statistics.stdev(recent_latencies):.2f} ms")
        print()
        print("⚡ LAPTOP PROCESSING PERFORMANCE")
        print("-"*60)
        print(f"  Average Processing: {avg_processing:.1f} μs")
        print(f"  Maximum Processing: {max(recent_processing):.1f} μs")
        print()
        
        # Network quality visualization
        print("📊 NETWORK LATENCY TREND (Last 20 messages)")
        print("-"*60)
        self.print_latency_graph(recent_latencies[-20:])
        
        print("="*80)
        print(f"💾 Data: laptop_analysis.csv | 🤖 Robot: {self.robot_ip}")
        print("Press Ctrl+C to stop monitoring")
        print("="*80)

    def print_latency_graph(self, latencies):
        """Print ASCII graph of latency"""
        if len(latencies) < 2:
            return
        
        min_lat, max_lat = min(latencies), max(latencies)
        if max_lat == min_lat:
            print(f"Stable latency: {min_lat:.2f} ms")
            return
        
        # Create simple bar chart
        for i, lat in enumerate(latencies):
            normalized = int((lat - min_lat) / (max_lat - min_lat) * 30)
            bar = "█" * normalized
            print(f"{i+1:2d}: {bar:<30} {lat:.1f}ms")

def main():
    rclpy.init()
    node = LaptopSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n🛑 Laptop Subscriber stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

