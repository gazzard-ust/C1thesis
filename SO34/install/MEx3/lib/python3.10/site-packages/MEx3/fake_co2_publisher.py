#!/usr/bin/env python3
"""
Fake CO2 Sensor Publisher
Publishes random CO2 concentration values between 420-440 ppm
for testing the Gazzard GUI without a real sensor.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random
import time


class FakeCO2Publisher(Node):
    def __init__(self):
        super().__init__('fake_co2_publisher')

        # Create publisher for CO2 concentration
        self.publisher = self.create_publisher(Float32, '/co2_concentration', 10)

        # Publish every 1 second
        self.timer = self.create_timer(1.0, self.publish_co2)

        self.get_logger().info('🔬 Fake CO2 Sensor Publisher started')
        self.get_logger().info('📊 Publishing random values between 420-440 ppm to /co2_concentration')

    def publish_co2(self):
        """Generate and publish random CO2 value between 420-440 ppm"""
        # Generate random value between 420 and 440
        co2_value = random.uniform(420.0, 440.0)

        # Create and publish message
        msg = Float32()
        msg.data = co2_value

        self.publisher.publish(msg)
        self.get_logger().info(f'📡 Published CO2: {co2_value:.1f} ppm')


def main(args=None):
    rclpy.init(args=args)

    fake_co2_publisher = FakeCO2Publisher()

    try:
        rclpy.spin(fake_co2_publisher)
    except KeyboardInterrupt:
        fake_co2_publisher.get_logger().info('🛑 Fake CO2 Publisher stopped')
    finally:
        fake_co2_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
