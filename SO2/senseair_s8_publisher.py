import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Temperature
import serial
import time
import json
from datetime import datetime

class SenseAirS8Node(Node):
    def __init__(self):
        super().__init__('senseair_s8_node')

        # Publishers
        self.co2_pub = self.create_publisher(Float32, 'co2_concentration', 10)
        self.data_pub = self.create_publisher(String, 'co2_sensor_data', 10)

        # Serial port configuration for SenseAir S8
        self.declare_parameter('serial_port', '/dev/serial0')
        self.declare_parameter('baudrate', 9600)

        port = self.get_parameter('serial_port').value
        baudrate = self.get_parameter('baudrate').value

        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            self.get_logger().info(f'Connected to SenseAir S8 on {port}')
        except Exception as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise

        # Timer for periodic readings (every 2 seconds)
        self.timer = self.create_timer(2.0, self.read_sensor)

        # For response time calculation
        self.last_reading_time = None
        self.last_co2_value = None

    def read_sensor(self):
        """Read CO2 concentration from SenseAir S8"""
        try:
            # Use Modbus "Read Register" command for full 0–10000 ppm range
            # Command: FE 04 00 03 00 01 D5 C5
            read_command = bytes.fromhex('FE 04 00 03 00 01 D5 C5')

            # Clear input buffer
            self.serial.flushInput()

            # Send command and measure response time
            start_time = time.perf_counter()
            self.serial.write(read_command)

            # Read response (7 bytes)
            response = self.serial.read(7)
            response_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

            if len(response) == 7 and response[0] == 0xFE and response[1] == 0x04:
                raw_value = (response[3] << 8) | response[4]
                co2_ppm = round(raw_value / 10)

                # Calculate sensor response time (change detection)
                sensor_response_time = 0.0
                if self.last_co2_value is not None and self.last_co2_value != co2_ppm:
                    if self.last_reading_time is not None:
                        sensor_response_time = (time.time() - self.last_reading_time) * 1000

                # Update last values
                if co2_ppm != self.last_co2_value:
                    self.last_reading_time = time.time()
                self.last_co2_value = co2_ppm

                # Publish CO2 concentration
                co2_msg = Float32()
                co2_msg.data = float(co2_ppm)
                self.co2_pub.publish(co2_msg)

                # Publish detailed data as JSON
                data_msg = String()
                sensor_data = {
                    'timestamp': datetime.now().isoformat(),
                    'co2_ppm': co2_ppm,
                    'communication_response_time_ms': round(response_time, 2),
                    'sensor_response_time_ms': round(sensor_response_time, 2)
                }
                data_msg.data = json.dumps(sensor_data)
                self.data_pub.publish(data_msg)

                self.get_logger().info(f'CO2: {co2_ppm} ppm, Comm Response: {response_time:.2f}ms')

            else:
                self.get_logger().warning('Invalid response from sensor')

        except Exception as e:
            self.get_logger().error(f'Error reading sensor: {e}')

    def __del__(self):
        if hasattr(self, 'serial') and self.serial.is_open:
            self.serial.close()

def main(args=None):
    rclpy.init(args=args)
    node = SenseAirS8Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
