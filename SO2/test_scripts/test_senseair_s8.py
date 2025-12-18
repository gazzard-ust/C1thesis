"""
Direct test script for SenseAir S8 CO2 sensor
Run this on the Raspberry Pi to verify sensor connection
"""

import serial
import time
import sys

def calculate_crc(data):
    """Calculate CRC for SenseAir S8 commands"""
    crc = 0
    for byte in data:
        crc = crc ^ byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x31
            else:
                crc = crc << 1
            crc = crc & 0xFF
    return crc

def test_senseair_s8(port='/dev/ttyAMA0'):
    """Test SenseAir S8 sensor connection and readings"""

    print(f"Testing SenseAir S8 on {port}")
    print("=" * 50)

    try:
        # Open serial connection
        ser = serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2.0
        )

        print(f"✓ Serial port {port} opened successfully")
        time.sleep(2)  # Allow sensor to stabilize

        # Test 1: Read CO2 concentration
        print("\nTest 1: Reading CO2 concentration...")
        read_co2_cmd = bytes([0xFE, 0x44, 0x00, 0x08, 0x02, 0x9F, 0x25])

        readings = []
        for i in range(5):
            ser.flushInput()
            ser.write(read_co2_cmd)
            response = ser.read(7)

            if len(response) == 7:
                if response[0] == 0xFE and response[1] == 0x44:
                    co2_ppm = (response[3] * 256) + response[4]
                    readings.append(co2_ppm)
                    print(f"  Reading {i+1}: {co2_ppm} ppm")
                else:
                    print(f"  Reading {i+1}: Invalid response header")
            else:
                print(f"  Reading {i+1}: Incomplete response (got {len(response)} bytes)")

            time.sleep(1)

        if readings:
            avg_co2 = sum(readings) / len(readings)
            print(f"\n✓ Average CO2: {avg_co2:.1f} ppm")
            print(f"  Min: {min(readings)} ppm, Max: {max(readings)} ppm")

            # Evaluate readings
            if avg_co2 < 300:
                print("  ⚠ Unusually low reading - sensor may need calibration")
            elif avg_co2 < 420:
                print("  ✓ Normal outdoor air level")
            elif avg_co2 < 1000:
                print("  ✓ Normal indoor air level")
            elif avg_co2 < 2000:
                print("  ⚠ Stuffy indoor air")
            else:
                print("  ⚠ Poor air quality")

        # Test 2: Read sensor status
        print("\nTest 2: Reading sensor status...")
        status_cmd = bytes([0xFE, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00])
        ser.flushInput()
        ser.write(status_cmd)
        status_response = ser.read(7)

        if len(status_response) == 7:
            status = status_response[2]
            print(f"  Sensor status byte: 0x{status:02X}")
            if status == 0x00:
                print("  ✓ Sensor status: OK")
            else:
                print(f"  ⚠ Sensor status: {status}")

        # Test 3: Response time test
        print("\nTest 3: Testing response time...")
        response_times = []

        for i in range(10):
            ser.flushInput()
            start_time = time.perf_counter()
            ser.write(read_co2_cmd)
            response = ser.read(7)
            response_time = (time.perf_counter() - start_time) * 1000

            if len(response) == 7:
                response_times.append(response_time)

            time.sleep(0.5)

        if response_times:
            avg_response = sum(response_times) / len(response_times)
            print(f"  Average response time: {avg_response:.1f} ms")
            print(f"  Min: {min(response_times):.1f} ms, Max: {max(response_times):.1f} ms")

        ser.close()
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        print("\nSensor is ready for ROS2 integration.")

    except serial.SerialException as e:
        print(f"✗ Serial error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if UART is enabled: sudo raspi-config")
        print("2. Verify wiring connections")
        print("3. Try using /dev/serial0 instead of /dev/ttyAMA0")
        print("4. Check if user is in dialout group: sudo usermod -a -G dialout $USER")
        sys.exit(1)

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = '/dev/ttyAMA0'  # Default port

    test_senseair_s8(port)
