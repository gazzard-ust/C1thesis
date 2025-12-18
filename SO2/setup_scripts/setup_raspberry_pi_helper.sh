#!/bin/bash
# This script helps set up the Raspberry Pi
# Run this on your LAPTOP and it will generate commands for the Pi

echo "Generating setup commands for Raspberry Pi..."

cat > raspberry_pi_commands.txt << 'EOL'
# Commands to run on Raspberry Pi:

# 1. Enable UART
sudo raspi-config
# Navigate: Interface Options -> Serial Port -> No (login shell) -> Yes (hardware)

# 2. Add user to dialout group
sudo usermod -a -G dialout $USER

# 3. Install Python serial
pip3 install pyserial

# 4. Test serial port
ls -l /dev/tty*
# Look for /dev/ttyAMA0 or /dev/serial0

# 5. Reboot
sudo reboot

# After reboot, test with:
python3 -c "import serial; print('Serial module OK')"
EOL

echo "Commands saved to raspberry_pi_commands.txt"
echo "Copy these commands to your Raspberry Pi"
