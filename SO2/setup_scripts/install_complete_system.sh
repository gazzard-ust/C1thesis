#!/bin/bash
# Complete installation script for CO2 monitoring system

set -e  # Exit on error

echo "==========================================="
echo "CO2 Monitoring System Installer"
echo "==========================================="

# Check if we're on the laptop
if [ ! -d ~/Documents/SO2 ]; then
    echo "Error: ~/Documents/SO2 not found. Please run from laptop."
    exit 1
fi

cd ~/Documents/SO2

# Create all necessary directories
echo "Creating directory structure..."
mkdir -p co2_sensor_pkg
mkdir -p co2_logs
mkdir -p setup_scripts
mkdir -p test_scripts

echo "✓ Directories created"

# Check for conda environment
echo "Checking conda environment..."
if conda info --envs | grep -q c1thesis; then
    echo "✓ c1thesis environment found"
else
    echo "⚠ c1thesis environment not found"
    echo "Create it with: conda create -n c1thesis python=3.8"
fi

# Create a system status checker
cat > check_system.py << 'EOL'
#!/usr/bin/env python3
import subprocess
import sys

def check_ros2():
    try:
        result = subprocess.run(['ros2', 'topic', 'list'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_topics():
    try:
        result = subprocess.run(['ros2', 'topic', 'list'], 
                              capture_output=True, text=True)
        topics = result.stdout
        co2_topics = ['/co2_concentration', '/co2_sensor_data']
        found = [t for t in co2_topics if t in topics]
        return found
    except:
        return []

print("System Check:")
print("-" * 40)

if check_ros2():
    print("✓ ROS2 is accessible")
    topics = check_topics()
    if topics:
        print(f"✓ CO2 topics found: {topics}")
    else:
        print("⚠ CO2 topics not found (is TurtleBot publishing?)")
else:
    print("✗ ROS2 not found or not sourced")

EOL

chmod +x check_system.py

echo "✓ System checker created"
echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Copy test script to Raspberry Pi and run it"
echo "2. Run CO2 logger on laptop: ./run_co2_logger.sh"
echo "3. Check system status: python3 check_system.py"
