# 🤖 MEX3 - Reactive Navigation with YOLOWorld

## 📋 Overview

This project implements a reactive navigation system for a TurtleBot using YOLOWorld for real-time object detection and tracking. The system combines computer vision, depth estimation, and autonomous navigation to track and approach specified objects while maintaining safety protocols.

## ✨ Features

- **👁️ Real-time Object Detection**: Utilizes YOLOWorld for detecting and tracking various objects
- **📏 Depth Estimation**: Implements depth-anything model for accurate distance calculations
- **🎯 Reactive Navigation**: Intelligent movement control with centering and approach behaviors
- **🛡️ Safety Systems**: Emergency stop capabilities and collision avoidance
- **🌐 Web Interface**: User-friendly web interface for remote control and monitoring
- **📹 Live Video Feed**: Real-time camera and depth visualization
- **🔍 Search Behaviors**: Automated 360-degree search patterns when targets are lost

## 🏗️ System Architecture

### 🧩 Core Components

1. **📸 Image Publisher** (`image_publisher.py`)
   - Captures video from camera (V4L2)
   - Publishes compressed images to ROS topic
   - Handles camera configuration and error recovery

2. **🧠 Image Subscriber** (`image_subscriber.py`)
   - Main navigation controller
   - Object detection using YOLOWorld
   - Depth estimation and distance calculation
   - Movement control and safety systems
   - Web server for user interface

### 🛠️ Key Technologies

- **🔗 ROS 2**: Robot Operating System for communication
- **🌍 YOLOWorld**: Open-vocabulary object detection
- **📐 Depth-Anything**: Monocular depth estimation
- **👀 OpenCV**: Computer vision processing
- **⚡ FastAPI**: Web interface backend
- **🔥 PyTorch**: Deep learning framework

## 📦 Installation

### 📋 Prerequisites

- 🔧 ROS 2 (Humble/Iron recommended)
- 🐍 Python 3.8+
- 💻 CUDA-capable GPU (recommended)

### 📚 Dependencies

```bash
# Install ROS 2 dependencies
sudo apt install ros-<distro>-cv-bridge ros-<distro>-sensor-msgs

# Install Python packages
pip install torch torchvision ultralytics transformers fastapi uvicorn opencv-python pillow numpy
```

### 🔨 Build Instructions

```bash
# Clone the repository
git clone https://github.com/SuperMadee/MEX3---Reactive-Navigation-with-YOLOWorld.git
cd MEX3---Reactive-Navigation-with-YOLOWorld

# Build the ROS 2 package
colcon build --packages-select MEx3

# Source the workspace
source install/setup.bash
```

## 🚀 Usage

### ▶️ Running the System

1. **📸 Start the Image Publisher** (Camera Node):
```bash
ros2 run MEx3 image_publisher
```

2. **🧠 Start the Navigation System** (Main Controller):
```bash
ros2 run MEx3 image_subscriber
```

3. **🌐 Access the Web Interface**:
   - Open browser and navigate to `http://localhost:8000`
   - Enter target object name (e.g., "bottle", "chair", "cup")
   - Click "Set Target" to begin tracking

### 🎮 Web Interface Controls

- **🎯 Set Target**: Enter object name and start tracking
- **🔄 Reset**: Clear current target and return to initial state  
- **🛑 Emergency Stop**: Immediately halt all robot movement

### 🔌 API Endpoints

- `GET /set_target?object_name=<name>`: 🎯 Set tracking target
- `GET /reset`: 🔄 Reset system state
- `GET /stop`: 🛑 Emergency stop
- `GET /robot_status`: 📊 Get current robot status
- `GET /video_feed`: 📹 Live camera feed
- `GET /depth_feed`: 📐 Live depth visualization

## ⚙️ Configuration

### 🤖 Robot Parameters

```python
# Safety distances (in cm)
stopping_distance = 30.0
emergency_stop_distance = 20.0
min_safe_distance = 15.0

# Detection parameters
detection_confidence_threshold = 0.15
min_detection_frames = 3

# Movement parameters
max_angular_speed = 0.3
center_tolerance = 30  # pixels
```

### 📷 Camera Settings

```python
# Camera configuration
frame_width = 640
frame_height = 480
fps = 30
focal_length = 525
```

## 🛡️ Safety Features

- **🛑 Emergency Stop**: Immediate halt capability via web interface or API
- **📏 Distance Monitoring**: Continuous depth-based collision avoidance
- **⏰ Safety Timeouts**: Automatic stop if no commands received
- **✅ Detection Stability**: Multiple frame confirmation before movement
- **🐌 Controlled Approach**: Progressive speed reduction near targets

## 🔧 Troubleshooting

### ⚠️ Common Issues

1. **📷 Camera Not Found**:
   - Check camera connection and permissions
   - Verify V4L2 device availability: `ls /dev/video*`

2. **👁️ No Object Detection**:
   - Ensure adequate lighting
   - Check if object is in YOLOWorld vocabulary
   - Adjust detection confidence threshold

3. **🤖 Robot Not Moving**:
   - Verify `/cmd_vel` topic is properly connected
   - Check for emergency stop state
   - Ensure target is properly set

### 🐛 Debug Commands

```bash
# Check ROS topics
ros2 topic list
ros2 topic echo /cmd_vel

# Monitor image stream
ros2 topic echo /image/compressed

# Check node status
ros2 node list
ros2 node info /image_subscriber
```

## 💻 Development

### 📁 File Structure

```
MEx3/
├── MEx3/
│   ├── __init__.py
│   ├── image_publisher.py    # Camera capture and publishing
│   └── image_subscriber.py   # Main navigation controller
├── package.xml              # ROS package configuration
├── setup.py                # Python package setup
├── setup.cfg               # Package configuration
└── resource/
    └── MEx3                # Package resource marker
```

### 🔑 Key Classes and Methods

#### 📸 ImagePublisher
- `publish_image()`: Captures and publishes camera frames
- `destroy_node()`: Cleanup camera resources

#### 🧠 ImageSubscriber
- `image_callback()`: Main processing pipeline
- `process_detections()`: Object detection and tracking
- `control_movement()`: Movement control logic
- `search_behavior()`: Target search patterns
- `safety_check()`: Safety monitoring

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✏️ Make your changes
4. 🧪 Test thoroughly
5. 📤 Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **👨‍💻 Madee** - Initial development and implementation
- 📧 Email: mspangaliman@up.edu.ph

## 🙏 Acknowledgments

- 🌍 YOLOWorld team for open-vocabulary object detection
- 📐 Depth-Anything team for monocular depth estimation
- 🔗 ROS 2 community for robotics framework
- 🤖 TurtleBot community for robot platform

## 🚀 Future Enhancements

- [ ] 🎯 Multi-object tracking capabilities
- [ ] 🗺️ SLAM integration for mapping
- [ ] 📍 Path planning for complex environments
- [ ] 📱 Mobile app interface
- [ ] 🔍 Advanced search patterns
- [ ] 🧠 Machine learning-based behavior optimization

---

📚 For more information, visit the [project repository](https://github.com/SuperMadee/MEX3---Reactive-Navigation-with-YOLOWorld).
