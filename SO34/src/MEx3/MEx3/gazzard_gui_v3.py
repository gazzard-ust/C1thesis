import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
import uvicorn
import cv2
import numpy as np
import torch
from threading import Thread, Event
from ultralytics import YOLOWorld
from transformers import pipeline
from PIL import Image
import time
import os
from collections import deque
from enum import Enum

# ============================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ============================================

# Fine-tuned YOLO-World XL model path
MODEL_PATH = "best.pt"

# Fine-tuned classes (must match training exactly)
CRACK_CLASSES = [
    "Dummy crack",
    "Paper crack",
    "PVC pipe crack",
]

# Detection settings
DETECTION_CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45

# Depth estimation model
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# Robot safety parameters (in cm)
STOPPING_DISTANCE_CM = 30.0      # D_min - minimum distance to crack
EMERGENCY_STOP_CM = 20.0

# ============================================
# CO2-GUIDED LOCALIZATION PARAMETERS
# ============================================

# CO2 Thresholds
BASELINE_CO2 = 400.0             # Will be calibrated at startup
CO2_DEVIATION_THRESHOLD = 50.0   # δ - baseline deviation to trigger active mode (ppm)
CO2_GRADIENT_THRESHOLD = 10.0    # ε - gradient threshold for direction decisions (ppm)
CO2_HIGH_THRESHOLD = 800.0       # High CO2 level indicating leak source (ppm)
CO2_LEAK_CONFIRMED = 1000.0      # Confirmed leak level (ppm)

# Calibration
CALIBRATION_DURATION = 10.0      # Seconds to calibrate baseline CO2
CO2_SAMPLE_WINDOW = 2.0          # Seconds to average CO2 readings

# Movement parameters
MOVEMENT_STEP = 0.05             # Δd - small movement step (m/s)
ROTATION_STEP = 0.15             # Angular velocity for scanning (rad/s)
SCAN_ANGLES = 8                  # Number of directions to scan (360/8 = 45° each)
SCAN_DURATION_PER_ANGLE = 2.0    # Seconds per scan direction

# ============================================
# END CONFIGURATION
# ============================================

class RobotState(Enum):
    """Robot operational states"""
    CALIBRATING = "CALIBRATING"          # Initial CO2 baseline calibration
    STANDBY = "STANDBY"                  # Monitoring CO2, no movement
    ACTIVE_SCANNING = "ACTIVE_SCANNING"  # Rotating to find CO2 direction
    LOCALIZATION = "LOCALIZATION"        # Moving toward source, detecting cracks
    TARGET_CONFIRMED = "TARGET_CONFIRMED" # Both crack and high CO2 confirmed
    EMERGENCY_STOP = "EMERGENCY_STOP"

app = FastAPI()
node = None
target_object = "crack"

class GasGuidedCrackDetector(Node):
    def __init__(self):
        super().__init__('gas_guided_crack_detector')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # ============================================
        # STATE MACHINE
        # ============================================
        self.state = RobotState.CALIBRATING
        self.previous_state = None
        
        # ============================================
        # CO2 SENSING
        # ============================================
        self.co2_history = deque(maxlen=100)  # Rolling CO2 readings
        self.baseline_co2 = BASELINE_CO2
        self.current_co2 = 0.0
        self.co2_at_position_a = 0.0
        self.co2_at_position_b = 0.0
        self.co2_gradient = 0.0
        self.last_co2_time = None
        self.co2_connected = False
        
        # Calibration
        self.calibration_start_time = None
        self.calibration_readings = []
        
        # Directional CO2 scanning
        self.scan_co2_readings = {}  # {angle: co2_value}
        self.current_scan_angle = 0
        self.scan_start_time = None
        self.best_direction = 0.0
        
        # ============================================
        # CRACK DETECTION
        # ============================================
        self.crack_detected = False
        self.crack_distance = float('inf')
        self.crack_position = None  # (x, y) in frame
        self.current_detections = []
        self.detection_confidence_threshold = DETECTION_CONFIDENCE
        
        # ============================================
        # MOVEMENT CONTROL
        # ============================================
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.movement_command = "Initializing..."
        self.estimated_distance = 0.0
        
        # Localization tracking
        self.position_a_data = None  # {'co2': val, 'crack': bool, 'distance': val}
        self.position_b_data = None
        self.localization_step = 'A'  # 'A' or 'B'
        self.step_start_time = None
        
        # Safety
        self.emergency_stop = False
        self.target_confirmed = False
        self.confirmation_data = {}
        
        # ============================================
        # FRAME PROCESSING
        # ============================================
        self.latest_frame = None
        self.latest_frame_array = None
        self.latest_camera = None
        self.latest_depth = None
        self.last_depth_map = None
        self.frame_counter = 0
        self.process_every_n_frames = 2
        
        # Inference timing
        self.yolo_inference_time = 0.0
        self.depth_inference_time = 0.0
        self.total_inference_time = 0.0
        
        # Inference speed statistics (for benchmark/thesis)
        self.yolo_times = deque(maxlen=500)
        self.depth_times = deque(maxlen=500)
        self.total_times = deque(maxlen=500)
        self.fps_history = deque(maxlen=100)
        self.benchmark_start_time = None
        self.frames_processed = 0
        
        # Camera parameters
        self.frame_width = 640
        self.frame_center = self.frame_width // 2
        self.center_tolerance = 30
        
        # Initialize models and publishers
        self.initialize_models()
        self.initialize_publishers()
        
        # Start calibration timer
        self.calibration_start_time = time.time()
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🚀 GAS-GUIDED CRACK DETECTION SYSTEM INITIALIZED")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"📊 CO2 Thresholds: baseline_δ={CO2_DEVIATION_THRESHOLD}ppm, gradient_ε={CO2_GRADIENT_THRESHOLD}ppm")
        self.get_logger().info(f"📊 High CO2: {CO2_HIGH_THRESHOLD}ppm, Leak Confirmed: {CO2_LEAK_CONFIRMED}ppm")
        self.get_logger().info(f"🎯 Stopping Distance: {STOPPING_DISTANCE_CM}cm")
        self.get_logger().info(f"⏳ Starting {CALIBRATION_DURATION}s CO2 baseline calibration...")
        self.get_logger().info("=" * 60)

    def initialize_models(self):
        self.get_logger().info("🚀 Initializing Models...")
        
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"❌ Model not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        self.get_logger().info(f"📦 Loading fine-tuned model: {MODEL_PATH}")
        self.model = YOLOWorld(MODEL_PATH)
        self.model.set_classes(CRACK_CLASSES)
        self.get_logger().info(f"✅ YOLO-World XL loaded with classes: {CRACK_CLASSES}")
        
        self.get_logger().info(f"📦 Loading depth model: {DEPTH_MODEL}")
        self.depth_pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL)
        self.get_logger().info("✅ All Models Loaded Successfully!")

    def initialize_publishers(self):
        self.movement_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.subscription = self.create_subscription(
            CompressedImage, '/image/compressed', self.image_callback, 1
        )
        
        self.co2_subscription = self.create_subscription(
            Float32, '/co2_concentration', self.co2_callback, 10
        )
        
        self.frame_event = Event()
        
        # Safety timer
        self.safety_timer = self.create_timer(0.5, self.safety_check)
        
        # State machine timer
        self.state_timer = self.create_timer(0.1, self.state_machine_update)

    # ============================================
    # CO2 PROCESSING
    # ============================================
    
    def co2_callback(self, msg):
        """Process incoming CO2 sensor data"""
        try:
            self.current_co2 = float(msg.data)
            self.last_co2_time = time.time()
            self.co2_connected = True
            self.co2_history.append((time.time(), self.current_co2))
            
            # During calibration, collect readings
            if self.state == RobotState.CALIBRATING:
                self.calibration_readings.append(self.current_co2)
            
        except Exception as e:
            self.get_logger().error(f"❌ CO2 Callback Error: {e}")

    def get_averaged_co2(self, window_seconds=CO2_SAMPLE_WINDOW):
        """Get time-averaged CO2 reading over specified window"""
        if not self.co2_history:
            return self.baseline_co2
        
        current_time = time.time()
        recent_readings = [
            co2 for timestamp, co2 in self.co2_history 
            if current_time - timestamp <= window_seconds
        ]
        
        if recent_readings:
            return sum(recent_readings) / len(recent_readings)
        return self.current_co2

    def is_co2_elevated(self):
        """Check if CO2 is above baseline + threshold"""
        return self.get_averaged_co2() > (self.baseline_co2 + CO2_DEVIATION_THRESHOLD)

    def is_co2_high(self):
        """Check if CO2 indicates leak source proximity"""
        return self.get_averaged_co2() >= CO2_HIGH_THRESHOLD

    def is_leak_confirmed(self):
        """Check if CO2 level confirms a leak"""
        return self.get_averaged_co2() >= CO2_LEAK_CONFIRMED

    # ============================================
    # STATE MACHINE
    # ============================================

    def state_machine_update(self):
        """Main state machine logic - called periodically"""
        
        if self.emergency_stop:
            self.state = RobotState.EMERGENCY_STOP
        
        if self.state != self.previous_state:
            self.get_logger().info(f"🔄 State Change: {self.previous_state} → {self.state}")
            self.previous_state = self.state
        
        if self.state == RobotState.CALIBRATING:
            self.handle_calibration_state()
        elif self.state == RobotState.STANDBY:
            self.handle_standby_state()
        elif self.state == RobotState.ACTIVE_SCANNING:
            self.handle_active_scanning_state()
        elif self.state == RobotState.LOCALIZATION:
            self.handle_localization_state()
        elif self.state == RobotState.TARGET_CONFIRMED:
            self.handle_target_confirmed_state()
        elif self.state == RobotState.EMERGENCY_STOP:
            self.handle_emergency_stop_state()

    def handle_calibration_state(self):
        """Calibrate CO2 baseline"""
        elapsed = time.time() - self.calibration_start_time
        
        if elapsed < CALIBRATION_DURATION:
            self.movement_command = f"Calibrating CO2 baseline... {elapsed:.1f}/{CALIBRATION_DURATION}s"
            self.stop_robot()
        else:
            # Calculate baseline from calibration readings
            if self.calibration_readings:
                self.baseline_co2 = sum(self.calibration_readings) / len(self.calibration_readings)
                self.get_logger().info(f"✅ CO2 Baseline Calibrated: {self.baseline_co2:.1f} ppm")
            else:
                self.get_logger().warning("⚠️ No CO2 readings during calibration, using default baseline")
            
            # Transition to STANDBY
            self.state = RobotState.STANDBY
            self.movement_command = "Standby - Monitoring CO2"

    def handle_standby_state(self):
        """Monitor CO2, remain stationary until elevated levels detected"""
        self.stop_robot()
        
        avg_co2 = self.get_averaged_co2()
        self.movement_command = f"Standby | CO2: {avg_co2:.0f} ppm (baseline: {self.baseline_co2:.0f})"
        
        # Check for elevated CO2
        if self.is_co2_elevated():
            self.get_logger().info(f"⚠️ CO2 ELEVATED: {avg_co2:.0f} ppm > {self.baseline_co2 + CO2_DEVIATION_THRESHOLD:.0f} ppm")
            self.get_logger().info("🔄 Entering ACTIVE SCANNING MODE")
            
            # Reset scan data
            self.scan_co2_readings = {}
            self.current_scan_angle = 0
            self.scan_start_time = time.time()
            
            self.state = RobotState.ACTIVE_SCANNING

    def handle_active_scanning_state(self):
        """Rotate to find direction with highest CO2"""
        
        angle_duration = SCAN_DURATION_PER_ANGLE
        total_angles = SCAN_ANGLES
        
        elapsed = time.time() - self.scan_start_time
        current_angle_index = int(elapsed / angle_duration)
        
        if current_angle_index < total_angles:
            # Still scanning
            angle_degrees = current_angle_index * (360 / total_angles)
            self.movement_command = f"Scanning direction {current_angle_index + 1}/{total_angles} ({angle_degrees:.0f}°)"
            
            # Rotate and sample CO2
            move_cmd = Twist()
            move_cmd.angular.z = ROTATION_STEP
            move_cmd.linear.x = 0.0
            self.movement_pub.publish(move_cmd)
            self.angular_speed = ROTATION_STEP
            
            # Record CO2 at this angle
            angle_key = current_angle_index
            if angle_key not in self.scan_co2_readings:
                self.scan_co2_readings[angle_key] = []
            self.scan_co2_readings[angle_key].append(self.get_averaged_co2())
            
        else:
            # Scanning complete - find best direction
            self.stop_robot()
            
            # Average CO2 for each direction
            avg_readings = {}
            for angle, readings in self.scan_co2_readings.items():
                avg_readings[angle] = sum(readings) / len(readings) if readings else 0
            
            if avg_readings:
                best_angle = max(avg_readings, key=avg_readings.get)
                best_co2 = avg_readings[best_angle]
                
                self.get_logger().info(f"📊 Scan Results: Best direction = {best_angle * (360/SCAN_ANGLES):.0f}° with CO2 = {best_co2:.0f} ppm")
                
                # Rotate to best direction
                self.best_direction = best_angle * (360 / SCAN_ANGLES)
                
                # Transition to LOCALIZATION
                self.localization_step = 'A'
                self.step_start_time = time.time()
                self.position_a_data = None
                self.position_b_data = None
                
                self.state = RobotState.LOCALIZATION
                self.get_logger().info("🔄 Entering LOCALIZATION MODE")
            else:
                # No readings, go back to standby
                self.state = RobotState.STANDBY

    def handle_localization_state(self):
        """Move toward source, sampling CO2 and detecting cracks"""
        
        move_cmd = Twist()
        avg_co2 = self.get_averaged_co2()
        
        # ============================================
        # SAMPLE AT POSITION A
        # ============================================
        if self.localization_step == 'A':
            # Sample CO2 and run detection at position A
            self.position_a_data = {
                'co2': avg_co2,
                'crack_detected': self.crack_detected,
                'crack_distance': self.crack_distance if self.crack_detected else float('inf'),
                'time': time.time()
            }
            
            self.movement_command = f"Position A | CO2: {avg_co2:.0f} | Crack: {'YES' if self.crack_detected else 'NO'}"
            
            # Check for immediate confirmation at A
            if self.check_confirmation_conditions(self.position_a_data):
                return  # State changed to TARGET_CONFIRMED
            
            # Move forward by Δd
            move_cmd.linear.x = MOVEMENT_STEP
            move_cmd.angular.z = 0.0
            self.movement_pub.publish(move_cmd)
            self.linear_speed = MOVEMENT_STEP
            
            # Wait for movement, then sample B
            if time.time() - self.step_start_time > 1.5:
                self.localization_step = 'B'
                self.step_start_time = time.time()
        
        # ============================================
        # SAMPLE AT POSITION B
        # ============================================
        elif self.localization_step == 'B':
            # Sample CO2 and run detection at position B
            self.position_b_data = {
                'co2': avg_co2,
                'crack_detected': self.crack_detected,
                'crack_distance': self.crack_distance if self.crack_detected else float('inf'),
                'time': time.time()
            }
            
            self.movement_command = f"Position B | CO2: {avg_co2:.0f} | Crack: {'YES' if self.crack_detected else 'NO'}"
            
            # Check for immediate confirmation at B
            if self.check_confirmation_conditions(self.position_b_data):
                return  # State changed to TARGET_CONFIRMED
            
            # Calculate gradient
            if self.position_a_data:
                self.co2_gradient = self.position_b_data['co2'] - self.position_a_data['co2']
                
                self.get_logger().info(f"📊 CO2 Gradient: ΔC = {self.co2_gradient:.1f} ppm (A={self.position_a_data['co2']:.0f}, B={self.position_b_data['co2']:.0f})")
                
                # ============================================
                # DECISION LOGIC
                # ============================================
                
                if self.co2_gradient > CO2_GRADIENT_THRESHOLD:
                    # Positive gradient - continue moving forward
                    self.get_logger().info("📈 Positive gradient - continuing forward")
                    move_cmd.linear.x = MOVEMENT_STEP
                    self.movement_pub.publish(move_cmd)
                    
                    # Reset for next A-B cycle
                    self.localization_step = 'A'
                    self.step_start_time = time.time()
                    self.position_a_data = None
                    
                elif self.co2_gradient < -CO2_GRADIENT_THRESHOLD:
                    # Negative gradient - wrong direction, rescan
                    self.get_logger().info("📉 Negative gradient - rotating and rescanning")
                    self.stop_robot()
                    
                    # Reset scan data
                    self.scan_co2_readings = {}
                    self.current_scan_angle = 0
                    self.scan_start_time = time.time()
                    
                    self.state = RobotState.ACTIVE_SCANNING
                    
                else:
                    # Flat gradient - stop and rescan
                    self.get_logger().info("➡️ Flat gradient - stopping and rescanning")
                    self.stop_robot()
                    
                    # Reset scan data
                    self.scan_co2_readings = {}
                    self.scan_start_time = time.time()
                    
                    self.state = RobotState.ACTIVE_SCANNING

    def check_confirmation_conditions(self, position_data):
        """
        Check if both conditions are met:
        1. Crack detected AND distance ≤ D_min
        2. CO2 > baseline + δ (preferably high/leak level)
        
        Returns True if target confirmed, False otherwise
        """
        crack_close = position_data['crack_detected'] and position_data['crack_distance'] <= STOPPING_DISTANCE_CM
        co2_elevated = position_data['co2'] > (self.baseline_co2 + CO2_DEVIATION_THRESHOLD)
        co2_high = position_data['co2'] >= CO2_HIGH_THRESHOLD
        
        if crack_close and co2_elevated:
            self.get_logger().info("=" * 60)
            self.get_logger().info("🎯 TARGET CONFIRMATION CONDITIONS MET!")
            self.get_logger().info(f"   ✅ Crack detected at {position_data['crack_distance']:.1f} cm (≤ {STOPPING_DISTANCE_CM} cm)")
            self.get_logger().info(f"   ✅ CO2 = {position_data['co2']:.0f} ppm (> {self.baseline_co2 + CO2_DEVIATION_THRESHOLD:.0f} ppm)")
            if co2_high:
                self.get_logger().info(f"   🔥 HIGH CO2 LEVEL - Likely leak source!")
            self.get_logger().info("=" * 60)
            
            self.confirmation_data = {
                'crack_distance': position_data['crack_distance'],
                'co2_level': position_data['co2'],
                'timestamp': time.time(),
                'co2_status': 'LEAK_CONFIRMED' if position_data['co2'] >= CO2_LEAK_CONFIRMED else ('HIGH' if co2_high else 'ELEVATED')
            }
            
            self.target_confirmed = True
            self.state = RobotState.TARGET_CONFIRMED
            return True
        
        return False

    def handle_target_confirmed_state(self):
        """Target confirmed - crack and high CO2 at same location"""
        self.stop_robot()
        
        co2_status = self.confirmation_data.get('co2_status', 'ELEVATED')
        distance = self.confirmation_data.get('crack_distance', 0)
        co2_level = self.confirmation_data.get('co2_level', 0)
        
        self.movement_command = f"🎯 LEAK FOUND | CO2: {co2_level:.0f} ppm | Distance: {distance:.1f} cm"
        
        # Log periodically
        if self.frame_counter % 100 == 0:
            self.get_logger().info(f"🎯 TARGET CONFIRMED - Crack at {distance:.1f}cm with CO2 {co2_level:.0f}ppm ({co2_status})")

    def handle_emergency_stop_state(self):
        """Emergency stop - complete halt"""
        self.stop_robot()
        self.movement_command = "⚠️ EMERGENCY STOP"

    # ============================================
    # MOVEMENT CONTROL
    # ============================================

    def stop_robot(self):
        """Immediately stop all robot movement"""
        move_cmd = Twist()
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = 0.0
        self.movement_pub.publish(move_cmd)
        self.linear_speed = 0.0
        self.angular_speed = 0.0

    def safety_check(self):
        """Periodic safety check"""
        # Check CO2 sensor connection
        if self.last_co2_time:
            if time.time() - self.last_co2_time > 5.0:
                self.co2_connected = False
                if self.state not in [RobotState.CALIBRATING, RobotState.EMERGENCY_STOP]:
                    self.get_logger().warning("⚠️ CO2 sensor disconnected!")

    # ============================================
    # IMAGE PROCESSING
    # ============================================

    def image_callback(self, msg):
        """Process incoming camera images"""
        start_time = time.time()
        
        try:
            # Decode image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None or frame.size == 0:
                return
            
            self.latest_frame_array = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare display frame
            frame_display = frame.copy()
            
            # Draw detections
            if self.current_detections:
                for det in self.current_detections:
                    x1, y1, x2, y2 = det['coordinates']
                    color = (0, 0, 255) if det.get('is_target', False) else (0, 255, 0)
                    cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{det['label']} {det['conf']:.2f}"
                    cv2.putText(frame_display, label, (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add status overlay
            self.add_status_overlay(frame_display)
            
            # Encode for streaming
            ret, jpeg = cv2.imencode('.jpg', frame_display, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            self.latest_frame = jpeg.tobytes()
            self.latest_camera = jpeg.tobytes()
            
            # Update depth visualization
            if self.last_depth_map is not None:
                depth_vis = self.create_depth_visualization(self.last_depth_map)
                ret, depth_jpeg = cv2.imencode('.jpg', depth_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                self.latest_depth = depth_jpeg.tobytes()
            
            self.frame_event.set()
            
            # Process AI every N frames
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames == 0:
                # YOLO detection
                yolo_start = time.time()
                results = self.model.predict(frame_rgb, conf=self.detection_confidence_threshold, 
                                            iou=IOU_THRESHOLD, verbose=False)
                self.process_detections(frame, results)
                self.yolo_inference_time = time.time() - yolo_start
                
                # Depth estimation
                depth_start = time.time()
                self.last_depth_map = self.estimate_depth(frame_rgb)
                self.depth_inference_time = time.time() - depth_start
                
                self.total_inference_time = self.yolo_inference_time + self.depth_inference_time
                
                # Record statistics for benchmark
                self.yolo_times.append(self.yolo_inference_time * 1000)
                self.depth_times.append(self.depth_inference_time * 1000)
                self.total_times.append(self.total_inference_time * 1000)
                self.frames_processed += 1
                
                if self.total_inference_time > 0:
                    self.fps_history.append(1.0 / self.total_inference_time)
                
                if self.benchmark_start_time is None:
                    self.benchmark_start_time = time.time()
                
        except Exception as e:
            self.get_logger().error(f"❌ Image Processing Error: {e}")

    def process_detections(self, frame, results):
        """Process YOLO detections and update crack status"""
        self.current_detections = []
        self.crack_detected = False
        self.crack_distance = float('inf')
        self.crack_position = None
        
        best_crack = None
        best_confidence = 0
        
        for r in results:
            if not hasattr(r.boxes, 'xyxy') or len(r.boxes.xyxy) == 0:
                continue
            
            for i in range(len(r.boxes.xyxy)):
                box = r.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                conf = float(r.boxes.conf[i])
                cls_id = int(r.boxes.cls[i])
                
                # Get label
                label = CRACK_CLASSES[cls_id] if cls_id < len(CRACK_CLASSES) else f"Class {cls_id}"
                
                # Check if this is a crack (any of our classes)
                is_crack = any(crack_class.lower() in label.lower() for crack_class in CRACK_CLASSES)
                
                self.current_detections.append({
                    'coordinates': (x1, y1, x2, y2),
                    'label': label,
                    'conf': conf,
                    'is_target': is_crack
                })
                
                # Track best crack detection
                if is_crack and conf > best_confidence:
                    best_confidence = conf
                    best_crack = {
                        'x': (x1 + x2) // 2,
                        'y': (y1 + y2) // 2,
                        'box': (x1, y1, x2, y2),
                        'conf': conf,
                        'label': label
                    }
        
        # Update crack status
        if best_crack:
            self.crack_detected = True
            self.crack_position = (best_crack['x'], best_crack['y'])
            
            # Calculate distance if depth map available
            if self.last_depth_map is not None:
                self.crack_distance = self.calculate_depth_at_point(
                    best_crack['x'], best_crack['y'], self.last_depth_map
                )
                self.estimated_distance = self.crack_distance

    def calculate_depth_at_point(self, x, y, depth_map):
        """Calculate depth at a specific point"""
        try:
            h, w = depth_map.shape[:2]
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            
            # Sample region around point
            region_size = 15
            y_min = max(0, y - region_size)
            y_max = min(h - 1, y + region_size)
            x_min = max(0, x - region_size)
            x_max = min(w - 1, x + region_size)
            
            region_depths = depth_map[y_min:y_max, x_min:x_max] * 100  # Convert to cm
            valid_depths = region_depths[(region_depths > 20) & (region_depths < 500)]
            
            if len(valid_depths) > 0:
                return float(np.median(valid_depths))
            return 100.0
            
        except Exception as e:
            self.get_logger().error(f"Depth calculation error: {e}")
            return 100.0

    def estimate_depth(self, frame_rgb):
        """Estimate depth using depth model"""
        try:
            image_pil = Image.fromarray(frame_rgb)
            depth_result = self.depth_pipe(image_pil)
            return np.array(depth_result['depth'])
        except Exception as e:
            self.get_logger().error(f"Depth Estimation Error: {e}")
            return np.zeros((480, 640), dtype=np.float32)

    def add_status_overlay(self, frame):
        """Add comprehensive status overlay to frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        y_offset = 25
        line_height = 22
        
        # State
        state_color = {
            RobotState.CALIBRATING: (255, 255, 0),
            RobotState.STANDBY: (0, 255, 255),
            RobotState.ACTIVE_SCANNING: (255, 165, 0),
            RobotState.LOCALIZATION: (0, 255, 0),
            RobotState.TARGET_CONFIRMED: (0, 255, 0),
            RobotState.EMERGENCY_STOP: (0, 0, 255),
        }.get(self.state, (255, 255, 255))
        
        cv2.putText(frame, f"State: {self.state.value}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        y_offset += line_height
        
        # CO2
        co2_color = (0, 255, 0) if self.current_co2 < CO2_HIGH_THRESHOLD else (0, 0, 255)
        cv2.putText(frame, f"CO2: {self.current_co2:.0f} ppm (base: {self.baseline_co2:.0f})", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, co2_color, 2)
        y_offset += line_height
        
        # Crack detection
        crack_text = f"Crack: {'DETECTED' if self.crack_detected else 'None'}"
        if self.crack_detected:
            crack_text += f" @ {self.crack_distance:.1f}cm"
        crack_color = (0, 255, 0) if self.crack_detected else (128, 128, 128)
        cv2.putText(frame, crack_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, crack_color, 2)
        y_offset += line_height
        
        # Movement
        cv2.putText(frame, f"Speed: L={self.linear_speed:.2f} A={self.angular_speed:.2f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += line_height
        
        # Status message
        cv2.putText(frame, self.movement_command[:50], (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        y_offset += line_height
        
        # Inference time
        cv2.putText(frame, f"Inference: {self.total_inference_time*1000:.1f}ms", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # Target confirmed indicator
        if self.state == RobotState.TARGET_CONFIRMED:
            cv2.rectangle(frame, (w//2 - 150, h//2 - 30), (w//2 + 150, h//2 + 30), (0, 255, 0), 3)
            cv2.putText(frame, "LEAK SOURCE FOUND!", (w//2 - 130, h//2 + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def create_depth_visualization(self, depth_map):
        """Create colorized depth visualization"""
        try:
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            if self.latest_frame_array is not None:
                target_h, target_w = self.latest_frame_array.shape[:2]
                depth_vis = cv2.resize(depth_vis, (target_w, target_h))
            
            return depth_vis
        except:
            return np.zeros((480, 640, 3), dtype=np.uint8)


# ============================================
# FASTAPI ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return get_html_page()

@app.get("/camera_feed")
def camera_feed():
    def generate():
        while True:
            if node and node.latest_camera:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + node.latest_camera + b'\r\n')
            time.sleep(0.033)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/depth_feed")
def depth_feed():
    def generate():
        while True:
            if node and node.latest_depth:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + node.latest_depth + b'\r\n')
            time.sleep(0.033)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/gas_concentration")
async def get_gas_concentration():
    if node:
        return {
            "connected": node.co2_connected,
            "concentration": node.current_co2,
            "baseline": node.baseline_co2,
            "elevated": node.is_co2_elevated(),
            "high": node.is_co2_high(),
            "unit": "ppm"
        }
    return {"connected": False, "concentration": 0, "unit": "ppm"}

@app.get("/robot_status")
async def robot_status():
    if node:
        return {
            "state": node.state.value,
            "co2": {
                "current": node.current_co2,
                "baseline": node.baseline_co2,
                "gradient": node.co2_gradient,
                "elevated": node.is_co2_elevated(),
                "high": node.is_co2_high()
            },
            "crack": {
                "detected": node.crack_detected,
                "distance": node.crack_distance,
                "position": node.crack_position
            },
            "movement": {
                "linear_speed": node.linear_speed,
                "angular_speed": node.angular_speed,
                "command": node.movement_command
            },
            "target_confirmed": node.target_confirmed,
            "confirmation_data": node.confirmation_data,
            "inference_times": {
                "yolo_ms": round(node.yolo_inference_time * 1000, 1),
                "depth_ms": round(node.depth_inference_time * 1000, 1),
                "total_ms": round(node.total_inference_time * 1000, 1)
            }
        }
    return {"state": "NOT_INITIALIZED"}

@app.get("/start")
async def start_search():
    """Start the gas-guided search"""
    if node:
        if node.state == RobotState.TARGET_CONFIRMED:
            node.target_confirmed = False
            node.confirmation_data = {}
        
        node.state = RobotState.STANDBY
        node.emergency_stop = False
        return {"message": "Search started - monitoring CO2 levels"}
    return {"message": "Node not initialized"}

@app.get("/stop")
async def stop_robot():
    """Emergency stop"""
    if node:
        node.emergency_stop = True
        node.state = RobotState.EMERGENCY_STOP
        node.stop_robot()
        return {"message": "Emergency stop activated"}
    return {"message": "Node not initialized"}

@app.get("/reset")
async def reset_system():
    """Reset the system to initial state"""
    if node:
        node.emergency_stop = False
        node.target_confirmed = False
        node.confirmation_data = {}
        node.co2_gradient = 0.0
        node.position_a_data = None
        node.position_b_data = None
        node.stop_robot()
        
        # Recalibrate
        node.calibration_start_time = time.time()
        node.calibration_readings = []
        node.state = RobotState.CALIBRATING
        
        return {"message": "System reset - recalibrating CO2 baseline"}
    return {"message": "Node not initialized"}

@app.get("/recalibrate")
async def recalibrate_co2():
    """Recalibrate CO2 baseline"""
    if node:
        node.calibration_start_time = time.time()
        node.calibration_readings = []
        node.state = RobotState.CALIBRATING
        return {"message": f"Recalibrating CO2 baseline for {CALIBRATION_DURATION}s"}
    return {"message": "Node not initialized"}


# ============================================
# BENCHMARK ENDPOINTS (for thesis/paper)
# ============================================

@app.get("/benchmark")
async def get_benchmark_results():
    """
    Get comprehensive inference speed benchmark results.
    
    USE THIS FOR YOUR THESIS - provides:
    - Mean, Std, Min, Max, Median for each component
    - Ready-to-cite format: "XX.XX ± YY.YY ms"
    
    Run system for 30-60 seconds before calling this endpoint.
    """
    if node and len(node.yolo_times) > 0:
        # Calculate YOLO statistics
        yolo_times = list(node.yolo_times)
        yolo_mean = np.mean(yolo_times)
        yolo_std = np.std(yolo_times)
        yolo_min = np.min(yolo_times)
        yolo_max = np.max(yolo_times)
        yolo_median = np.median(yolo_times)
        
        # Calculate Depth statistics
        depth_times = list(node.depth_times)
        depth_mean = np.mean(depth_times)
        depth_std = np.std(depth_times)
        depth_min = np.min(depth_times)
        depth_max = np.max(depth_times)
        depth_median = np.median(depth_times)
        
        # Calculate Total statistics
        total_times = list(node.total_times)
        total_mean = np.mean(total_times)
        total_std = np.std(total_times)
        total_min = np.min(total_times)
        total_max = np.max(total_times)
        total_median = np.median(total_times)
        
        # Calculate FPS statistics
        fps_list = list(node.fps_history)
        fps_mean = np.mean(fps_list) if fps_list else 0
        fps_std = np.std(fps_list) if fps_list else 0
        fps_min = np.min(fps_list) if fps_list else 0
        fps_max = np.max(fps_list) if fps_list else 0
        
        # Calculate elapsed time
        elapsed = time.time() - node.benchmark_start_time if node.benchmark_start_time else 0
        
        return {
            "benchmark_info": {
                "frames_processed": node.frames_processed,
                "elapsed_time_sec": round(elapsed, 2),
                "samples_collected": len(yolo_times),
                "device": node.device,
                "model": MODEL_PATH,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "yolo_world_xl_ms": {
                "mean": round(yolo_mean, 2),
                "std": round(yolo_std, 2),
                "min": round(yolo_min, 2),
                "max": round(yolo_max, 2),
                "median": round(yolo_median, 2)
            },
            "depth_anything_v2_ms": {
                "mean": round(depth_mean, 2),
                "std": round(depth_std, 2),
                "min": round(depth_min, 2),
                "max": round(depth_max, 2),
                "median": round(depth_median, 2)
            },
            "total_pipeline_ms": {
                "mean": round(total_mean, 2),
                "std": round(total_std, 2),
                "min": round(total_min, 2),
                "max": round(total_max, 2),
                "median": round(total_median, 2)
            },
            "fps": {
                "mean": round(fps_mean, 2),
                "std": round(fps_std, 2),
                "min": round(fps_min, 2),
                "max": round(fps_max, 2)
            },
            "for_thesis": {
                "yolo_world_xl": f"{yolo_mean:.2f} ± {yolo_std:.2f} ms",
                "depth_anything_v2": f"{depth_mean:.2f} ± {depth_std:.2f} ms",
                "total_pipeline": f"{total_mean:.2f} ± {total_std:.2f} ms",
                "throughput": f"{fps_mean:.2f} ± {fps_std:.2f} FPS"
            }
        }
    return {"message": "No benchmark data yet. Run the system for a few seconds first."}


@app.get("/benchmark/save")
async def save_benchmark_results():
    """
    Save benchmark results to a JSON file.
    File saved as: inference_benchmark_YYYYMMDD_HHMMSS.json
    """
    import json
    
    if node and len(node.yolo_times) > 0:
        yolo_times = list(node.yolo_times)
        depth_times = list(node.depth_times)
        total_times = list(node.total_times)
        fps_list = list(node.fps_history)
        
        results = {
            "benchmark_info": {
                "frames_processed": node.frames_processed,
                "elapsed_time_sec": round(time.time() - node.benchmark_start_time, 2) if node.benchmark_start_time else 0,
                "samples_collected": len(yolo_times),
                "device": node.device,
                "model": MODEL_PATH,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "yolo_world_xl_ms": {
                "mean": round(float(np.mean(yolo_times)), 2),
                "std": round(float(np.std(yolo_times)), 2),
                "min": round(float(np.min(yolo_times)), 2),
                "max": round(float(np.max(yolo_times)), 2),
                "median": round(float(np.median(yolo_times)), 2)
            },
            "depth_anything_v2_ms": {
                "mean": round(float(np.mean(depth_times)), 2),
                "std": round(float(np.std(depth_times)), 2),
                "min": round(float(np.min(depth_times)), 2),
                "max": round(float(np.max(depth_times)), 2),
                "median": round(float(np.median(depth_times)), 2)
            },
            "total_pipeline_ms": {
                "mean": round(float(np.mean(total_times)), 2),
                "std": round(float(np.std(total_times)), 2),
                "min": round(float(np.min(total_times)), 2),
                "max": round(float(np.max(total_times)), 2),
                "median": round(float(np.median(total_times)), 2)
            },
            "fps": {
                "mean": round(float(np.mean(fps_list)), 2) if fps_list else 0,
                "std": round(float(np.std(fps_list)), 2) if fps_list else 0
            }
        }
        
        filename = f"inference_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        return {"message": f"Saved to {filename}", "results": results}
    return {"message": "No benchmark data yet."}


@app.get("/benchmark/reset")
async def reset_benchmark():
    """Reset benchmark statistics to start fresh measurement."""
    if node:
        node.yolo_times.clear()
        node.depth_times.clear()
        node.total_times.clear()
        node.fps_history.clear()
        node.frames_processed = 0
        node.benchmark_start_time = time.time()
        return {"message": "Benchmark reset. Statistics cleared."}
    return {"message": "Node not initialized"}


def get_html_page():
    """Generate the web interface HTML"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gazzard - Gas-Guided Crack Detection</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                color: white;
                padding: 20px;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            
            .header {
                text-align: center;
                padding: 20px;
                margin-bottom: 20px;
                background: rgba(255,255,255,0.1);
                border-radius: 15px;
            }
            .header h1 { font-size: 2em; color: #FFD700; }
            .header p { color: #87CEEB; }
            
            .status-bar {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }
            .status-card {
                background: rgba(255,255,255,0.1);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            .status-card h3 { font-size: 0.9em; color: #888; margin-bottom: 5px; }
            .status-card .value { font-size: 1.5em; font-weight: bold; }
            .status-card.danger .value { color: #ff6b6b; }
            .status-card.warning .value { color: #ffd93d; }
            .status-card.success .value { color: #6bcb77; }
            
            .video-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            .video-container {
                background: #000;
                border-radius: 10px;
                overflow: hidden;
                position: relative;
            }
            .video-container img { width: 100%; height: auto; display: block; }
            .video-label {
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0,0,0,0.7);
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 0.9em;
            }
            
            .controls {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-bottom: 20px;
            }
            button {
                padding: 15px 40px;
                font-size: 1.1em;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: transform 0.1s, box-shadow 0.2s;
            }
            button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0,0,0,0.3); }
            button:active { transform: translateY(0); }
            
            .btn-start { background: linear-gradient(135deg, #6bcb77 0%, #4caf50 100%); color: white; }
            .btn-stop { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%); color: white; }
            .btn-reset { background: linear-gradient(135deg, #4ecdc4 0%, #44a3aa 100%); color: white; }
            
            .state-indicator {
                text-align: center;
                padding: 20px;
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .state-indicator .state {
                font-size: 2em;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .state-CALIBRATING { color: #ffd93d; }
            .state-STANDBY { color: #87CEEB; }
            .state-ACTIVE_SCANNING { color: #ffa500; }
            .state-LOCALIZATION { color: #6bcb77; }
            .state-TARGET_CONFIRMED { color: #6bcb77; animation: pulse 1s infinite; }
            .state-EMERGENCY_STOP { color: #ff6b6b; }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .confirmation-banner {
                display: none;
                background: linear-gradient(135deg, #6bcb77 0%, #4caf50 100%);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 20px;
                animation: pulse 1s infinite;
            }
            .confirmation-banner.show { display: block; }
            .confirmation-banner h2 { font-size: 2em; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Gazzard</h1>
                <p>Gas Leak Detection and Localization using Autonomous Unmanned Vehicle</p>
            </div>
            
            <div class="confirmation-banner" id="confirmationBanner">
                <h2>🎯 LEAK SOURCE FOUND!</h2>
                <p id="confirmationDetails"></p>
            </div>
            
            <div class="state-indicator">
                <div class="state" id="currentState">INITIALIZING</div>
                <div id="stateMessage">Starting up...</div>
            </div>
            
            <div class="status-bar">
                <div class="status-card" id="co2Card">
                    <h3>CO₂ Level</h3>
                    <div class="value" id="co2Value">-- ppm</div>
                </div>
                <div class="status-card" id="crackCard">
                    <h3>Crack Detection</h3>
                    <div class="value" id="crackValue">--</div>
                </div>
                <div class="status-card" id="distanceCard">
                    <h3>Distance</h3>
                    <div class="value" id="distanceValue">-- cm</div>
                </div>
                <div class="status-card" id="speedCard">
                    <h3>Speed</h3>
                    <div class="value" id="speedValue">0.00 m/s</div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn-start" onclick="startSearch()">▶ Start</button>
                <button class="btn-stop" onclick="stopRobot()">⏹ Stop</button>
                <button class="btn-reset" onclick="resetSystem()">↻ Reset</button>
            </div>
            
            <div class="video-section">
                <div class="video-container">
                    <div class="video-label">📷 Camera Feed</div>
                    <img src="/camera_feed" alt="Camera">
                </div>
                <div class="video-container">
                    <div class="video-label">🌈 Depth Map</div>
                    <img src="/depth_feed" alt="Depth">
                </div>
            </div>
        </div>
        
        <script>
            function updateStatus() {
                fetch('/robot_status')
                    .then(r => r.json())
                    .then(data => {
                        // State
                        const stateEl = document.getElementById('currentState');
                        stateEl.textContent = data.state;
                        stateEl.className = 'state state-' + data.state;
                        document.getElementById('stateMessage').textContent = data.movement?.command || '';
                        
                        // CO2
                        const co2 = data.co2?.current || 0;
                        document.getElementById('co2Value').textContent = co2.toFixed(0) + ' ppm';
                        const co2Card = document.getElementById('co2Card');
                        co2Card.className = 'status-card ' + (data.co2?.high ? 'danger' : (data.co2?.elevated ? 'warning' : ''));
                        
                        // Crack
                        const crackDetected = data.crack?.detected;
                        document.getElementById('crackValue').textContent = crackDetected ? 'DETECTED' : 'None';
                        document.getElementById('crackCard').className = 'status-card ' + (crackDetected ? 'success' : '');
                        
                        // Distance
                        const distance = data.crack?.distance || 0;
                        document.getElementById('distanceValue').textContent = (distance < 1000 ? distance.toFixed(1) : '--') + ' cm';
                        
                        // Speed
                        document.getElementById('speedValue').textContent = (data.movement?.linear_speed || 0).toFixed(2) + ' m/s';
                        
                        // Confirmation banner
                        const banner = document.getElementById('confirmationBanner');
                        if (data.target_confirmed) {
                            banner.classList.add('show');
                            const conf = data.confirmation_data;
                            document.getElementById('confirmationDetails').textContent = 
                                `Crack at ${conf.crack_distance?.toFixed(1)}cm | CO₂: ${conf.co2_level?.toFixed(0)} ppm (${conf.co2_status})`;
                        } else {
                            banner.classList.remove('show');
                        }
                    })
                    .catch(e => console.error('Status error:', e));
            }
            
            function startSearch() { fetch('/start').then(r => r.json()).then(d => console.log(d)); }
            function stopRobot() { fetch('/stop').then(r => r.json()).then(d => console.log(d)); }
            function resetSystem() { fetch('/reset').then(r => r.json()).then(d => console.log(d)); }
            
            setInterval(updateStatus, 500);
            updateStatus();
        </script>
    </body>
    </html>
    """)


# ============================================
# MAIN
# ============================================

def main(args=None):
    global node
    rclpy.init(args=args)
    node = GasGuidedCrackDetector()
    
    # Start web server in background
    Thread(target=uvicorn.run, args=(app,), 
           kwargs={"host": "0.0.0.0", "port": 5000, "log_level": "warning"}, 
           daemon=True).start()
    
    print("\n" + "=" * 60)
    print("🌐 Web Interface: http://localhost:5000")
    print("=" * 60 + "\n")
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
