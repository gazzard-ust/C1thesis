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

# ============================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ============================================

# Fine-tuned YOLO-World XL model path
# Trained on crack detection dataset: mAP@50-95: 0.8692 | mAP@50: 0.9841
MODEL_PATH = "best.pt"

# Fine-tuned classes (must match training exactly)
CRACK_CLASSES = [
    "Dummy crack",
    "Paper crack",
    "PVC pipe crack",
]

# Detection settings
DETECTION_CONFIDENCE = 0.25  # Confidence threshold (0.25 for fine-tuned model)
IOU_THRESHOLD = 0.45

# Depth estimation model
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# Robot safety parameters (in cm)
STOPPING_DISTANCE_CM = 30.0
EMERGENCY_STOP_CM = 20.0
MIN_SAFE_DISTANCE_CM = 15.0

# ============================================
# END CONFIGURATION
# ============================================

app = FastAPI()
node = None  # Global node
target_object = "crack"  # Default target for crack detection

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.centering_status = "NONE"

        self.last_command_time = self.get_clock().now().nanoseconds
        self.safety_timer = self.create_timer(0.5, self.safety_check)
        self.search_enabled = False
        self.target_reached = False
        self.last_target_x = None
        self.last_target_y = None

        # DETECTION PARAMETERS - OPTIMIZED FOR FINE-TUNED MODEL
        self.target_lost_frames = 0
        self.detection_confidence_threshold = DETECTION_CONFIDENCE  # Use config value
        self.min_detection_frames = 1  # Accept detection immediately
        self.current_detection_count = 0  # Track consecutive detections
        self.detection_memory_frames = 60  # Remember detection for this many frames
        self.last_known_direction = 0
        self.current_turn_speed = 0.0

        # ADVANCED GEOMETRIC FILTERING - TEMPORARILY DISABLED FOR TESTING
        # To see if YOLO-World detects anything at all
        self.min_crack_aspect_ratio = 0.1  # Nearly disabled
        self.max_crack_area_ratio = 1.0  # Allow full frame
        self.min_crack_size = 1  # Allow tiny detections
        self.max_crack_size = 10000  # Allow huge detections
        self.enable_aspect_ratio_filter = False  # DISABLED for testing

        self.latest_frame = None
        self.latest_frame_array = None
        self.latest_camera = None
        self.latest_depth = None
        self.last_depth_map = None
        self.current_detections = []

        self.estimated_distance = 0.0
        self.initial_distance = None  # Store initial distance when object is first detected
        self.prev_distance = None     # Store previous frame's distance for approach rate
        self.movement_command = "Waiting for target"  # Default status message
        self.linear_speed = 0.0
        self.angular_speed = 0.0

        # CRITICAL SAFETY PARAMETERS - FROM CONFIG
        self.stopping_distance = STOPPING_DISTANCE_CM
        self.emergency_stop_distance = EMERGENCY_STOP_CM
        self.min_safe_distance = MIN_SAFE_DISTANCE_CM
        self.emergency_stop = False
        
        # Centering parameters
        self.center_tolerance = 30  # Pixels
        self.centering_gain = 0.004  # For turn control
        self.max_angular_speed = 0.3  # Max turning speed
        self.min_angular_speed = 0.0  # Min turning speed
        
        # Remove calibration offset - use true center
        self.calibration_offset = 0
        
        # Visual feedback
        self.centering_status = "NONE"
        self.alignment_state = "ALIGNING"  # Start in alignment mode
        self.aligned_frames_count = 0
        self.min_aligned_frames = 5  # Need this many consecutive aligned frames before moving
        
        # Error tracking for stability
        self.prev_errors = [0, 0, 0, 0, 0]
        self.error_idx = 0
        
        # SEARCH BEHAVIOR PARAMETERS
        self.search_timeout = 15.0  
        self.auto_search_delay = 15  # INCREASED - frames to wait before initiating auto-search
        self.search_pattern_index = 0
        self.search_direction = 1  # 1 for clockwise, -1 for counterclockwise
        self.search_pattern_change_time = time.time()
         # ENHANCED SEARCH PATTERNS WITH MORE VARIETY
        self.search_patterns = [
            # Phase 1: Gentle exploration around last known position
            {'speed': 0.03, 'duration': 3.0, 'name': 'Gentle Scan'},
            
            # Phase 2: Wider oscillating scan with direction changes
            {'speed': 0.07, 'duration': 4.0, 'name': 'Oscillating Scan'},
            
            # Phase 3: Medium sweep in one direction
            {'speed': 0.12, 'duration': 3.0, 'name': 'Medium Sweep'},
            
            # Phase 4: Short pause to stabilize
            {'speed': 0.0, 'duration': 1.0, 'name': 'Pause'},
            
            # Phase 5: Reverse direction medium sweep
            {'speed': 0.12, 'duration': 3.0, 'name': 'Reverse Sweep'},
            
            # Phase 6: Another short pause
            {'speed': 0.0, 'duration': 1.0, 'name': 'Pause'},
            
            # Phase 7: Faster wider sweep
            {'speed': 0.18, 'duration': 4.0, 'name': 'Fast Sweep'},
            
            # Phase 8: Maximum range sweep (full 360 search)
            {'speed': 0.25, 'duration': 6.0, 'name': 'Full Scan'},
            
            # Phase 9: Deceleration and stabilization
            {'speed': 0.12, 'duration': 2.0, 'name': 'Deceleration'},
            
            # Phase 10: Return to moderate scanning
            {'speed': 0.1, 'duration': 4.0, 'name': 'Moderate Scan'}
        ]
        
        # Initialize models and publishers
        self.initialize_models()
        self.initialize_publishers()
        
        # Camera parameters
        self.focal_length = 525
        self.frame_width = 640
        self.frame_center = (self.frame_width // 2) + self.calibration_offset
        
        # Search behavior parameters
        self.search_start_time = None  # Will be set when search is first enabled
        self.rotation_direction = 1
        self.last_rotation_change = time.time()
        
        # Object detection parameters
        self.first_detection_time = None
        self.object_detection_history = []
        
        # Historical detection tracking
        self.detection_buffer = []  # Store recent detection results
        self.detection_buffer_size = 10  # Track last 10 frames

        # FPS optimization: Skip frames for heavy AI processing
        self.frame_counter = 0
        self.process_every_n_frames = 2  # Process YOLO+Depth every 2 frames (doubles effective FPS)
        self.last_processed_detections = []
        self.last_processed_depth = None

        # Inference speed tracking
        self.yolo_inference_time = 0.0  # Time for YOLO detection
        self.depth_inference_time = 0.0  # Time for depth estimation
        self.total_inference_time = 0.0  # Total AI processing time

        # Detection metrics tracking
        self.avg_confidence = 0.0  # Average confidence of target detections
        self.detection_count = 0  # Total target detections
        self.confidence_history = []  # Rolling history for average

        # mAP50-95 tracking (COCO metric)
        # Stores ground truth and predictions for mAP calculation
        self.ground_truth_boxes = []  # Format: [{'bbox': [x1,y1,x2,y2], 'class': 'crack', 'frame': N}]
        self.predicted_boxes = []     # Format: [{'bbox': [x1,y1,x2,y2], 'class': 'crack', 'conf': X, 'frame': N}]
        self.map50_95_history = []    # Track mAP50-95 over time
        self.map50_history = []       # Track mAP50 over time
        self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5, 0.55, ..., 0.95

        self.get_logger().info(f"🔒 Safety: stopping={STOPPING_DISTANCE_CM}cm, emergency={EMERGENCY_STOP_CM}cm")
        self.get_logger().info(f"🎯 Detection confidence: {DETECTION_CONFIDENCE}")
        self.get_logger().info("⏳ Robot initialized - waiting for target")

    def initialize_360_search(self):
        """Initialize an immediate 360-degree search pattern"""
        # Set up simplified 360-degree search pattern
        self.search_patterns = [
            {'speed': 0.15, 'duration': 10.0, 'name': '360° Search'}  # Continuous 360° scan
        ]
        
        # Reset search pattern index
        self.search_pattern_index = 0
        self.search_pattern_change_time = time.time()
        
        # Set search direction based on image width (default clockwise)
        self.search_direction = 1
        
        # Enable search immediately
        self.search_enabled = True
        
        # Log the activation
        self.get_logger().info("🔄 Immediate 360° search pattern activated")


    def initialize_models(self):
        self.get_logger().info("🚀 Initializing Models...")
        
        # ============================================
        # FINE-TUNED YOLO-WORLD XL MODEL
        # ============================================
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"❌ Model not found: {MODEL_PATH}")
            self.get_logger().error("Please ensure best.pt is in the same directory")
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        self.get_logger().info(f"📦 Loading fine-tuned model: {MODEL_PATH}")
        self.model = YOLOWorld(MODEL_PATH)

        # Set fine-tuned classes (must match training exactly)
        self.model.set_classes(CRACK_CLASSES)
        self.get_logger().info(f"✅ YOLO-World XL loaded with classes: {CRACK_CLASSES}")
        
        # Depth estimation model
        self.get_logger().info(f"📦 Loading depth model: {DEPTH_MODEL}")

        self.depth_pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL)
        self.get_logger().info("✅ All Models Loaded Successfully!")

    def initialize_publishers(self):
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        self.movement_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            CompressedImage, '/image/compressed', self.image_callback, 1
        )

        # CO2 sensor subscription
        self.co2_subscription = self.create_subscription(
            Float32, '/co2_concentration', self.co2_callback, 10
        )
        self.latest_co2_ppm = None
        self.last_co2_time = None

        self.frame_event = Event()

    
    def image_callback(self, msg):
        start_time = time.time()

        try:
            # Step 1: Convert ROS2 CompressedImage to OpenCV frame
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Validate frame
            if frame is None or frame.size == 0:
                self.get_logger().error("❌ Invalid frame received!")
                return

            # Step 2: Store original frame and convert to RGB
            self.latest_frame_array = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # CRITICAL: Immediately prepare and stream the frame WITHOUT AI processing
            # This ensures smooth video feed regardless of AI inference time
            frame_display = frame.copy()

            # Draw previous detections if available (non-blocking)
            if self.current_detections:
                for det in self.current_detections:
                    x1, y1, x2, y2 = det['coordinates']
                    color = (0, 0, 255) if det.get('is_target', False) else (0, 255, 0)
                    cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{det['label']} {det['conf']:.2f}"
                    cv2.putText(frame_display, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Quick frame preparation for streaming
            ret, jpeg = cv2.imencode('.jpg', frame_display, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            self.latest_frame = jpeg.tobytes()
            self.latest_camera = jpeg.tobytes()

            # Create simple depth visualization (reuse previous if available)
            if self.last_processed_depth is not None:
                depth_vis = self.create_depth_visualization(self.last_processed_depth)
                ret, depth_jpeg = cv2.imencode('.jpg', depth_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                self.latest_depth = depth_jpeg.tobytes()

            # Signal new frame is ready (video stream can proceed immediately)
            self.frame_event.set()

            # FPS OPTIMIZATION: Only run heavy AI processing every N frames
            self.frame_counter += 1
            should_process = (self.frame_counter % self.process_every_n_frames == 0)

            if should_process:
                # Step 3: Object detection with YOLOWorld - MEASURE INFERENCE TIME
                yolo_start = time.time()
                results = self.model.predict(frame_rgb, conf=self.detection_confidence_threshold, iou=0.4, verbose=False)
                target_found, target_x, target_y = self.process_detections(frame, results)
                self.yolo_inference_time = time.time() - yolo_start

                # Step 4: Depth estimation - ALWAYS RUN alongside detection
                depth_start = time.time()
                depth_map = self.estimate_depth(frame_rgb)
                self.depth_inference_time = time.time() - depth_start
                self.last_processed_depth = depth_map

                # Calculate total inference time
                self.total_inference_time = self.yolo_inference_time + self.depth_inference_time

                # Update mAP metrics (calculates mAP50-95)
                self.update_map_metrics()

                # Debug logging for detection
                if self.frame_counter % 30 == 0:
                    self.get_logger().info(f"🔍 Detection: target_found={target_found}, detections={len(self.current_detections)}, target_object='{target_object}'")
            else:
                # Reuse cached results from last processed frame
                target_found, target_x, target_y = False, None, None
                if self.current_detections:
                    # Use previous detections
                    pass
                depth_map = self.last_processed_depth if self.last_processed_depth is not None else np.zeros((480, 640), dtype=np.float32)

            # Step 5: Movement control logic - Incorporate detection buffer for stability
            if target_object and target_object.strip():
                # Update detection buffer
                self.update_detection_buffer(target_found, target_x, target_y)

                # Use stable detection result based on buffer
                stable_detection = self.get_stable_detection()

                # Process movement based on stable detection
                self.process_movement(stable_detection['found'],
                                    stable_detection['x'],
                                    stable_detection['y'],
                                    depth_map)
            else:
                # Send stop command if no valid target
                move_cmd = Twist()
                self.movement_pub.publish(move_cmd)
                self.movement_command = "No Target"
                self.linear_speed = 0.0
                self.angular_speed = 0.0

        except Exception as e:
            self.get_logger().error(f"❌ Image Processing Error: {e}")
            # Fallback to black frame on critical error
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', black_frame)
            self.latest_frame = jpeg.tobytes()
            self.frame_event.set()

        # Log processing time only occasionally to avoid blocking
        processing_time = time.time() - start_time
        if self.frame_counter % 30 == 0:  # Log every 30 frames (~1 second)
            self.get_logger().info(f"⏱ Processing FPS: {1/processing_time:.2f}" if processing_time > 0 else "")

    def co2_callback(self, msg):
        """Callback for CO2 sensor data"""
        try:
            self.latest_co2_ppm = float(msg.data)
            self.last_co2_time = time.time()
            # Log CO2 only every 10th reading to avoid blocking
            if not hasattr(self, 'co2_log_counter'):
                self.co2_log_counter = 0
            self.co2_log_counter += 1
            if self.co2_log_counter % 10 == 0:
                self.get_logger().info(f"📊 CO2 Concentration: {self.latest_co2_ppm:.0f} ppm")
        except Exception as e:
            self.get_logger().error(f"❌ CO2 Callback Error: {e}")

    def update_detection_buffer(self, target_found, target_x, target_y):
        """Update the detection buffer with current detection results"""
        # Create detection entry
        detection = {
            'found': target_found,
            'x': target_x,
            'y': target_y,
            'time': time.time()
        }
        
        # Add to buffer
        self.detection_buffer.append(detection)
        
        # Trim buffer if needed
        if len(self.detection_buffer) > self.detection_buffer_size:
            self.detection_buffer.pop(0)


    def get_stable_detection(self):
        """Process detection buffer to provide stable detection results"""
        if not self.detection_buffer:
            return {'found': False, 'x': None, 'y': None}
        
        # Count recent detections
        recent_found_count = sum(1 for det in self.detection_buffer if det['found'])
        
        # If at least min_detection_frames detections in the buffer, consider target found
        if recent_found_count >= self.min_detection_frames:
            # Get coordinates from most recent successful detection
            for det in reversed(self.detection_buffer):
                if det['found']:
                    return {'found': True, 'x': det['x'], 'y': det['y']}
        
        # If we were recently tracking but lost it, provide last known position
        if recent_found_count > 0:
            for det in reversed(self.detection_buffer):
                if det['found']:
                    # Indicate not currently found but provide last position for recovery
                    return {'found': False, 'x': det['x'], 'y': det['y']}
        
        # No recent detections
        return {'found': False, 'x': None, 'y': None}


    def prepare_streaming_frames(self, frame, depth_map):
        try:
            # Validate and create depth visualization
            depth_vis = self.create_depth_visualization(depth_map)
            frame = cv2.resize(frame, (depth_vis.shape[1], depth_vis.shape[0]))
            
            # Draw detections on both frames
            self.draw_on_both_frames(frame, depth_vis)
            
            # Add status information to frames
            self.add_status_info(frame)

            # Save individual camera frame (with detections) before concatenating
            ret, camera_jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            self.latest_camera = camera_jpeg.tobytes()

            # Concatenate the frames for a side-by-side view
            combined = cv2.hconcat([frame, depth_vis])
            
            # Add header text
            self.add_frame_headers(combined)
            
            ret, jpeg = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            self.latest_frame = jpeg.tobytes()
            
            ret, depth_jpeg = cv2.imencode('.jpg', depth_vis)
            self.latest_depth = depth_jpeg.tobytes()
            
        except Exception as e:
            self.get_logger().error(f"Frame Prep Error: {e}")
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', black_frame)
            self.latest_frame = jpeg.tobytes()

    def process_detections(self, frame, results):
        """Process detection results and identify target objects"""
        global target_object
        
        self.current_detections = []
        target_found = False
        target_x, target_y = None, None
        
        # Record detection attempt for monitoring
        if not hasattr(self, 'detection_attempts'):
            self.detection_attempts = 0
        self.detection_attempts += 1
        
        # Track consecutive frames with no detections
        if not hasattr(self, 'consecutive_empty_frames'):
            self.consecutive_empty_frames = 0
        
        # Get total number of detections for logging
        total_detections = 0
        for r in results:
            if hasattr(r.boxes, 'xyxy') and len(r.boxes.xyxy) > 0:
                total_detections += len(r.boxes.xyxy)

        # DEBUG: Log YOLO raw detection count
        if self.frame_counter % 30 == 0:
            self.get_logger().info(f"📊 YOLO raw detections: {total_detections}")

        # EDGE-BASED CRACK DETECTION (Fallback)
        # Uses Canny edge detection to find dark cracks/damage on surfaces
        # Activates when YOLO detects nothing - good for thin cracks on pipes
        color_detections = []
        if total_detections == 0 and target_object and "crack" in target_object.lower():
            color_detections = self.detect_colored_objects(frame)
            if len(color_detections) > 0 and self.frame_counter % 30 == 0:
                self.get_logger().info(f"🔍 Edge detection found {len(color_detections)} cracks (YOLO: 0)")

        # If no objects at all were detected, increment empty frame counter
        if total_detections == 0 and len(color_detections) == 0:
            self.consecutive_empty_frames += 1

            # If we've had many empty frames, log a warning
            if self.consecutive_empty_frames % 10 == 0:
                self.get_logger().warning(f"⚠️ No objects detected for {self.consecutive_empty_frames} consecutive frames - conf_threshold={self.detection_confidence_threshold}")

            # Reset consecutive detection counter when no objects seen
            self.current_detection_count = 0
        else:
            # Reset counter when we do see objects
            self.consecutive_empty_frames = 0
        
        # Process all detections with improved matching logic
        confidence_threshold = self.detection_confidence_threshold
        detected_labels = []
        
        for r in results:
            if not hasattr(r.boxes, 'xyxy') or len(r.boxes.xyxy) == 0:
                continue
                
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                # Record this label
                if label not in detected_labels:
                    detected_labels.append(label)

                # Check if this object matches our target with improved matching
                is_target = False

                # ONLY match if we have a specific target object name
                if target_object and target_object.strip():
                    # Target matching for fine-tuned crack detection model
                    target_name = target_object.lower()
                    label_lower = label.lower()

                    # Direct match
                    if target_name == label_lower:
                        is_target = True
                    # Partial match (e.g., "crack" matches "Dummy crack")
                    elif target_name in label_lower:
                        is_target = True
                    # Match any fine-tuned crack class when target is "crack"
                    elif target_name == "crack" and any(crack_class.lower() in label_lower for crack_class in CRACK_CLASSES):
                        is_target = True
                    # Common substitutions for other objects
                    elif target_name == "bottle" and any(word in label_lower for word in ["container", "flask", "jar"]):
                        is_target = True
                    # Fallback: Legacy crack-related terms (for edge detection fallback)
                    elif target_name == "crack" and any(word in label_lower for word in [
                        "slit", "damage", "cut", "tear", "crack", "fracture"
                    ]):
                        is_target = True

                # GEOMETRIC FILTERING FOR CRACK DETECTION
                # Apply filters when target is "crack" to reduce false positives
                if is_target and "crack" in target_name and self.enable_aspect_ratio_filter:
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    frame_area = frame.shape[0] * frame.shape[1]

                    # Calculate aspect ratio (elongation)
                    if bbox_height > 0 and bbox_width > 0:
                        aspect_ratio = max(bbox_width / bbox_height, bbox_height / bbox_width)
                    else:
                        aspect_ratio = 1.0

                    # Log all crack detections before filtering
                    if self.frame_counter % 10 == 0:
                        self.get_logger().info(f"🔍 Crack detected: aspect={aspect_ratio:.2f}, area={bbox_area}/{frame_area}, size={bbox_width}x{bbox_height}, conf={conf:.3f}")

                    # Filter 1: Cracks should be elongated (not square/circular)
                    if aspect_ratio < self.min_crack_aspect_ratio:
                        if self.frame_counter % 10 == 0:
                            self.get_logger().info(f"❌ Filtered (aspect): {aspect_ratio:.2f} < {self.min_crack_aspect_ratio}")
                        is_target = False
                        continue

                    # Filter 2: Cracks shouldn't take up too much of the frame
                    area_ratio = bbox_area / frame_area
                    if area_ratio > self.max_crack_area_ratio:
                        if self.frame_counter % 10 == 0:
                            self.get_logger().info(f"❌ Filtered (area): {area_ratio:.2%} > {self.max_crack_area_ratio:.2%}")
                        is_target = False
                        continue

                    # Filter 3: Minimum size (too small might be noise)
                    if bbox_width < self.min_crack_size and bbox_height < self.min_crack_size:
                        if self.frame_counter % 10 == 0:
                            self.get_logger().info(f"❌ Filtered (too small): {bbox_width}x{bbox_height}")
                        is_target = False
                        continue

                    # Filter 4: Maximum size (too large might be wall/edge)
                    if bbox_width > self.max_crack_size or bbox_height > self.max_crack_size:
                        if self.frame_counter % 10 == 0:
                            self.get_logger().info(f"❌ Filtered (too large): {bbox_width}x{bbox_height}")
                        is_target = False
                        continue

                    # Log successful filter pass
                    self.get_logger().info(f"✅ VALID CRACK: aspect={aspect_ratio:.2f}, area={area_ratio:.2%}, size={bbox_width}x{bbox_height}, conf={conf:.3f}")
                
                # Store detection info
                self.current_detections.append({
                    'coordinates': (x1, y1, x2, y2),
                    'label': label,
                    'conf': conf,
                    'is_target': is_target
                })

                # Add prediction for mAP calculation (all detections, not just targets)
                self.add_prediction([x1, y1, x2, y2], "crack", conf, self.frame_counter)

                # Draw bounding box
                color = (0, 0, 255) if is_target else (0, 255, 0)
                self.draw_detection(frame, x1, y1, x2, y2, label, conf, color)

                # If this is our target, record its position
                if is_target:
                    if target_found:
                        # We already found a target - skip additional targets
                        continue

                    # Update confidence metrics
                    self.confidence_history.append(conf)
                    if len(self.confidence_history) > 100:  # Keep last 100 detections
                        self.confidence_history.pop(0)
                    self.avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
                    self.detection_count += 1

                    target_found = True
                    target_x, target_y = (x1 + x2)//2, (y1 + y2)//2
                    
                    # Increment consecutive detection counter
                    self.current_detection_count += 1
                    
                    # Store last known position for recovery
                    self.last_target_x = target_x
                    self.last_target_y = target_y
                    
                    # Calculate direction relative to center for recovery
                    frame_center = self.frame_center
                    if target_x > frame_center:
                        self.last_known_direction = -1  # Target is to the right, need to turn left
                    else:
                        self.last_known_direction = 1   # Target is to the left, need to turn right
                    
                    # IMPORTANT: If we've seen the target consistently for min_detection_frames, 
                    # disable search mode and reset counter
                    if self.current_detection_count >= self.min_detection_frames:
                        self.search_enabled = False
                        self.target_lost_frames = 0
                        if self.detection_attempts % 5 == 0:
                            self.get_logger().info(f"🎯 Target confirmed: {label} ({conf:.2f}) at ({target_x}, {target_y}) - Search disabled")
                    
                    # Log detection
                    if self.detection_attempts % 10 == 0:
                        self.get_logger().info(f"👁️ Target visible: {label} ({conf:.2f}) [{self.current_detection_count}/{self.min_detection_frames}]")

        # PROCESS COLOR-BASED DETECTIONS (fallback when YOLO fails)
        if len(color_detections) > 0:
            for det in color_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['conf']
                label = "crack (color)"  # Mark as color-detected

                # Check if this is our target
                is_target = False
                if target_object and "crack" in target_object.lower():
                    is_target = True

                # Store detection info
                self.current_detections.append({
                    'coordinates': (x1, y1, x2, y2),
                    'label': label,
                    'conf': conf,
                    'is_target': is_target
                })

                # Add prediction for mAP calculation
                self.add_prediction([x1, y1, x2, y2], "crack", conf, self.frame_counter)

                # Draw bounding box (orange for color-based detection)
                color = (0, 165, 255) if is_target else (0, 200, 200)  # Orange/yellow
                self.draw_detection(frame, x1, y1, x2, y2, label, conf, color)

                # If this is our target, record its position
                if is_target and not target_found:
                    # Update confidence metrics
                    self.confidence_history.append(conf)
                    if len(self.confidence_history) > 100:
                        self.confidence_history.pop(0)
                    self.avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
                    self.detection_count += 1

                    target_found = True
                    target_x, target_y = (x1 + x2)//2, (y1 + y2)//2

                    # Increment consecutive detection counter
                    self.current_detection_count += 1

                    # Store last known position
                    self.last_target_x = target_x
                    self.last_target_y = target_y

                    # Calculate direction
                    frame_center = self.frame_center
                    if target_x > frame_center:
                        self.last_known_direction = -1
                    else:
                        self.last_known_direction = 1

                    # Disable search if confirmed
                    if self.current_detection_count >= self.min_detection_frames:
                        self.search_enabled = False
                        self.target_lost_frames = 0
                        if self.detection_attempts % 5 == 0:
                            self.get_logger().info(f"🎯 Target confirmed (COLOR): {label} ({conf:.2f}) at ({target_x}, {target_y})")

        # If target not found, reset consecutive detection counter
        if not target_found:
            self.current_detection_count = 0
            
            # If we saw objects but not our target, log additional info occasionally
            if total_detections > 0 and self.detection_attempts % 10 == 0:
                # List what was detected
                self.get_logger().info(f"👁️ Detected objects but no target: {', '.join(detected_labels)}")
                
                # If we're looking for a specific target, remind what we're looking for
                if target_object and target_object.strip():
                    self.get_logger().info(f"🔍 Currently looking for: {target_object}")
        
        # Always return current frame detection result
        return target_found, target_x, target_y


    def estimate_depth(self, frame_rgb):
        try:
            image_pil = Image.fromarray(frame_rgb)
            depth_result = self.depth_pipe(image_pil)
            return np.array(depth_result['depth'])
        except Exception as e:
            self.get_logger().error(f"Depth Estimation Error: {e}")
            return np.zeros((480, 640), dtype=np.float32)  # Default depth map


    def draw_detection(self, img, x1, y1, x2, y2, label, conf, color):
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-th-5), (x1+tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Draw center for target
        if color == (0, 0, 255):
            # Calculate the center point
            center_x = (x1+x2)//2
            center_y = (y1+y2)//2
            center = (center_x, center_y)
            
            # Draw a larger, more visible center point
            cv2.circle(img, center, 8, color, -1)
            
            # Draw the frame center (blue vertical line)
            cv2.line(img, (self.frame_center, 0), (self.frame_center, img.shape[0]), (255, 0, 0), 1)
            
            # Draw the tolerance zone
            tolerance = self.center_tolerance
            
            # Draw left and right bounds of tolerance zone
            left_bound = self.frame_center - tolerance
            right_bound = self.frame_center + tolerance
            
            # Draw bounds as dotted/dashed lines
            for y in range(0, img.shape[0], 10):
                # Left bound segments
                cv2.line(img, (left_bound, y), (left_bound, min(y+5, img.shape[0])), (0, 165, 255), 1)
                # Right bound segments
                cv2.line(img, (right_bound, y), (right_bound, min(y+5, img.shape[0])), (0, 165, 255), 1)
            
            # Calculate error from center
            error = center_x - self.frame_center
            
            # Display important status information
            y_offset = 40
            
            # Check for emergency state
            is_emergency = self.emergency_stop or (hasattr(self, 'emergency_triggered') and self.emergency_triggered)
            
            # Display current and initial distance
            if hasattr(self, 'estimated_distance'):
                distance_text = f"Current: {self.estimated_distance:.1f}cm (Stop at {self.stopping_distance:.1f}cm)"
                cv2.putText(img, distance_text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
                
                # If we have an initial distance, show it
                if hasattr(self, 'initial_distance') and self.initial_distance is not None:
                    initial_text = f"Initial: {self.initial_distance:.1f}cm"
                    cv2.putText(img, initial_text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_offset += 30
            
            # Display emergency status if active
            if is_emergency:
                # Show emergency status in red
                cv2.putText(img, "⚠️ EMERGENCY STOP ⚠️", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
            # Otherwise show target status
            elif hasattr(self, 'target_reached') and self.target_reached:
                cv2.putText(img, "TARGET REACHED - STOPPED", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Display current speed
            if hasattr(self, 'linear_speed'):
                speed_text = f"Speed: {self.linear_speed:.2f} m/s"
                cv2.putText(img, speed_text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
                
            # Display alignment error
            turn_text = f"Error: {error}px (Target "
            if abs(error) <= self.center_tolerance:
                turn_text += "CENTERED)"
            elif error > 0:
                turn_text += "RIGHT of center)"
            else:
                turn_text += "LEFT of center)"
                
            cv2.putText(img, turn_text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def calculate_object_depth(self, x, y, depth_map):
        try:
            # First, verify the input coordinates are valid
            if not (0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]):
                self.get_logger().error(f"Invalid coordinates: x={x}, y={y}, shape={depth_map.shape}")
                return 100.0  # Return a safe default value
            
            # Get depth value at the target's center point
            center_depth = depth_map[y, x] * 100  # Convert to cm
            
            # Use a larger region around the point for more reliable depth
            region_size = 15  # Increased for better averaging
            y_min = max(0, y - region_size)
            y_max = min(depth_map.shape[0] - 1, y + region_size)
            x_min = max(0, x - region_size)
            x_max = min(depth_map.shape[1] - 1, x + region_size)
            
            # Calculate average depth in the region
            region_depths = depth_map[y_min:y_max, x_min:x_max] * 100
            
            # Filter out any zero or extreme values
            valid_depths = region_depths[(region_depths > 20) & (region_depths < 500)]
            
            if len(valid_depths) == 0:
                self.get_logger().warning("No valid depth values in region, using center point")
                if center_depth > 20 and center_depth < 500:
                    return float(center_depth)
                else:
                    self.get_logger().error(f"Invalid center depth: {center_depth}")
                    return 100.0  # Safe default
            
            # Use a more robust median instead of mean
            median_depth = float(np.median(valid_depths))
            
            # HISTORY-BASED FILTERING: Add to history and use moving average
            if not hasattr(self, 'depth_history'):
                self.depth_history = []
                
            # Add current reading to history, keeping last 5 readings
            self.depth_history.append(median_depth)
            if len(self.depth_history) > 5:
                self.depth_history.pop(0)
                
            # Check for rapid changes - if current depth reading is significantly different
            # from the average of previous readings, it might be a spike
            if len(self.depth_history) >= 3:
                avg_depth = sum(self.depth_history[:-1]) / len(self.depth_history[:-1])
                current_depth = self.depth_history[-1]
                
                # If the difference is more than 30%, consider it a spike
                if abs(current_depth - avg_depth) / avg_depth > 0.3:
                    self.get_logger().warning(f"⚠️ Detected depth spike: {current_depth:.1f}cm vs avg {avg_depth:.1f}cm")
                    # Use average instead of current reading
                    median_depth = avg_depth
                    
            # Log depth information
            self.get_logger().info(f"Depth stats: center={center_depth:.1f}cm, median={median_depth:.1f}cm")
            
            return median_depth
            
        except Exception as e:
            self.get_logger().error(f"Depth Calculation Error: {e}")
            return 100.0  # Fallback to a safe value
    
    def control_movement(self, move_cmd, target_x, object_depth, speed_factor=1.0):
        try:
            # Calculate the center of the image and the error
            frame_center = self.frame_center
            error = int(target_x) - frame_center
            
            # Log original error calculation for debugging
            self.get_logger().info(f"DEBUG: Original error calculation: target_x={target_x}, center={frame_center}, error={error}")
            
            # Initialize state tracking if not exists
            if not hasattr(self, 'alignment_state'):
                self.alignment_state = "ALIGNING"  # Start in alignment mode
                
            if not hasattr(self, 'aligned_frames_count'):
                self.aligned_frames_count = 0
                
            if not hasattr(self, 'min_aligned_frames'):
                self.min_aligned_frames = 5  # Need this many consecutive aligned frames before moving
            
            # Use a moderate tolerance for considering the target as centered
            self.center_tolerance = 25
            is_centered = abs(error) <= self.center_tolerance
            
            # Initialize movement speeds (always start with zero)
            angular_speed = 0.0
            linear_speed = 0.0
            
            # STRICT STATE MACHINE APPROACH with EXPLICIT DIRECTION REVERSAL
            if self.alignment_state == "ALIGNING":
                # In alignment mode: Only turn, never move forward
                if not is_centered:
                    # CRITICAL FIX: REVERSE THE TURNING DIRECTION
                    # If error is negative (target on left), we need to turn RIGHT (positive angular velocity)
                    # If error is positive (target on right), we need to turn LEFT (negative angular velocity)
                    # This is the opposite of what would normally be expected but matches your robot's behavior
                    turning_direction = 1 if error < 0 else -1  # REVERSED LOGIC
                    
                    # Use a very gentle fixed turning speed
                    turn_speed = 0.04  # Gentle turning speed
                    angular_speed = turn_speed * turning_direction
                    
                    # Reset aligned frames counter since we're not aligned
                    self.aligned_frames_count = 0
                    
                    # Set status message
                    direction_text = "RIGHT" if turning_direction > 0 else "LEFT"
                    self.movement_command = f"REVERSED ALIGNING: Turning {direction_text} for error {error}px"
                    
                    # Extra debug logging
                    self.get_logger().info(f"⚠️ REVERSED TURNING: error={error}, direction={turning_direction}, " + 
                                        f"speed={angular_speed}, turning {direction_text}")
                else:
                    # We're centered - increment aligned frames counter
                    self.aligned_frames_count += 1
                    
                    self.movement_command = f"CENTERING: Aligned for {self.aligned_frames_count}/{self.min_aligned_frames} frames"
                    self.get_logger().info(f"✓ TARGET CENTERED: {self.aligned_frames_count}/{self.min_aligned_frames} frames")
                    
                    # If we've been aligned for enough consecutive frames, switch to approach mode
                    if self.aligned_frames_count >= self.min_aligned_frames:
                        self.alignment_state = "APPROACHING"
                        self.get_logger().info("🚶 SWITCHING TO APPROACH MODE - target is stably centered")
            
            elif self.alignment_state == "APPROACHING":
                # In approach mode: Move forward if centered, otherwise go back to alignment
                if is_centered:
                    # Calculate safe forward speed based on distance
                    buffer = 5.0  # Buffer in cm for early stopping
                    
                    # Only move if beyond stopping distance
                    if object_depth > (self.stopping_distance + buffer):
                        # Very conservative speed based on distance
                        if object_depth > self.stopping_distance * 3:
                            linear_speed = 0.07 * speed_factor  # Far away
                        elif object_depth > self.stopping_distance * 2:
                            linear_speed = 0.05 * speed_factor  # Getting closer
                        elif object_depth > self.stopping_distance:
                            linear_speed = 0.03 * speed_factor  # Very close
                        
                        self.movement_command = f"APPROACHING: Moving forward ({object_depth:.1f}cm)"
                        self.get_logger().info(f"⬆️ APPROACHING: distance={object_depth:.1f}cm, speed={linear_speed:.2f}")
                    else:
                        # At stopping distance - completely stop
                        self.movement_command = "Target reached. Stopping."
                        self.target_reached = True
                        self.search_enabled = False
                else:
                    # Lost centering during approach - switch back to alignment mode
                    self.alignment_state = "ALIGNING"
                    self.aligned_frames_count = 0
                    self.movement_command = "Lost centering - realigning"
                    self.get_logger().info("⚠️ LOST CENTERING - switching back to alignment mode")
            
            # Apply the calculated velocities to the command
            move_cmd.linear.x = linear_speed
            move_cmd.angular.z = angular_speed
            
            # Store values for status reporting
            self.linear_speed = linear_speed
            self.angular_speed = angular_speed
            self.estimated_distance = object_depth
            self.centering_status = "CENTERED" if is_centered else "ALIGNING"
            
        except Exception as e:
            self.get_logger().error(f"Control Error: {e}")
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.0

    def process_movement(self, target_found, target_x, target_y, depth_map):
        try:
            move_cmd = Twist()
            global target_object
            
            # CRITICAL FIX: Always ensure the robot is stopped if no target object is set
            if not target_object or not target_object.strip():
                move_cmd.linear.x = 0.0
                move_cmd.angular.z = 0.0
                self.movement_command = "Waiting for target to be set"
                self.linear_speed = 0.0
                self.angular_speed = 0.0
                self.movement_pub.publish(move_cmd)
                return
            
            # If we've already reached the target, maintain complete stop
            if self.target_reached:
                move_cmd.linear.x = 0.0
                move_cmd.angular.z = 0.0
                self.movement_command = "Target reached. Stopped."
                self.linear_speed = 0.0
                self.angular_speed = 0.0
                self.search_enabled = False  # Explicitly disable search mode
                self.movement_pub.publish(move_cmd)
                return
            
            # Handle emergency stop cases
            if hasattr(self, 'emergency_triggered') and self.emergency_triggered:
                self.get_logger().warning("🔒 EMERGENCY STOP LATCHED - robot will remain stopped for safety")
                move_cmd.linear.x = 0.0
                move_cmd.angular.z = 0.0
                self.movement_command = "EMERGENCY STOP LATCHED"
                self.linear_speed = 0.0
                self.angular_speed = 0.0
                self.movement_pub.publish(move_cmd)
                return

            if self.emergency_stop:
                move_cmd.linear.x = 0.0
                move_cmd.angular.z = 0.0
                self.movement_command = "EMERGENCY STOP"
                self.linear_speed = 0.0
                self.angular_speed = 0.0
                self.movement_pub.publish(move_cmd)
                return

            # If target is found, process it normally
            if target_found:
                # Reset the target lost counter since we can see it
                self.target_lost_frames = 0
                
                # CRITICAL: Disable search mode when target is found
                self.search_enabled = False
                
                # Calculate object depth
                object_depth = self.calculate_object_depth(target_x, target_y, depth_map)
                self.estimated_distance = object_depth
                
                # Store the last time we saw the target
                self.last_target_seen_time = time.time()
                
                # Check if we're at stopping distance
                buffer = 5.0  
                if object_depth <= (self.stopping_distance + buffer):
                    self.get_logger().info(f"🎯 TARGET APPROACH: within {buffer}cm buffer at {object_depth:.1f}cm, stopping robot")
                    move_cmd.linear.x = 0.0
                    move_cmd.angular.z = 0.0
                    self.movement_command = "Target reached. Stopped."
                    self.target_reached = True
                    self.search_enabled = False
                    self.show_target_reached_notification = True
                    self.target_reached_time = time.time()
                    self.target_reached_distance = object_depth
                    self.movement_pub.publish(move_cmd)
                    return
                else:
                    # Continue with the strict stop-and-go approach
                    self.control_movement(move_cmd, target_x, object_depth)
            else:
                # Use search behavior if search is enabled (which should be the default upon setting a target)
                if self.search_enabled:
                    self.search_behavior(move_cmd)
                else:
                    # Not searching and no target visible - stay stopped
                    move_cmd.angular.z = 0.0
                    self.movement_command = "Waiting for target"
                    self.linear_speed = 0.0
                    self.angular_speed = 0.0
            
            # Publish movement command
            self.movement_pub.publish(move_cmd)
            self.last_command_time = self.get_clock().now().nanoseconds
            
        except Exception as e:
            self.get_logger().error(f"Movement Processing Error: {e}")
            # Safety: stop on error
            stop_cmd = Twist()

    def search_behavior(self, move_cmd):
        """Defines the robot's search behavior when target is not visible"""
        
        # CRITICAL SAFETY CHECK: Never search if target has been reached
        if self.target_reached:
            self.get_logger().warning("🚫 Search behavior called when target already reached. Ignoring and staying stopped.")
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.0
            self.movement_command = "Target reached. Stopped."
            self.linear_speed = 0.0
            self.angular_speed = 0.0
            self.search_enabled = False
            return

        # Only proceed with search if explicitly enabled
        if not self.search_enabled:
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.0
            self.movement_command = "Waiting for target"
            self.linear_speed = 0.0
            self.angular_speed = 0.0
            return
        
        # If we get here, search is enabled - proceed with search
        # Get current pattern (or default to simple rotation if no patterns defined)
        if not hasattr(self, 'search_patterns') or len(self.search_patterns) == 0:
            angular_speed = 0.15  # Default speed
            pattern_name = "Default Search"
        else:
            current_pattern = self.search_patterns[self.search_pattern_index]
            angular_speed = current_pattern['speed'] * self.search_direction
            pattern_name = current_pattern.get('name', "Pattern " + str(self.search_pattern_index+1))
        
        # Set movement commands
        move_cmd.angular.z = angular_speed
        move_cmd.linear.x = 0.0   # Don't move forward during search
        
        self.linear_speed = 0.0
        self.angular_speed = angular_speed
        
        # Update the movement command with search information
        self.movement_command = f"Searching for {target_object} [{pattern_name}]"
        
        # Add timeout check for search
        if hasattr(self, 'search_start_time') and self.search_start_time is not None:
            search_duration = time.time() - self.search_start_time
            if search_duration > 60.0:  # Extended 60 second max search time for 360 patterns
                self.get_logger().warning("⚠️ Search timeout reached (60s). Stopping for safety.")
                move_cmd.linear.x = 0.0
                move_cmd.angular.z = 0.0
                self.movement_command = "Search timeout - stopped"
                self.linear_speed = 0.0
                self.angular_speed = 0.0
                self.search_enabled = False
        else:
            # First time entering search mode, record the start time
            self.search_start_time = time.time()


    # Also add this function to ensure target_reached state persists during resets
    def set_target(object_name: str):
        """Set the target object to track and immediately start searching"""
        try:
            global target_object, node
            if not object_name.strip():
                return {"message": "Invalid empty target name"}
            if not node:
                return {"message": "ROS node not initialized"}
                
            target_object = object_name.strip().lower()
            
            # IMMEDIATELY ENABLE SEARCH MODE
            node.target_reached = False  # Clear any target reached state
            node.search_enabled = True   # Enable search immediately
            node.emergency_stop = False  # Clear any emergency stop
            
            # Reset any custom state flags
            if hasattr(node, 'emergency_triggered'):
                node.emergency_triggered = False
            if hasattr(node, 'stop_confirmation_count'):
                node.stop_confirmation_count = 0
            if hasattr(node, 'last_target_seen_time'):
                node.last_target_seen_time = None
            if hasattr(node, 'detection_attempts'):
                node.detection_attempts = 0
            if hasattr(node, 'consecutive_empty_frames'):
                node.consecutive_empty_frames = 0
                
            # Reset depth history
            if hasattr(node, 'depth_history'):
                node.depth_history = []
            
            # Reset the error buffer
            node.prev_errors = [0, 0, 0, 0, 0]
            node.error_idx = 0
                
            # Reset search timer and set to immediate search mode
            node.search_start_time = time.time()
            node.auto_search_delay = 0  # No delay before search
            node.target_lost_frames = 999  # Force immediate search
            
            # Initialize the 360-degree search pattern
            node.initialize_360_search()
            
            # Reset all motion parameters
            stop_cmd = Twist()
            node.movement_pub.publish(stop_cmd)  # Ensure robot is initially stopped
            node.movement_command = "Starting 360° search for " + target_object
            
            node.get_logger().info(f"🚀 New target set: {target_object} - Immediate 360° search enabled")
            return {"message": f"🚀 Tracking: {target_object.capitalize()} - 360° search started"}
            
        except Exception as e:
            return {"message": f"Error setting target: {str(e)}"}

    def add_status_info(self, frame):
        # Add target and movement status to the frame
        if target_object:
            cv2.putText(frame, f"Target: {target_object}", (10, frame.shape[0] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No target set", (10, frame.shape[0] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display current movement command
        cv2.putText(frame, f"Status: {self.movement_command}", (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def create_depth_visualization(self, depth_map):
        try:
            # Normalize depth map
            depth_normalized = cv2.normalize(
                depth_map, None, 0, 255, 
                cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            # Apply color map
            depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Get target shape from latest_frame_array
            if self.latest_frame_array is not None:
                target_height, target_width = self.latest_frame_array.shape[:2]
            else:
                # Fallback to default 640x480 if no frame
                target_width, target_height = 640, 480
                
            # Resize the depth map
            depth_vis = cv2.resize(
                depth_vis, 
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR
            )
            
            return depth_vis
            
        except Exception as e:
            self.get_logger().error(f"Depth Visualization Error: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame fallback


    def draw_on_both_frames(self, original_frame, depth_frame):
        for detection in self.current_detections:
            x1, y1, x2, y2 = detection['coordinates']
            label = detection['label']
            conf = detection['conf']
            color = (0, 0, 255) if detection['is_target'] else (0, 255, 0)
            
            # Draw on original frame
            self.draw_detection(original_frame, x1, y1, x2, y2, label, conf, color)
            
            # Draw on depth frame
            self.draw_detection(depth_frame, x1, y1, x2, y2, label, conf, color)

    def add_frame_headers(self, combined_frame):
        pass

    def detect_colored_objects(self, frame):
        """Detect dark cracks/damage using edge detection and thresholding

        CURRENTLY DISABLED for camera feeds - creates too many false positives.
        Only works well with test crack images (crack1.png - crack6.png).
        For real camera, use YOLO-World detections only.

        Args:
            frame: BGR image

        Returns:
            List of detections: [{'bbox': [x1,y1,x2,y2], 'conf': float, 'label': str}, ...]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection to find crack edges
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to connect nearby crack segments
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours of potential cracks
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area - cracks should be moderate size (not too small, not huge)
            if area < 200:  # Too small - likely noise
                continue
            if area > 50000:  # Too large - likely the pipe itself or background
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio - cracks are typically elongated
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

            # Filter out very square objects (not crack-like)
            if aspect_ratio < 1.5:  # Not elongated enough
                continue

            # Calculate confidence based on edge density in the region
            roi_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi_edges > 0) / (w * h) if (w * h) > 0 else 0

            # Confidence based on edge density and aspect ratio
            # Higher edge density = more likely to be a crack
            # Higher aspect ratio = more crack-like
            aspect_score = min(aspect_ratio / 10.0, 1.0)  # Normalize
            confidence = float(0.6 * edge_density + 0.4 * aspect_score)

            # Only keep detections with reasonable edge density
            if edge_density > 0.05:  # At least 5% of bbox contains edges
                detections.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'conf': min(confidence, 0.99),  # Cap at 0.99
                    'label': 'crack'
                })

        return detections

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes

        Args:
            box1, box2: [x1, y1, x2, y2] format

        Returns:
            float: IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def calculate_ap_at_iou(self, iou_threshold):
        """Calculate Average Precision at a specific IoU threshold

        Args:
            iou_threshold: float, IoU threshold (e.g., 0.5 for mAP50)

        Returns:
            float: Average Precision value
        """
        if not self.ground_truth_boxes or not self.predicted_boxes:
            return 0.0

        # Sort predictions by confidence (descending)
        predictions = sorted(self.predicted_boxes, key=lambda x: x['conf'], reverse=True)

        # Track which ground truths have been matched
        gt_matched = [False] * len(self.ground_truth_boxes)

        true_positives = []
        false_positives = []

        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching ground truth box
            for gt_idx, gt in enumerate(self.ground_truth_boxes):
                if gt_matched[gt_idx]:
                    continue

                if gt['class'] != pred['class']:
                    continue

                iou = self.calculate_iou(pred['bbox'], gt['bbox'])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if match is above threshold
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                true_positives.append(1)
                false_positives.append(0)
                gt_matched[best_gt_idx] = True
            else:
                true_positives.append(0)
                false_positives.append(1)

        # Calculate cumulative TP and FP
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)

        # Calculate precision and recall
        recalls = tp_cumsum / len(self.ground_truth_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Add boundary conditions
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))

        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Calculate area under curve (11-point interpolation)
        ap = 0.0
        for recall_threshold in np.linspace(0, 1, 11):
            precisions_above = precisions[recalls >= recall_threshold]
            if len(precisions_above) > 0:
                ap += precisions_above[0]
        ap /= 11.0

        return ap

    def calculate_map50_95(self):
        """Calculate mAP50-95 (COCO metric)

        Average of mAP at IoU thresholds from 0.5 to 0.95 (step 0.05)

        Returns:
            tuple: (mAP50-95, mAP50, per_threshold_aps)
        """
        if not self.ground_truth_boxes or not self.predicted_boxes:
            return 0.0, 0.0, []

        aps = []
        for iou_thresh in self.iou_thresholds:
            ap = self.calculate_ap_at_iou(iou_thresh)
            aps.append(ap)

        map50_95 = np.mean(aps)
        map50 = aps[0]  # First threshold is 0.5

        return map50_95, map50, aps

    def update_map_metrics(self):
        """Calculate and log current mAP metrics"""
        if len(self.predicted_boxes) > 0 and len(self.ground_truth_boxes) > 0:
            map50_95, map50, per_threshold_aps = self.calculate_map50_95()

            self.map50_95_history.append(map50_95)
            self.map50_history.append(map50)

            # Keep only last 100 measurements
            if len(self.map50_95_history) > 100:
                self.map50_95_history.pop(0)
                self.map50_history.pop(0)

            # Log every 30 frames
            if self.frame_counter % 30 == 0:
                self.get_logger().info(f"📊 mAP50-95: {map50_95:.3f} | mAP50: {map50:.3f} | Predictions: {len(self.predicted_boxes)} | GT: {len(self.ground_truth_boxes)}")

                # Log per-threshold breakdown occasionally
                if self.frame_counter % 300 == 0:
                    threshold_str = ", ".join([f"{thresh:.2f}:{ap:.3f}" for thresh, ap in zip(self.iou_thresholds, per_threshold_aps)])
                    self.get_logger().info(f"📈 AP per IoU threshold: {threshold_str}")

    def add_ground_truth(self, bbox, class_name, frame_number):
        """Add ground truth annotation for mAP calculation

        Args:
            bbox: [x1, y1, x2, y2]
            class_name: str
            frame_number: int
        """
        self.ground_truth_boxes.append({
            'bbox': bbox,
            'class': class_name,
            'frame': frame_number
        })

        # Keep only recent frames (last 1000 for memory efficiency)
        if len(self.ground_truth_boxes) > 1000:
            self.ground_truth_boxes.pop(0)

    def add_prediction(self, bbox, class_name, confidence, frame_number):
        """Add prediction for mAP calculation

        Args:
            bbox: [x1, y1, x2, y2]
            class_name: str
            confidence: float
            frame_number: int
        """
        self.predicted_boxes.append({
            'bbox': bbox,
            'class': class_name,
            'conf': confidence,
            'frame': frame_number
        })

        # Keep only recent frames (last 1000 for memory efficiency)
        if len(self.predicted_boxes) > 1000:
            self.predicted_boxes.pop(0)

    def safety_check(self):
        current_time = self.get_clock().now().nanoseconds
        time_diff = (current_time - self.last_command_time) / 1e9
        
        if time_diff > 1.0:  # 1 second timeout
            stop_cmd = Twist()
            self.movement_pub.publish(stop_cmd)
            self.get_logger().warning("🛑 No commands received for 1s, emergency stop!")

# FastAPI Endpoints
@app.get("/set_target")
async def set_target(object_name: str):
    """Set the target object to track"""
    try:
        global target_object, node
        if not object_name.strip():
            return {"message": "Invalid empty target name"}
        if not node:
            return {"message": "ROS node not initialized"}
            
        target_object = object_name.strip().lower()
        
        # Reset all state flags to initial search mode
        node.target_reached = False  # Clear the target reached state
        node.search_enabled = True   # Enable initial search
        node.emergency_stop = False  # Clear any emergency stop
        
        # Reset any custom state flags
        if hasattr(node, 'emergency_triggered'):
            node.emergency_triggered = False
        if hasattr(node, 'stop_confirmation_count'):
            node.stop_confirmation_count = 0
        if hasattr(node, 'last_target_seen_time'):
            node.last_target_seen_time = None
        if hasattr(node, 'detection_attempts'):
            node.detection_attempts = 0
        if hasattr(node, 'consecutive_empty_frames'):
            node.consecutive_empty_frames = 0
            
        # Reset depth history
        if hasattr(node, 'depth_history'):
            node.depth_history = []
        
        # Reset the error buffer
        node.prev_errors = [0, 0, 0, 0, 0]
        node.error_idx = 0
            
        # Reset search timer and initialize 360-degree search pattern
        node.search_start_time = time.time()
        node.initialize_360_search()  # IMPORTANT: Start rotating immediately

        # Reset motion command status
        node.movement_command = f"Searching for {target_object}"

        node.get_logger().info(f"🚀 New target set: {target_object} - 360° search enabled")
        return {"message": f"🚀 Tracking: {target_object.capitalize()} - Search active"}
        
    except Exception as e:
        return {"message": f"Error setting target: {str(e)}"}


@app.get("/reset_search")
async def reset_search():
    """Reset the search state without changing the target"""
    global node
    if node:
        # Don't change target or target_reached state, just reset search
        node.search_enabled = True
        node.search_start_time = time.time()
        node.movement_command = "Restarted search for existing target"
        
        # Reset search timeout
        if hasattr(node, 'consecutive_empty_frames'):
            node.consecutive_empty_frames = 0
        
        node.get_logger().info("🔄 Search state reset - searching for existing target")
        return {"message": "Search state reset successfully"}
    else:
        return {"message": "Robot node not initialized."}


@app.get("/target_status")
async def target_status():
    """Get information about the target status, including if it was just reached"""
    global node
    if node:
        # Get notification state
        notification_active = False
        notification_time = 0
        notification_distance = 0
        notification_age = 0
        
        if hasattr(node, 'show_target_reached_notification') and node.show_target_reached_notification:
            notification_active = True
            notification_time = getattr(node, 'target_reached_time', time.time())
            notification_distance = getattr(node, 'target_reached_distance', 0)
            notification_age = time.time() - notification_time
            
            # Auto-expire notification after 10 seconds
            if notification_age > 10:
                node.show_target_reached_notification = False
                notification_active = False
        
        return {
            "target_reached": node.target_reached,
            "emergency_stop": node.emergency_stop or (hasattr(node, 'emergency_triggered') and node.emergency_triggered),
            "distance": float(node.estimated_distance) if hasattr(node, 'estimated_distance') else 0.0,
            "notification": {
                "active": notification_active,
                "message": "TARGET REACHED!" if notification_active else "",
                "distance": float(notification_distance) if notification_active else 0.0,
                "time_ago": float(notification_age) if notification_active else 0.0
            },
            "movement_command": node.movement_command
        }
    else:
        return {"error": "Robot node not initialized"}


@app.get("/reset")
async def reset_target():
    global target_object, node
    
    # Clear the target object
    target_object = ""
    
    # Reset all movement and detection states
    node.search_enabled = False
    node.target_reached = False
    
    # Reset alignment state machine variables
    node.alignment_state = "ALIGNING"
    node.aligned_frames_count = 0
    
    # Reset position tracking
    node.last_target_x = None
    node.last_target_y = None
    node.last_known_direction = 0
    node.target_lost_frames = 0
    
    # Reset distance tracking
    if hasattr(node, 'estimated_distance'):
        node.estimated_distance = 0.0
    if hasattr(node, 'initial_distance'):
        node.initial_distance = None
    if hasattr(node, 'depth_history'):
        node.depth_history = []
        
    # Reset error tracking
    node.prev_errors = [0, 0, 0, 0, 0]
    node.error_idx = 0
    
    # Reset detection counters
    node.current_detections = []
    if hasattr(node, 'detection_attempts'):
        node.detection_attempts = 0
    if hasattr(node, 'consecutive_empty_frames'):
        node.consecutive_empty_frames = 0
    
    # Reset any emergency states
    node.emergency_stop = False
    if hasattr(node, 'emergency_triggered'):
        node.emergency_triggered = False
    
    # Reset speed variables
    node.linear_speed = 0.0
    node.angular_speed = 0.0
    if hasattr(node, 'current_turn_speed'):
        node.current_turn_speed = 0.0
    
    # Reset status messages
    node.movement_command = "Reset complete - Ready for new target"
    node.centering_status = "NONE"
    
    # Try to reset YOLOWorld model's classes if possible
    try:
        # Reset to fine-tuned classes from config
        node.model.set_classes(CRACK_CLASSES)
        node.get_logger().info(f"Reset detection model to fine-tuned classes: {CRACK_CLASSES}")
    except Exception as e:
        node.get_logger().warning(f"Could not reset model classes: {e}")
    
    # Stop the robot movement
    stop_cmd = Twist()
    node.movement_pub.publish(stop_cmd)
    # Send stop command again to ensure it's received
    time.sleep(0.1)
    node.movement_pub.publish(stop_cmd)
    
    node.get_logger().info("🔄 COMPLETE RESET - System ready for new target")
    return {"message": "Reset complete. You can now enter a new target."}

@app.get("/reset_target_reached")
async def reset_target_reached():
    """Special endpoint to reset the target_reached flag for debugging"""
    global node
    if node:
        node.target_reached = False
        node.get_logger().info("🔄 Reset target_reached flag to False for debugging")
        return {"message": "Reset target_reached flag to False"}
    else:
        return {"message": "Robot node not initialized."}

@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            node.frame_event.wait()
            node.frame_event.clear()
            if node.latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + node.latest_frame + b'\r\n')
            time.sleep(0.033)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/depth_feed")
def depth_feed():
    """Stream depth frames using Server-Sent Events (SSE)."""
    global node
    def generate():
        frame_count = 0
        while True:
            node.frame_event.wait()  # Wait for a new frame
            node.frame_event.clear()
            if node.latest_depth:
                # Log only occasionally to avoid blocking
                frame_count += 1
                if frame_count % 300 == 0:  # Log every 10 seconds
                    node.get_logger().info("📡 Streaming depth feed active")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + node.latest_depth + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/robot_status")
async def robot_status():
    try:
        global node
        if node:
            # Get base status from robot
            is_emergency = node.emergency_stop or (hasattr(node, 'emergency_triggered') and node.emergency_triggered)
            is_target_reached = node.target_reached and not is_emergency

            # Build appropriate status message based on state flags
            status_message = node.movement_command

            # Override with more specific messages for key states
            if is_emergency:
                status_message = "⚠️ EMERGENCY STOP ⚠️"
            elif is_target_reached:
                status_message = "Target reached. Stopped."

            # Return comprehensive status data
            return {
                "distance": float(node.estimated_distance) if hasattr(node, 'estimated_distance') else 0.0,
                "command": node.movement_command,
                "linear_speed": float(node.linear_speed),
                "angular_speed": float(node.angular_speed),
                "target_reached": node.target_reached,
                "emergency_stop": is_emergency,
                "search_enabled": node.search_enabled,
                "inference_times": {
                    "yolo_ms": round(node.yolo_inference_time * 1000, 1),
                    "depth_ms": round(node.depth_inference_time * 1000, 1),
                    "total_ms": round(node.total_inference_time * 1000, 1),
                    "yolo_fps": round(1.0 / node.yolo_inference_time, 1) if node.yolo_inference_time > 0 else 0,
                    "depth_fps": round(1.0 / node.depth_inference_time, 1) if node.depth_inference_time > 0 else 0
                },
                "avg_confidence": float(node.avg_confidence) if hasattr(node, 'avg_confidence') else 0.0,
                "detection_count": node.detection_count if hasattr(node, 'detection_count') else 0
            }
        return {"distance": None, "command": "Stopped", "linear_speed": 0.0, "angular_speed": 0.0}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug")
async def debug_info():
    """Endpoint for debugging distance and targeting issues"""
    global node
    if node:
        # Gather all relevant debug information
        debug_data = {
            "target_found": len(node.current_detections) > 0,
            "target_reached": node.target_reached,
            "current_distance": float(node.estimated_distance) if hasattr(node, 'estimated_distance') else None,
            "stopping_distance": float(node.stopping_distance),
            "emergency_stop": node.emergency_stop,
            "detection_count": len(node.current_detections),
            "search_enabled": node.search_enabled,
            "depth_history": node.depth_history if hasattr(node, 'depth_history') else [],
            "stop_confirmation_count": node.stop_confirmation_count if hasattr(node, 'stop_confirmation_count') else 0,
            "movement": {
                "linear_speed": float(node.linear_speed),
                "angular_speed": float(node.angular_speed),
                "command": node.movement_command
            }
        }
        
        # Reset depth history for testing
        if hasattr(node, 'depth_history'):
            node.depth_history = []
            
        # Reset stop confirmation count
        if hasattr(node, 'stop_confirmation_count'):
            node.stop_confirmation_count = 0
            
        return debug_data
    else:
        return {"message": "Robot node not initialized."}

@app.get("/map_metrics")
async def get_map_metrics():
    """Get current mAP50-95 and mAP50 metrics"""
    global node
    if node:
        if len(node.map50_95_history) > 0:
            current_map50_95 = node.map50_95_history[-1]
            current_map50 = node.map50_history[-1]
            avg_map50_95 = sum(node.map50_95_history) / len(node.map50_95_history)
            avg_map50 = sum(node.map50_history) / len(node.map50_history)

            return {
                "map50_95": {
                    "current": float(current_map50_95),
                    "average": float(avg_map50_95),
                    "history": [float(x) for x in node.map50_95_history[-20:]]  # Last 20 measurements
                },
                "map50": {
                    "current": float(current_map50),
                    "average": float(avg_map50),
                    "history": [float(x) for x in node.map50_history[-20:]]
                },
                "predictions_count": len(node.predicted_boxes),
                "ground_truth_count": len(node.ground_truth_boxes),
                "iou_thresholds": node.iou_thresholds
            }
        else:
            return {
                "map50_95": {"current": 0.0, "average": 0.0, "history": []},
                "map50": {"current": 0.0, "average": 0.0, "history": []},
                "predictions_count": len(node.predicted_boxes),
                "ground_truth_count": len(node.ground_truth_boxes),
                "message": "No mAP data yet - need both predictions and ground truth"
            }
    return {"error": "Node not initialized"}

@app.post("/add_ground_truth")
async def add_ground_truth_annotation(x1: float, y1: float, x2: float, y2: float, class_name: str = "crack"):
    """Add ground truth annotation for mAP calculation

    Args:
        x1, y1, x2, y2: Bounding box coordinates
        class_name: Object class (default: "crack")
    """
    global node
    if node:
        bbox = [x1, y1, x2, y2]
        node.add_ground_truth(bbox, class_name, node.frame_counter)
        return {
            "message": f"Ground truth added: {class_name} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]",
            "total_ground_truths": len(node.ground_truth_boxes)
        }
    return {"error": "Node not initialized"}

@app.get("/reset_map")
async def reset_map_data():
    """Reset all mAP tracking data"""
    global node
    if node:
        node.ground_truth_boxes = []
        node.predicted_boxes = []
        node.map50_95_history = []
        node.map50_history = []
        return {"message": "mAP data reset successfully"}
    return {"error": "Node not initialized"}

@app.get("/reset_emergency")
async def reset_emergency():
    """Reset the emergency stop state to allow movement again"""
    global node
    if node:
        # Reset emergency flags
        if hasattr(node, 'emergency_triggered'):
            node.emergency_triggered = False
        
        # Reset other state variables
        node.target_reached = False
        node.emergency_stop = False
        
        # Clear depth history
        if hasattr(node, 'depth_history'):
            node.depth_history = []
            
        # Reset movement parameters
        node.movement_command = "Emergency reset - robot released"
        
        node.get_logger().warning("🔓 Emergency stop state has been reset! Robot will move again.")
        return {"message": "Emergency stop state has been reset successfully"}
    else:
        return {"message": "Robot node not initialized."}


@app.get("/reset_target_state")
async def reset_target_state():
    """Reset all targeting and stopping state for fresh detection"""
    global node
    if node:
        # Reset all state variables related to targeting and stopping
        node.target_reached = False
        node.estimated_distance = 0.0
        if hasattr(node, 'depth_history'):
            node.depth_history = []
        if hasattr(node, 'stop_confirmation_count'):
            node.stop_confirmation_count = 0
        node.last_target_x = None
        node.last_target_y = None
        node.initial_distance = None
        node.prev_distance = None
        
        # Re-enable search
        node.search_enabled = True
        
        # Send stop command to ensure robot is stationary
        stop_cmd = Twist()
        node.movement_pub.publish(stop_cmd)
        node.movement_command = "State reset - ready for new detection"
        
        node.get_logger().info("🔄 Complete target state reset - ready for new detection")
        return {"message": "Target state reset successfully"}
    else:
        return {"message": "Robot node not initialized."}

@app.get("/stop")
async def stop_robot():
    """Emergency stop the robot and prevent further movement"""
    global node
    if node:
        # Set emergency_stop flag to True but do NOT set target_reached flag
        node.emergency_stop = True
        
        # Create a permanent emergency flag (doesn't reset on normal operations)
        node.emergency_triggered = True
        
        # Disable search but don't claim we reached the target
        node.search_enabled = False
        
        # Reset the motor commands
        node.linear_speed = 0.0
        node.angular_speed = 0.0
        node.movement_command = "⚠️ EMERGENCY STOP ⚠️"
        
        # Publish a zero velocity command to halt the robot immediately
        stop_cmd = Twist()
        node.movement_pub.publish(stop_cmd)
        
        # Publish again to make sure it gets there
        time.sleep(0.1)
        node.movement_pub.publish(stop_cmd)
        
        node.get_logger().warning("🛑 EMERGENCY STOP activated by user command")
        return {"message": "Emergency stop activated - robot stopped"}
    else:
        return {"message": "Robot node not initialized."}

@app.get("/background")
async def get_background():
    """Serve the background image"""
    return FileResponse("/home/c1/Documents/SO34/7.png")

@app.get("/gas_concentration")
async def get_gas_concentration():
    """Return real CO2 sensor data from SenseAir S8"""
    global node
    if node and hasattr(node, 'latest_co2_ppm') and node.latest_co2_ppm is not None:
        # Check if data is recent (within last 10 seconds)
        if node.last_co2_time and (time.time() - node.last_co2_time) < 10.0:
            return {
                "concentration": int(node.latest_co2_ppm),
                "unit": "ppm",
                "connected": True,
                "last_update": time.time() - node.last_co2_time
            }
        else:
            # Data exists but is stale
            node.get_logger().warning(f"⚠️ Stale CO2 data - last update {time.time() - node.last_co2_time:.1f}s ago")

    # No data or stale data - sensor not connected
    return {
        "concentration": 0,
        "unit": "ppm",
        "connected": False,
        "debug_message": "No CO2 data received - check ROS_DOMAIN_ID on turtlebot"
    }

@app.get("/connection_status")
async def get_connection_status():
    """Check if turtlebot is connected"""
    global node

    # Check camera feed connection
    camera_connected = node and hasattr(node, 'latest_frame_array') and node.latest_frame_array is not None

    # Check CO2 sensor connection
    co2_connected = False
    if node and hasattr(node, 'latest_co2_ppm') and node.latest_co2_ppm is not None:
        if node.last_co2_time and (time.time() - node.last_co2_time) < 10.0:
            co2_connected = True

    # Overall connection status
    if camera_connected or co2_connected:
        return {
            "connected": True,
            "message": "Turtlebot Connected",
            "camera": camera_connected,
            "co2_sensor": co2_connected
        }

    return {
        "connected": False,
        "message": "No Turtlebot Connected",
        "camera": False,
        "co2_sensor": False
    }

@app.get("/ros_diagnostics")
async def ros_diagnostics():
    """Diagnostic endpoint for ROS2 connection issues"""
    import os
    global node

    diagnostics = {
        "ros_domain_id": os.environ.get('ROS_DOMAIN_ID', 'not set (default 0)'),
        "node_initialized": node is not None,
    }

    if node:
        diagnostics["co2_data"] = {
            "has_data": hasattr(node, 'latest_co2_ppm') and node.latest_co2_ppm is not None,
            "latest_value": float(node.latest_co2_ppm) if hasattr(node, 'latest_co2_ppm') and node.latest_co2_ppm else None,
            "last_update_ago": (time.time() - node.last_co2_time) if hasattr(node, 'last_co2_time') and node.last_co2_time else None
        }
        diagnostics["camera_data"] = {
            "has_frame": hasattr(node, 'latest_frame_array') and node.latest_frame_array is not None
        }
        diagnostics["movement_state"] = {
            "target_object": target_object if target_object else "None",
            "search_enabled": node.search_enabled,
            "target_reached": node.target_reached,
            "emergency_stop": node.emergency_stop,
            "linear_speed": float(node.linear_speed),
            "angular_speed": float(node.angular_speed),
            "movement_command": node.movement_command
        }

    return diagnostics

@app.get("/camera_feed")
async def camera_feed():
    """Stream only the camera feed"""
    def generate():
        while True:
            node.frame_event.wait()
            node.frame_event.clear()
            if hasattr(node, 'latest_camera') and node.latest_camera:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + node.latest_camera + b'\r\n')
            time.sleep(0.033)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
    <head>
        <title>Gazzard - Gas Leak Detection and Localization</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Comic Sans MS', cursive, sans-serif;
                background-image: url('/background');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                min-height: 100vh;
                position: relative;
                overflow-x: hidden;
            }

            /* Main container */
            .main-container {
                position: relative;
                z-index: 10;
                padding: 20px;
                max-width: 1600px;
                margin: 0 auto;
            }

            /* Header */
            .header {
                text-align: center;
                margin-bottom: 20px;
            }

            .logo {
                display: inline-block;
                background: white;
                border: 4px solid black;
                padding: 10px 30px;
                border-radius: 10px;
                margin-bottom: 10px;
            }

            .logo-text {
                font-size: 48px;
                font-weight: bold;
                color: #FFD700;
                text-shadow: 3px 3px 0px black;
                font-style: italic;
            }

            .subtitle {
                font-size: 24px;
                color: #FF8C00;
                text-shadow: 2px 2px 0px white, 3px 3px 0px black;
                font-weight: bold;
            }

            /* Top controls */
            .top-controls {
                background: rgba(135, 206, 235, 0.5);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border: 2px solid rgba(0,0,0,0.2);
            }

            .gas-display {
                background: white;
                padding: 10px 20px;
                border-radius: 5px;
                border: 2px solid black;
                font-size: 18px;
                font-weight: bold;
            }

            .button-group {
                display: flex;
                gap: 15px;
            }

            button {
                padding: 12px 30px;
                font-size: 18px;
                font-weight: bold;
                border: 3px solid black;
                border-radius: 8px;
                cursor: pointer;
                box-shadow: 3px 3px 0px black;
                transition: all 0.1s;
                font-family: 'Comic Sans MS', cursive;
            }

            button:active {
                box-shadow: 1px 1px 0px black;
                transform: translate(2px, 2px);
            }

            .search-btn {
                background: linear-gradient(180deg, #87CEEB 0%, #4682B4 100%);
                color: white;
            }

            .cancel-btn {
                background: linear-gradient(180deg, #FFB6C1 0%, #FF69B4 100%);
                color: white;
            }

            /* Video section */
            .video-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 40px;
            }

            .video-container {
                position: relative;
                background: white;
                border: 4px solid black;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 5px 5px 0px black;
            }

            .video-container img {
                width: 100%;
                height: auto;
                display: block;
            }

            /* Connection status */
            .connection-status {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                z-index: 100;
            }

            .connected {
                color: #00FF00;
            }

            .disconnected {
                color: #FF0000;
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            <!-- Header -->
            <div class="header">
                <div class="logo">
                    <span class="logo-text">Gazzard: Gas Leak Detection and Localization using<br>an Autonomous Unmanned Vehicle</span>
                </div>
            </div>

            <!-- Top Controls -->
            <div class="top-controls">
                <div class="gas-display">
                    Gas Concentration: <span id="gasConcentration">Turtlebot not connected</span><span id="gasUnit"></span>
                </div>
                <div class="button-group">
                    <button class="search-btn" onclick="startSearch()">Search</button>
                    <button class="cancel-btn" onclick="cancelSearch()">Cancel</button>
                </div>
            </div>

            <!-- Video Section -->
            <div class="video-section">
                <div class="video-container">
                    <img src="/camera_feed" alt="Camera Feed">
                </div>
                <div class="video-container">
                    <img src="/depth_feed" alt="Depth Map">
                </div>
            </div>
        </div>
    
    <script>
        // Update gas concentration
        function updateGasConcentration() {
            fetch('/gas_concentration')
                .then(response => response.json())
                .then(data => {
                    const gasElement = document.getElementById('gasConcentration');
                    const unitElement = document.getElementById('gasUnit');
                    if (data.connected) {
                        gasElement.textContent = data.concentration;
                        unitElement.textContent = ' ' + data.unit;
                    } else {
                        gasElement.textContent = 'Turtlebot not connected';
                        unitElement.textContent = '';
                    }
                })
                .catch(error => console.error('Error fetching gas concentration:', error));
        }


        // Search button - start looking for gas leak (crack)
        function startSearch() {
            fetch('/set_target?object_name=crack')
                .then(response => response.json())
                .then(data => {
                    console.log('Search started:', data.message);
                })
                .catch(error => console.error('Error starting search:', error));
        }

        // Cancel button - reset and stop
        function cancelSearch() {
            fetch('/reset')
                .then(response => response.json())
                .then(data => {
                    console.log('Search cancelled:', data.message);
                })
                .catch(error => console.error('Error cancelling search:', error));
        }

        // Update every second
        setInterval(updateGasConcentration, 1000);

        // Initial updates
        updateGasConcentration();
    </script>
    </body>
    </html>
    """)

def main(args=None):
    global node
    rclpy.init(args=args)
    node = ImageSubscriber()
    Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 5000}, daemon=True).start()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
