#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import glob
import os
import requests

class CrackImagePublisher(Node):
    def __init__(self):
        super().__init__('crack_image_publisher')
        self.publisher = self.create_publisher(CompressedImage, '/image/compressed', 1)

        # Publish at 2 FPS (0.5 seconds) to allow time to see each crack image
        self.timer = self.create_timer(0.5, self.publish_image)

        # Load all crack images from the SO34 directory
        crack_dir = '/home/c1/Documents/SO34'
        self.image_paths = sorted(glob.glob(os.path.join(crack_dir, 'crack*.png')))

        if not self.image_paths:
            self.get_logger().error(f"No crack images found in {crack_dir}")
            return

        self.get_logger().info(f"Found {len(self.image_paths)} crack images")

        # Pre-compute ground truth bounding boxes for all crack images
        self.ground_truth_boxes = {}
        for path in self.image_paths:
            bbox = self.extract_crack_bbox(path)
            if bbox is not None:
                self.ground_truth_boxes[path] = bbox
                self.get_logger().info(f"  ✓ {os.path.basename(path)} - bbox: {bbox}")
            else:
                self.get_logger().warning(f"  ✗ {os.path.basename(path)} - no crack detected")

        self.current_index = 0
        self.frame_counter = 0
        self.get_logger().info("Crack Image Publisher Started - cycling through crack images with ground truth")

    def extract_crack_bbox(self, image_path):
        """Extract bounding box from crack image by detecting non-black regions

        Args:
            image_path: Path to crack image

        Returns:
            [x1, y1, x2, y2] bounding box at 640x480 resolution, or None if no crack found
        """
        # Load image (with alpha channel if available)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        # Resize to standard resolution (640x480)
        img = cv2.resize(img, (640, 480))

        # Check if image has alpha channel
        if img.shape[2] == 4:
            # Use alpha channel to detect crack (non-transparent pixels)
            alpha = img[:, :, 3]
            mask = (alpha > 10).astype(np.uint8) * 255
        else:
            # Convert to grayscale and detect non-black pixels
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Threshold for non-black pixels (crack objects are colored/bright)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest contour (assuming it's the crack)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add some padding (10 pixels)
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(640, x + w + padding)
        y2 = min(480, y + h + padding)

        return [float(x1), float(y1), float(x2), float(y2)]

    def publish_image(self):
        if not self.image_paths:
            return

        # Load current image
        image_path = self.image_paths[self.current_index]
        frame = cv2.imread(image_path)

        if frame is None:
            self.get_logger().error(f"Failed to load {image_path}")
            return

        # Resize to standard resolution if needed
        frame = cv2.resize(frame, (640, 480))

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Create ROS2 message
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.array(buffer).tobytes()

        # Publish image
        self.publisher.publish(msg)

        # Send ground truth to gazzard_gui_v2 via HTTP
        if image_path in self.ground_truth_boxes:
            bbox = self.ground_truth_boxes[image_path]
            try:
                response = requests.post(
                    'http://localhost:5000/add_ground_truth',
                    params={
                        'x1': bbox[0],
                        'y1': bbox[1],
                        'x2': bbox[2],
                        'y2': bbox[3],
                        'class_name': 'crack'
                    },
                    timeout=0.5
                )
                if response.status_code == 200:
                    self.get_logger().info(f"📸 Published: {os.path.basename(image_path)} | GT bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}] ✓")
                else:
                    self.get_logger().warning(f"📸 Published: {os.path.basename(image_path)} | GT upload failed: {response.status_code}")
            except requests.exceptions.RequestException as e:
                self.get_logger().warning(f"📸 Published: {os.path.basename(image_path)} | GT upload failed: {e}")
        else:
            self.get_logger().info(f"📸 Published: {os.path.basename(image_path)} | No GT available")

        # Increment frame counter
        self.frame_counter += 1

        # Move to next image (loop back to start)
        self.current_index = (self.current_index + 1) % len(self.image_paths)

def main(args=None):
    rclpy.init(args=args)
    node = CrackImagePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
