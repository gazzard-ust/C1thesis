#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from cv_bridge import CvBridge

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(CompressedImage, '/image/compressed', 1)

        # Increased FPS: 0.033s = ~30 FPS (changed from 0.1s = 10 FPS)
        self.timer = self.create_timer(0.033, self.publish_image)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            return

        self.get_logger().info("Image Publisher Node Started")

    def publish_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Lost camera connection. Retrying...")
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
            return

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.array(buffer).tobytes()

        self.publisher.publish(msg)
        self.get_logger().info("Published rotated compressed image")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Image Publisher Shutdown")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
