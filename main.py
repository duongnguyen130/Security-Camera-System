import cv2
import pygame
import threading
import os
import torch
import logging
from ultralytics import YOLO  # Importing YOLOv8

import camera_utils
import detection_utils
import fps_utils  # Import FPS utilities
import ui_utils
from emailAlert import email_alert  # Importing the email_alert function

# Set up logging
logging.basicConfig(level=logging.INFO)

# IP Camera URLs
ip_camera_urls = [
    "rtsp://admin:888888@10.0.0.224:8554/profile0",
    "rtsp://admin:888888@10.0.0.161:8554/profile0",
    "rtsp://admin:888888@10.0.0.152:8554/profile0",
    "rtsp://admin:888888@10.0.0.122:8554/profile0"
]

# Initialize the video capture objects for each IP camera
caps, frames, rets = camera_utils.initialize_cameras(ip_camera_urls)

# Start threads for capturing frames from all cameras
threads = [threading.Thread(target=camera_utils.capture_camera, args=(i, caps, frames, rets)) for i in range(len(caps))]
for thread in threads:
    thread.start()

# Initialize pygame for alarm sound
pygame.init()
pygame.mixer.music.load("Alarm/alarm.wav")

# Load the YOLOv8 model
logging.info("Loading YOLOv8 model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")
model = YOLO('yolov8n.pt').to(device)  # Specify the YOLOv8 Nano model for better FPS and move it to GPU

# Target classes for detection
target_classes = ['car', 'bus', 'truck', 'person']

count = 0
number_of_photos = 3

# Directory to save detected photos
photo_dir = "AlertPhoto"
if not os.path.exists(photo_dir):
    os.makedirs(photo_dir)

# Load polygons from file
polygon_file = "polygons.json"
polygons = detection_utils.load_polygons(polygon_file)
current_polygon = []

# Set up OpenCV window and mouse callback
cv2.namedWindow('Combined IP Camera Feeds', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Combined IP Camera Feeds', detection_utils.draw_polygon, {'polygons': polygons, 'polygon_file': polygon_file, 'current_polygon': current_polygon})

# Initialize FPS counter
fps_counter = fps_utils.FPS()

while True:
    try:
        if all(rets):
            combined_frame_resized = camera_utils.combine_and_resize_frames(frames)
            
            # Detect objects in the combined frame
            frame_detected = combined_frame_resized.copy()
            logging.info("Running detection...")
            results = model.predict(source=combined_frame_resized, device=device)

            # Create an overlay for the polygons
            overlay = combined_frame_resized.copy()
            combined_frame_resized = detection_utils.draw_polygons_on_frame(overlay, combined_frame_resized, polygons)

            # Process detection results
            count = detection_utils.process_detections(results, combined_frame_resized, frame_detected, polygons, target_classes, pygame, email_alert, photo_dir, count, number_of_photos)

            # Draw buttons (if you still need them)
            combined_frame_resized = ui_utils.draw_buttons(combined_frame_resized)

            # Update FPS counter
            fps_counter.update()
            fps = fps_counter.get_fps()

            # Display FPS
            combined_frame_resized = fps_utils.display_fps(combined_frame_resized, fps)

            # Display the processed frame
            cv2.imshow('Combined IP Camera Feeds', combined_frame_resized)

            # Check for button clicks
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Set mouse callback for button click detection
            cv2.setMouseCallback('Combined IP Camera Feeds', detection_utils.draw_polygon, {'polygons': polygons, 'polygon_file': polygon_file, 'current_polygon': current_polygon})
    except Exception as e:
        logging.error(f"Error occurred: {e}")

# Release video captures and destroy all windows
camera_utils.release_cameras(caps)
cv2.destroyAllWindows()
