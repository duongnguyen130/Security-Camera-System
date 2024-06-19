import cv2
import pygame
import threading
import os
import camera_utils
import detection_utils
from emailAlert import email_alert  # Importing the email_alert function
import torch

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

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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

# Set up OpenCV window and mouse callback
cv2.namedWindow('Combined IP Camera Feeds', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Combined IP Camera Feeds', detection_utils.draw_polygon, {'polygons': polygons, 'polygon_file': polygon_file})

while True:
    if all(rets):
        combined_frame_resized = camera_utils.combine_and_resize_frames(frames)
        
        # Detect objects in the combined frame
        frame_detected = combined_frame_resized.copy()
        results = model(combined_frame_resized)

        # Create an overlay for the polygons
        overlay = combined_frame_resized.copy()
        combined_frame_resized = detection_utils.draw_polygons_on_frame(overlay, combined_frame_resized, polygons)

        # Process detection results
        count = detection_utils.process_detections(results, combined_frame_resized, frame_detected, polygons, target_classes, pygame, email_alert, photo_dir, count, number_of_photos)

        # Display the processed frame
        cv2.imshow('Combined IP Camera Feeds', combined_frame_resized)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and destroy all windows
camera_utils.release_cameras(caps)
cv2.destroyAllWindows()
