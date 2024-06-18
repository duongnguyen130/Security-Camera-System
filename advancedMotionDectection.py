import cv2
import torch
import numpy as np
import pygame
import threading

# IP Camera URLs
ip_camera_urls = [
    "rtsp://admin:888888@10.0.0.224:8554/profile0",
    "rtsp://admin:888888@10.0.0.161:8554/profile0",
    "rtsp://admin:888888@10.0.0.152:8554/profile0",
    "rtsp://admin:888888@10.0.0.122:8554/profile0"
]

# Initialize the video capture objects for each IP camera
caps = [cv2.VideoCapture(url) for url in ip_camera_urls]

# Check if all video streams are opened successfully
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Could not open video stream from IP camera {i + 1}.")
        exit()

frames = [None] * len(caps)
rets = [False] * len(caps)

# Define functions to capture frames from each camera
def capture_camera(index):
    global frames, rets
    while True:
        rets[index], frames[index] = caps[index].read()

# Start threads for capturing frames from all cameras
threads = [threading.Thread(target=capture_camera, args=(i,)) for i in range(len(caps))]
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

# Polygon points
polygons = []
current_polygon = []

# Function to draw polygon (ROI)
def draw_polygon(event, x, y, flags, param):
    global current_polygon, polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
        if len(current_polygon) == 4:  # Require 4 clicks to create a polygon
            polygons.append(current_polygon)
            current_polygon = []
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_polygon = []

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(np.array(polygon), (point[0], point[1]), False)
    return result >= 0

# Set up OpenCV window and mouse callback
cv2.namedWindow('Combined IP Camera Feeds', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Combined IP Camera Feeds', draw_polygon)

while True:
    if all(rets):
        # Resize frames to the same width, keeping a reasonable size
        new_width = 640  # Adjust the target width as needed
        frame1_resized = cv2.resize(frames[0], (new_width, int(frames[0].shape[0] * new_width / frames[0].shape[1])))
        frame2_resized = cv2.resize(frames[1], (new_width, int(frames[1].shape[0] * new_width / frames[1].shape[1])))
        frame3_resized = cv2.resize(frames[2], (new_width, int(frames[2].shape[0] * new_width / frames[2].shape[1])))
        frame4_resized = cv2.resize(frames[3], (new_width, int(frames[3].shape[0] * new_width / frames[3].shape[1])))

        # Concatenate frames 1 and 3 vertically
        frame1_3 = np.vstack((frame1_resized, frame3_resized))

        # Concatenate frame 2 and 4 vertically
        frame2_4 = np.vstack((frame2_resized, frame4_resized))

        # Concatenate the two combined frames horizontally
        combined_frame = np.hstack((frame1_3, frame2_4))

        # Get the current window size
        window_width = cv2.getWindowImageRect('Combined IP Camera Feeds')[2]
        window_height = cv2.getWindowImageRect('Combined IP Camera Feeds')[3]

        # Resize the combined frame to fit the window size
        combined_frame_resized = cv2.resize(combined_frame, (window_width, window_height))

        # Detect objects in the combined frame
        frame_detected = combined_frame_resized.copy()
        results = model(combined_frame_resized)

        # Create an overlay for the polygons
        overlay = combined_frame_resized.copy()

        # Draw and fill all polygons with transparency
        for polygon in polygons:
            cv2.fillPoly(overlay, [np.array(polygon)], (0, 255, 0))
            cv2.polylines(overlay, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Blend the overlay with the original frame
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, combined_frame_resized, 1 - alpha, 0, combined_frame_resized)

        # Process detection results
        for index, row in results.pandas().xyxy[0].iterrows():
            if row['name'] in target_classes:
                name = str(row['name'])
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw bounding box and center point
                cv2.rectangle(combined_frame_resized, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(combined_frame_resized, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.circle(combined_frame_resized, (center_x, center_y), 5, (0, 0, 255), -1)

                # Check if object is inside any polygon
                for polygon in polygons:
                    if inside_polygon((center_x, center_y), polygon) and name == 'person':
                        mask = np.zeros_like(frame_detected)
                        points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).reshape((-1, 1, 2))
                        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
                        frame_detected = cv2.bitwise_and(frame_detected, mask)

                        # Save detected image
                        if count < number_of_photos:
                            cv2.imwrite(f"Detected Photos/detected{count}.jpg", frame_detected)

                        # Play alarm sound
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.play()

                        # Indicate detection on the frame
                        cv2.putText(combined_frame_resized, "Target", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(combined_frame_resized, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(combined_frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        count += 1

        # Display the processed frame
        cv2.imshow('Combined IP Camera Feeds', combined_frame_resized)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video captures and destroy all windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
