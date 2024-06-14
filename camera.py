import cv2
import numpy as np
import threading
import tkinter as tk

# Replace with the appropriate URLs for your IP cameras
ip_camera_url_1 = "rtsp://admin:888888@10.0.0.152:8554/profile0"
ip_camera_url_2 = "rtsp://admin:888888@10.0.0.122:8554/profile0"
ip_camera_url_3 = "rtsp://admin:888888@10.0.0.139:8554/profile0"

# Initialize the video capture objects
cap1 = cv2.VideoCapture(ip_camera_url_1)
cap2 = cv2.VideoCapture(ip_camera_url_2)
cap3 = cv2.VideoCapture(ip_camera_url_3)

if not cap1.isOpened():
    print("Error: Could not open video stream from IP camera 1.")
    exit()

if not cap2.isOpened():
    print("Error: Could not open video stream from IP camera 2.")
    exit()

if not cap3.isOpened():
    print("Error: Could not open video stream from IP camera 3.")
    exit()

frame1 = None
frame2 = None
frame3 = None
ret1 = False
ret2 = False
ret3 = False

# Define a function to capture frames from the first camera
def capture_camera1():
    global frame1, ret1
    while True:
        ret1, frame1 = cap1.read()

# Define a function to capture frames from the second camera
def capture_camera2():
    global frame2, ret2
    while True:
        ret2, frame2 = cap2.read()

# Define a function to capture frames from the third camera
def capture_camera3():
    global frame3, ret3
    while True:
        ret3, frame3 = cap3.read()

# Start threads for capturing frames from all cameras
thread1 = threading.Thread(target=capture_camera1)
thread2 = threading.Thread(target=capture_camera2)
thread3 = threading.Thread(target=capture_camera3)

thread1.start()
thread2.start()
thread3.start()

# Create a resizable window
cv2.namedWindow('Combined IP Camera Feeds', cv2.WINDOW_NORMAL)

while True:
    if ret1 and ret2 and ret3:
        # Resize frames to the same width, keeping a reasonable size
        new_width = 640  # Adjust the target width as needed
        frame1_resized = cv2.resize(frame1, (new_width, int(frame1.shape[0] * new_width / frame1.shape[1])))
        frame2_resized = cv2.resize(frame2, (new_width, int(frame2.shape[0] * new_width / frame2.shape[1])))
        frame3_resized = cv2.resize(frame3, (new_width, int(frame3.shape[0] * new_width / frame3.shape[1])))

        # Concatenate frames 1 and 3 vertically
        frame1_3_combined = np.vstack((frame1_resized, frame3_resized))

        # Create a blank frame with the same size as frame 2
        blank_frame = np.zeros_like(frame2_resized)

        # Concatenate frame 2 and the blank frame vertically
        frame2_blank_combined = np.vstack((frame2_resized, blank_frame))

        # Concatenate the two combined frames horizontally
        combined_frame = np.hstack((frame1_3_combined, frame2_blank_combined))

        # Get the current window size
        window_width = cv2.getWindowImageRect('Combined IP Camera Feeds')[2]
        window_height = cv2.getWindowImageRect('Combined IP Camera Feeds')[3]

        # Resize the combined frame to fit the window size
        combined_frame_resized = cv2.resize(combined_frame, (window_width, window_height))

        # Display the combined frame
        cv2.imshow('Combined IP Camera Feeds', combined_frame_resized)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the captures
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()