import cv2
import numpy as np

def initialize_cameras(ip_camera_urls):
    caps = [cv2.VideoCapture(url) for url in ip_camera_urls]
    frames = [None] * len(caps)
    rets = [False] * len(caps)

    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Could not open video stream from IP camera {i + 1}.")
            exit()

    return caps, frames, rets

def capture_camera(index, caps, frames, rets):
    while True:
        rets[index], frames[index] = caps[index].read()

def combine_and_resize_frames(frames):
    new_width = 640  # Adjust the target width as needed
    frame1_resized = cv2.resize(frames[0], (new_width, int(frames[0].shape[0] * new_width / frames[0].shape[1])))
    frame2_resized = cv2.resize(frames[1], (new_width, int(frames[1].shape[0] * new_width / frames[1].shape[1])))
    frame3_resized = cv2.resize(frames[2], (new_width, int(frames[2].shape[0] * new_width / frames[2].shape[1])))
    frame4_resized = cv2.resize(frames[3], (new_width, int(frames[3].shape[0] * new_width / frames[3].shape[1])))

    frame1_3 = np.vstack((frame1_resized, frame3_resized))
    frame2_4 = np.vstack((frame2_resized, frame4_resized))
    combined_frame = np.hstack((frame1_3, frame2_4))

    window_width = cv2.getWindowImageRect('Combined IP Camera Feeds')[2]
    window_height = cv2.getWindowImageRect('Combined IP Camera Feeds')[3]

    combined_frame_resized = cv2.resize(combined_frame, (window_width, window_height))
    
    return combined_frame_resized

def release_cameras(caps):
    for cap in caps:
        cap.release()
