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
    # Resize the frames to smaller dimensions while maintaining quality
    new_width = 960  # Half of Full HD width
    new_height = 540  # Half of Full HD height
    
    frame1_resized = cv2.resize(frames[0], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    frame2_resized = cv2.resize(frames[1], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    frame3_resized = cv2.resize(frames[2], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    frame4_resized = cv2.resize(frames[3], (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create a blank frame with Full HD resolution (1920x1080)
    combined_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Place the resized frames into the blank frame
    combined_frame[0:new_height, 0:new_width] = frame1_resized
    combined_frame[0:new_height, new_width:new_width*2] = frame2_resized
    combined_frame[new_height:new_height*2, 0:new_width] = frame3_resized
    combined_frame[new_height:new_height*2, new_width:new_width*2] = frame4_resized
    
    return combined_frame

def release_cameras(caps):
    for cap in caps:
        cap.release()
