import time
import cv2

class FPS:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def update(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed_time
            self.start_time = current_time
            self.frame_count = 0

    def get_fps(self):
        return self.fps

def display_fps(frame, fps):
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame
