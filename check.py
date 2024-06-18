import face_recognition
import cv2
import numpy as np
import os, sys
import math
import threading




def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    
    def __init__(self):
        self.encode_faces()
        self.run_recognition()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
        
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        
        print("Known faces:", self.known_face_names)


    def run_recognition(self):
        # Replace with the appropriate URLs for your IP cameras
        ip_camera_url_1 = "rtsp://admin:888888@10.0.0.152:8554/profile0"
        ip_camera_url_2 = "rtsp://admin:888888@10.0.0.122:8554/profile0"
        ip_camera_url_3 = "rtsp://admin:888888@10.0.0.139:8554/profile0"
        ip_camera_url_4 = "rtsp://admin:888888@10.0.0.122:8554/profile0"
        
        # Initialize the video capture objects
        cap1 = cv2.VideoCapture(ip_camera_url_1)
        cap2 = cv2.VideoCapture(ip_camera_url_2)
        cap3 = cv2.VideoCapture(ip_camera_url_3)
        cap4 = cv2.VideoCapture(ip_camera_url_4)
        
        if not cap1.isOpened():
            print("Error: Could not open video stream from IP camera 1.")
            exit()

        if not cap2.isOpened():
            print("Error: Could not open video stream from IP camera 2.")
            exit()

        if not cap3.isOpened():
            print("Error: Could not open video stream from IP camera 3.")
            exit()

        if not cap4.isOpened():
            print("Error: Could not open video stream from webcam.")
            exit()
        
        frame1 = None
        frame2 = None
        frame3 = None
        frame4 = None

        ret1 = False
        ret2 = False
        ret3 = False
        ret4 = False

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

        # Define a function to capture frames from the fourth camera
        def capture_camera4():
            global frame4, ret4
            while True:
                ret4, frame4 = cap4.read()
        # Start threads for capturing frames from all cameras
        thread1 = threading.Thread(target=capture_camera1)
        thread2 = threading.Thread(target=capture_camera2)
        thread3 = threading.Thread(target=capture_camera3)
        thread4 = threading.Thread(target=capture_camera4)

        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        
        # Create a resizable window
        cv2.namedWindow('Combined IP Camera Feeds', cv2.WINDOW_NORMAL)
        
        while True:
            if ret1 and ret2 and ret3 and ret4:
                # Resize frames to the same width, keeping a reasonable size
                new_width = 640  # Adjust the target width as needed
                frame1_resized = cv2.resize(frame1, (new_width, int(frame1.shape[0] * new_width / frame1.shape[1])))
                frame2_resized = cv2.resize(frame2, (new_width, int(frame2.shape[0] * new_width / frame2.shape[1])))
                frame3_resized = cv2.resize(frame3, (new_width, int(frame3.shape[0] * new_width / frame3.shape[1])))
                frame4_resized = cv2.resize(frame4, (new_width, int(frame4.shape[0] * new_width / frame4.shape[1])))

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
            
                if self.process_current_frame:
                    small_frame = cv2.resize(combined_frame_resized, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                    # Find all faces
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = 'Unknown'
                        confidence = 'Unknown'
                    
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])
                        
                        self.face_names.append(f'{name} ({confidence})')

                self.process_current_frame = not self.process_current_frame

                # Display annotations
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(combined_frame_resized, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(combined_frame_resized, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                    cv2.putText(combined_frame_resized, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                # Display the frame
                cv2.imshow('IP Camera Feed', combined_frame_resized)

                # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything is done, release the captures
        cap1.release()
        cap2.release()
        cap3.release()
        cap4.release()       
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()