import cv2
import numpy as np
import json
import os
import threading
from emailAlert import email_alert  # Import the email_alert function

def load_polygons(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON from {file_path}. Returning an empty list.")
                return []
    else:
        return []

def save_polygons(file_path, polygons):
    with open(file_path, 'w') as file:
        json.dump(polygons, file)

def draw_polygon(event, x, y, flags, param):
    current_polygon = param['current_polygon']
    polygons = param['polygons']
    polygon_file = param['polygon_file']

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
        if len(current_polygon) == 4:
            polygons.append(current_polygon.copy())
            save_polygons(polygon_file, polygons)
            current_polygon.clear()
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_polygon.clear()

def inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(np.array(polygon), (point[0], point[1]), False)
    return result >= 0

def draw_polygons_on_frame(overlay, frame, polygons):
    for polygon in polygons:
        cv2.fillPoly(overlay, [np.array(polygon)], (0, 255, 0))
        cv2.polylines(overlay, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def process_detections(results, combined_frame_resized, frame_detected, polygons, target_classes, pygame, email_alert, photo_dir, count, number_of_photos):
    # YOLOv8 results are returned as a list of dictionaries
    for result in results:
        boxes = result.boxes.cpu().numpy()  # Boxes object
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            name = result.names[class_id]  # Get class name

            if name in target_classes:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.rectangle(combined_frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 3)
                text_offset_y = 20
                text_position = (x1 + 5, y1 + text_offset_y)
                cv2.putText(combined_frame_resized, name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.circle(combined_frame_resized, (center_x, center_y), 5, (0, 0, 255), -1)

                for polygon in polygons:
                    if inside_polygon((center_x, center_y), polygon) and name == 'person':
                        mask = np.zeros_like(frame_detected)
                        points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).reshape((-1, 1, 2))
                        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
                        frame_detected = cv2.bitwise_and(frame_detected, mask)

                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.play()

                        cv2.putText(combined_frame_resized, "Target", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(combined_frame_resized, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(combined_frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        count += 1

                        if count < number_of_photos:
                            photo_path = os.path.join(photo_dir, f"detected{count}.jpg")
                            threading.Thread(target=save_and_alert, args=(photo_path, combined_frame_resized)).start()
    return count

def save_and_alert(photo_path, image):
    cv2.imwrite(photo_path, image)
    email_alert("Human Detected", "We have detected human movement in your restricted area. Please check the attached image.", "duongphucnhatnguyen@gmail.com", photo_path)
    email_alert("Human Detected", "We have detected human movement in your restricted area. Please check the attached image.", "tranthitam76@gmail.com", photo_path)
    email_alert("Human Detected", "We have detected human movement in your restricted area. Please check the attached image.", "duongvanty@gmail.com", photo_path)
