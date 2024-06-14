import cv2
import face_recognition
import multiprocessing as mp

# Function to encode known faces
def encode_known_faces(reference_data):
    for person in reference_data:
        person["encodings"] = []
        for image in person["images"]:
            encodings = face_recognition.face_encodings(image)
            if encodings:
                person["encodings"].append(encodings[0])
    return reference_data

# Function to process a frame and recognize faces
def process_frame(frame, reference_data, output_queue):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_encodings = face_recognition.face_encodings(rgb_small_frame)

    matched_person = None
    if face_encodings:
        for face_encoding in face_encodings:
            for person in reference_data:
                matches = face_recognition.compare_faces(person["encodings"], face_encoding)
                if True in matches:
                    matched_person = person["name"]
                    break
            if matched_person:
                break
    
    output_queue.put(matched_person)

def main():
    # Load reference images for multiple people
    reference_data = [
        {
            "name": "Person1",
            "images": [
                face_recognition.load_image_file("img/person1-1.jpg"),
                face_recognition.load_image_file("img/person1-2.jpg"),
                face_recognition.load_image_file("img/person1-3.jpg")
            ]
        },
        {
            "name": "Person2",
            "images": [
                face_recognition.load_image_file("img/person2-1.jpg"),
                face_recognition.load_image_file("img/person2-2.jpg"),
                face_recognition.load_image_file("img/person2-3.jpg")
            ]
        }
        # Add more people as needed
    ]

    # Encode the known faces
    reference_data = encode_known_faces(reference_data)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counter = 0
    output_queue = mp.Queue()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if counter % 30 == 0:
            process = mp.Process(target=process_frame, args=(frame.copy(), reference_data, output_queue))
            process.start()
            process.join()

        counter += 1

        if not output_queue.empty():
            matched_person = output_queue.get()
            if matched_person:
                cv2.putText(frame, matched_person, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "No Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Camera 1", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mp.freeze_support()
    main()
