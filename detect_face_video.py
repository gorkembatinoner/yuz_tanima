import cv2
import face_recognition
import threading
import numpy as np
from database_operations import load_faces_from_database

# Veritabanından yüz verilerini yükle
known_face_encodings, known_face_names = load_faces_from_database()

# Verilerin yüklendiğini kontrol et
if not known_face_encodings:
    print("Veritabanında yüz verisi bulunamadı.")
    exit(1)  # Eğer veriler boşsa çık

cap = cv2.VideoCapture(0)
process_frame_every = 2
resize_factor = 0.5
frame_counter = 0
last_face_locations = []
last_face_names = []
frames_to_keep_faces = 5
face_holding_frames = 0
lock = threading.Lock()

def process_frame(frame):
    global last_face_locations, last_face_names, face_holding_frames
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if matches and len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        face_names.append(name)

    with lock:
        last_face_locations = face_locations
        last_face_names = face_names
        face_holding_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if frame_counter % process_frame_every == 0:
        thread = threading.Thread(target=process_frame, args=(frame,))
        thread.start()
    else:
        with lock:
            face_holding_frames += 1
            if face_holding_frames > frames_to_keep_faces:
                last_face_locations = []
                last_face_names = []

    with lock:
        for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
            top = int(top / resize_factor)
            right = int(right / resize_factor)
            bottom = int(bottom / resize_factor)
            left = int(left / resize_factor)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    frame_counter += 1

cap.release()
cv2.destroyAllWindows()
