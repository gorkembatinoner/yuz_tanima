import cv2
import face_recognition
import os
import numpy as np
import threading

# Yüz veritabanını yükle
def load_face_database(database_path="faces/"):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(database_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(database_path, filename)
            image = face_recognition.load_image_file(filepath)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
    return known_face_encodings, known_face_names


# Yüz veritabanını yükle
database_path = "faces/"
known_face_encodings, known_face_names = load_face_database(database_path)

# Video akışı için değişkenler
cap = cv2.VideoCapture(0)
process_frame_every = 2  # Her 2 karede bir işleme yap
resize_factor = 0.5  # Çözünürlüğü yarıya indirme
frame_counter = 0

# Son algılanan yüz bilgilerini saklamak için değişkenler
last_face_locations = []
last_face_names = []
frames_to_keep_faces = 5  # Algılanan yüzleri kaç kare boyunca tutacağımız
face_holding_frames = 0

# Çekilen kareler ve yüz algılama sonuçları için bir kilit (threading için)
lock = threading.Lock()

def process_frame(frame):
    """Yüz algılama ve tanıma işlemini yapan fonksiyon."""
    global last_face_locations, last_face_names, face_holding_frames
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Yüz algılama ve tanıma
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Bilinmiyor"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    # Sonuçları güncelle
    with lock:
        last_face_locations = face_locations
        last_face_names = face_names
        face_holding_frames = 0  # Saklama süresini sıfırla

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    # Görüntüyü ters çevir (ayna efekti)
    frame = cv2.flip(frame, 1)

    # Yüz algılama işlemini yalnızca belirli karelerde yap
    if frame_counter % process_frame_every == 0:
        # Yüz algılama için ayrı bir iş parçacığı başlat
        thread = threading.Thread(target=process_frame, args=(frame,))
        thread.start()
    else:
        # Önceki yüz bilgilerini göster
        with lock:
            face_holding_frames += 1
            if face_holding_frames > frames_to_keep_faces:
                last_face_locations = []
                last_face_names = []

    # Algılanan yüzlerin konumlarını çiz
    with lock:
        for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
            top = int(top / resize_factor)
            right = int(right / resize_factor)
            bottom = int(bottom / resize_factor)
            left = int(left / resize_factor)

            # Yüzün etrafına bir dikdörtgen çiz
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Yüzün altına adı ekle
            cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sonuçları göster
    cv2.imshow("Face Recognition", frame)

    # Çıkmak için 'ESC' tuşuna bas
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Kare sayacını artır
    frame_counter += 1

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
#a