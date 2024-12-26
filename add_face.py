import face_recognition
import numpy as np
import os
from database_operations import add_face_to_database  # Doğru modülü içeri aktar


# Fotoğrafı yükleyip encoding verisini çıkarma fonksiyonu
def process_and_add_face(image_path):
    # Fotoğrafı yükle
    image = face_recognition.load_image_file(image_path)

    # Yüz encoding verisini çıkar
    encoding = face_recognition.face_encodings(image)

    if encoding:
        # Kullanıcıdan bir isim iste
        name = input("Bu yüz için bir isim girin (örneğin: Nilay): ")

        # Encoding verisini veritabanına ekle
        add_face_to_database(name, np.array(encoding[0]))
        print(f"Veri başarıyla eklendi: {name}")
    else:
        print("Yüz tespiti yapılmadı.")

# Bu fonksiyonu çağırarak yeni fotoğrafı ekleyebilirsiniz
if __name__ == "__main__":
    image_path = input("Fotoğraf dosyasının yolunu girin (örneğin: faces/nilay.png): ")
    process_and_add_face(image_path)
