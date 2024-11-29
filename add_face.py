import face_recognition
import numpy as np
import os
from database_operations import add_face_to_database  # Doğru modülü içeri aktar

# Fotoğrafı yükleyip encoding verisini çıkarma fonksiyonu
def process_and_add_face(image_path):
   
    image = face_recognition.load_image_file(image_path)
  
    encoding = face_recognition.face_encodings(image)

    if encoding:
        # Encoding verisini veritabanına ekle
        name = os.path.basename(image_path).split('.')[0]  # Fotoğrafın ismini al (örneğin: 'nilay')
        add_face_to_database(name, np.array(encoding[0]))
        print(f"Veri başarıyla eklendi: {name}")
    else:
        print("Yüz tespiti yapılmadı.")

# Bu fonksiyonu çağırarak yeni fotoğrafı ekleyebilirsiniz
if __name__ == "__main__":
    image_path = input("Fotoğraf dosyasının yolunu girin (örneğin: faces/nilay.png): ")
    process_and_add_face(image_path)


