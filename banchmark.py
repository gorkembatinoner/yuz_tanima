import cv2
import time
import sys

def benchmark(num_times, image_path='test.jpg'):
    start = time.perf_counter()  # Daha taşınabilir
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier('smile.xml')
    eye_cascade = cv2.CascadeClassifier("eye.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    overhead_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(num_times):
        # Yüzleri tespit et
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            # Her yüzün etrafını çizin
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Yüzün içindeki bölgeyi kes
            face_roi = gray[y:y + h, x:x + w]
            # Gülümseme tespiti
            smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                # Gülümsemeyi işaretleyin
                cv2.rectangle(img, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

    face_detect_time = time.perf_counter() - start

    return overhead_time, face_detect_time

if __name__ == '__main__':
    num_times = int(sys.argv[1])
    image_path = sys.argv[2] if len(sys.argv) > 2 else 'test.jpg'
    overhead_time, face_detect_time = benchmark(num_times, image_path)
    print(f"Overhead time: {overhead_time:.2f}s")
    print(f"Face detection time for {num_times} runs: {face_detect_time:.2f}s")

    # Sonuçları görüntüle
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
