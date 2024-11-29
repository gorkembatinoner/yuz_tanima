import cv2
import time
import sys

def benchmark(num_times, image_path='test.jpg'):
    start = time.perf_counter()  # Daha taşınabilir
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    overhead_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(num_times):
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_detect_time = time.perf_counter() - start

    return overhead_time, face_detect_time

if __name__ == '__main__':
    num_times = int(sys.argv[1])
    image_path = sys.argv[2] if len(sys.argv) > 2 else 'test.jpg'
    overhead_time, face_detect_time = benchmark(num_times, image_path)
    print(f"Overhead time: {overhead_time:.2f}s")
    print(f"Face detection time for {num_times} runs: {face_detect_time:.2f}s")
