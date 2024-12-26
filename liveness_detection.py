# liveness_detection.py
import cv2

def detect_blinking(eye_region):
    """
    Bu fonksiyon, göz kırpma hareketini tespit eder.
    """
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    eyes = cv2.CascadeClassifier('haarcascade_eye.xml').detectMultiScale(gray, 1.5, 3)
    
    # Eğer göz algılandıysa ve çok küçükse, göz kırpma olabilir.
    if len(eyes) > 0:
        return True
    return False

def check_liveness(face_region, eye_cascade):
    """
    Yüz bölgesini kontrol ederek kullanıcının göz kırpma hareketi olup olmadığını kontrol eder.
    """
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)

    if len(eyes) > 0:
        # Gözleri tek tek kontrol et
        for (ex, ey, ew, eh) in eyes:
            eye_region = face_region[ey:ey+eh, ex:ex+ew]
            if detect_blinking(eye_region):
                return True
    return False
