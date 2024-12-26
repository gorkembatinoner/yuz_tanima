# emotion_detection.py
from deepface import DeepFace
from collections import deque
import cv2
from liveness_detection import check_liveness  # Canlılık tespiti için yeni eklenen modül

# Duygu tespiti için bir kuyruğu (deque) global olarak tanımlıyoruz
emotion_queue = deque(maxlen=15)  # Son 15 karenin duygularını sakla

def detect_expression_and_liveness(gray_roi, color_frame, x, y, w, h, eye_cascade):
    global emotion_queue

    # DeepFace ile duygu tespiti
    face_roi = color_frame[y:y + h, x:x + w]  # Yüz bölgesini al
    try:
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
    except Exception:
        dominant_emotion = "Notr"  # Duygu tespiti yapılamazsa varsayılan değer

    # Yeni duyguyu kuyruğa ekle
    emotion_queue.append(dominant_emotion)

    # Kuyruktaki en sık geçen duyguyu seç
    if emotion_queue:
        stable_emotion = max(set(emotion_queue), key=emotion_queue.count)
    else:
        stable_emotion = "Notr"

    # Canlılık testi
    liveness_status = "Canli" if check_liveness(color_frame[y:y+h, x:x+w], eye_cascade) else "Canli Degil"

    # Duygu ve canlılık durumunu çerçeve üzerine yaz
    cv2.putText(color_frame, f"{stable_emotion} - {liveness_status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return stable_emotion, liveness_status
