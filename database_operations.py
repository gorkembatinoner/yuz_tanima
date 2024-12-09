import mysql.connector
import numpy as np
import json

def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Rapertuar.1",
        database="FaceRecognitionDB"
    )

def add_face_to_database(name, encoding):
    connection = connect_to_database()
    with connection.cursor() as cursor:
        sql = "INSERT INTO faces (name, encoding) VALUES (%s, %s)"
        encoding_json = json.dumps(encoding.tolist())  # JSON formatına çevir
        cursor.execute(sql, (name, encoding_json))
    connection.commit()
    connection.close()

def load_faces_from_database():
    connection = connect_to_database()
    with connection.cursor() as cursor:
        sql = "SELECT name, encoding FROM faces"
        cursor.execute(sql)
        results = cursor.fetchall()

    connection.close()
    known_face_encodings = []
    known_face_names = []
    for name, encoding_json in results:
        known_face_encodings.append(np.array(json.loads(encoding_json)))  # JSON'dan yükle
        known_face_names.append(name)
    return known_face_encodings, known_face_names

