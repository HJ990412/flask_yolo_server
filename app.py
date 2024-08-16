import torch
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import mysql.connector
from mysql.connector import Error
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 모델 불러오기 (서버 시작 시 한 번만 실행)
model_path = './weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# MySQL 데이터베이스 연결 설정
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="43.202.94.152",
            user="test1",
            password="12345",
            database="testdb"
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error: {e}")
        return None

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    connection = None
    cursor = None
    try:
        user_id = request.json.get('user_id', '0')  # 로그인하지 않은 경우 기본값으로 '0' 사용
        print(f"user_id: {user_id}")
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 이미지를 PIL 형식으로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # YOLOv5 모델로 예측
        results = model(img_pil)
        results.render()  # 결과를 이미지에 렌더링

        # 학습한 클래스 이름 출력
        class_names = results.names

        # 예측 결과 값을 추출
        predictions = []
        for *box, conf, cls in results.xyxy[0].tolist():
            prediction = {
                "class": class_names[int(cls)],
                "confidence": conf,
                "box": box
            }
            predictions.append(prediction)
            print(prediction)  # 예측 결과 값을 출력

        message = predictions[0]['class'] if predictions else 'No objects detected'

        # 결과 이미지를 가져와 OpenCV 형식으로 변환
        img_result = results.ims[0]
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

        # 처리된 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.jpg', img_result)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # MySQL 데이터베이스 연결
        connection = create_connection()
        if connection is None:
            raise Exception("Failed to connect to the database")

        cursor = connection.cursor()
        query = "INSERT INTO images (user_id, message, image) VALUES (%s, %s, %s)"
        cursor.execute(query, (user_id, message, processed_image_base64))
        connection.commit()

        result = {
            'status': 'success',
            'message': message,
            'image_id': cursor.lastrowid,
            'processed_image': processed_image_base64  # 예측 결과 이미지 반환
        }
    except Exception as e:
        result = {'status': 'error', 'message': str(e)}
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


