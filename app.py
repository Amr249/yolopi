from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO
import time

app = Flask(__name__)

# ==============================
# LOAD MODEL (YOUR FILE)
# ==============================
MODEL_PATH = "yolo11n.pt"
model = YOLO(MODEL_PATH)
labels = model.names

# ==============================
# CAMERA (USB CAM OR PI CAM)
# ==============================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Bounding box colors
bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133),
    (88,159,106), (96,202,231), (159,124,168),
    (169,162,241), (98,118,150), (172,176,184)
]

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        object_count = 0

        for det in detections:
            conf = det.conf.item()
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy().squeeze())
            cls_id = int(det.cls.item())
            label = labels[cls_id]

            color = bbox_colors[cls_id % len(bbox_colors)]
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            text = f"{label} {int(conf*100)}%"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            object_count += 1

        cv2.putText(frame, f"Objects: {object_count}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>YOLO Raspberry Pi</title>
      </head>
      <body>
        <h2>YOLO Live Detection</h2>
        <img src="/video_feed">
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
