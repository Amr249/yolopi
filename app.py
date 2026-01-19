from flask import Flask, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading

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

# Global stats
stats = {
    'object_count': 0,
    'fps': 0,
    'detections': []
}
stats_lock = threading.Lock()

# Bounding box colors
bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133),
    (88,159,106), (96,202,231), (159,124,168),
    (169,162,241), (98,118,150), (172,176,184)
]

def generate_frames():
    fps_buffer = []
    fps_avg_len = 30
    last_time = time.time()
    
    while True:
        current_time = time.time()
        success, frame = cap.read()
        if not success:
            break

        # YOLO inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        object_count = 0
        current_detections = []

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
            current_detections.append({
                'label': label,
                'confidence': int(conf*100)
            })

        # Calculate FPS
        dt = current_time - last_time
        if dt > 0:
            fps = 1.0 / dt
            fps_buffer.append(fps)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
        else:
            avg_fps = 0
        last_time = current_time

        # Update global stats
        with stats_lock:
            stats['object_count'] = object_count
            stats['fps'] = avg_fps
            stats['detections'] = current_detections

        # Draw stats on frame
        cv2.putText(frame, f"Objects: {object_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        # Encode frame as JPEG with lower quality for better performance
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YOLO Live Detection - Raspberry Pi</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            .header {
                background: rgba(255, 255, 255, 0.95);
                padding: 25px 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                margin-bottom: 25px;
                text-align: center;
            }
            
            .header h1 {
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 700;
            }
            
            .header p {
                color: #666;
                font-size: 1.1em;
            }
            
            .main-content {
                display: grid;
                grid-template-columns: 1fr 350px;
                gap: 25px;
                margin-bottom: 25px;
            }
            
            @media (max-width: 1024px) {
                .main-content {
                    grid-template-columns: 1fr;
                }
            }
            
            .video-container {
                background: rgba(255, 255, 255, 0.95);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                position: relative;
                overflow: hidden;
            }
            
            .video-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea, #764ba2);
            }
            
            .video-wrapper {
                position: relative;
                width: 100%;
                padding-bottom: 75%;
                background: #000;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
            }
            
            .video-wrapper img {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: contain;
            }
            
            .stats-panel {
                background: rgba(255, 255, 255, 0.95);
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                height: fit-content;
            }
            
            .stats-panel h2 {
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.5em;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            
            .stat-item {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: transform 0.2s;
            }
            
            .stat-item:hover {
                transform: translateX(5px);
            }
            
            .stat-label {
                font-weight: 600;
                color: #555;
            }
            
            .stat-value {
                font-size: 1.3em;
                font-weight: 700;
                color: #667eea;
            }
            
            .detections-list {
                margin-top: 20px;
            }
            
            .detections-list h3 {
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.2em;
            }
            
            .detection-item {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 10px;
                border-left: 4px solid #667eea;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .detection-label {
                font-weight: 600;
                color: #333;
            }
            
            .detection-confidence {
                background: #667eea;
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 600;
            }
            
            .no-detections {
                text-align: center;
                color: #999;
                padding: 20px;
                font-style: italic;
            }
            
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #4caf50;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .footer {
                background: rgba(255, 255, 255, 0.95);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                text-align: center;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 YOLO Live Detection</h1>
                <p><span class="status-indicator"></span>Real-time Object Detection System</p>
            </div>
            
            <div class="main-content">
                <div class="video-container">
                    <div class="video-wrapper">
                        <img src="/video_feed" alt="Live Camera Feed" id="videoFeed">
                    </div>
                </div>
                
                <div class="stats-panel">
                    <h2>📊 Statistics</h2>
                    <div class="stat-item">
                        <span class="stat-label">Objects Detected</span>
                        <span class="stat-value" id="objectCount">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">FPS</span>
                        <span class="stat-value" id="fps">0</span>
                    </div>
                    
                    <div class="detections-list">
                        <h3>🔍 Current Detections</h3>
                        <div id="detectionsList">
                            <div class="no-detections">No objects detected</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Powered by YOLO11n on Raspberry Pi | Real-time Object Detection</p>
            </div>
        </div>
        
        <script>
            // Update stats every 500ms
            setInterval(function() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('objectCount').textContent = data.object_count;
                        document.getElementById('fps').textContent = data.fps.toFixed(1);
                        
                        const detectionsList = document.getElementById('detectionsList');
                        if (data.detections && data.detections.length > 0) {
                            detectionsList.innerHTML = data.detections.map(det => 
                                `<div class="detection-item">
                                    <span class="detection-label">${det.label}</span>
                                    <span class="detection-confidence">${det.confidence}%</span>
                                </div>`
                            ).join('');
                        } else {
                            detectionsList.innerHTML = '<div class="no-detections">No objects detected</div>';
                        }
                    })
                    .catch(error => console.error('Error fetching stats:', error));
            }, 500);
            
            // Handle image load errors
            document.getElementById('videoFeed').onerror = function() {
                this.src = '/video_feed?' + new Date().getTime();
            };
        </script>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    with stats_lock:
        return jsonify(stats)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
