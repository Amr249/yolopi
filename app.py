"""
PHASE 1: YOLO-only baseline for stable real-time perception on Raspberry Pi
Clean, fast, and stable YOLO object detection pipeline running on CPU only.
No depth estimation, no ONNX, no ROS, no experimental features.
"""
from flask import Flask, Response, jsonify
import cv2
import numpy as np
import time
import threading
from yolo3d_detector import YOLODetector

app = Flask(__name__)

# ==============================
# PHASE 1: YOLO-ONLY CONFIGURATION
# ==============================
# PHASE 1: Stable YOLO-only perception on Raspberry Pi
ENABLE_DEPTH = False  # PHASE 1: Disable depth completely
ENABLE_ONNX = False  # PHASE 1: Disable ONNX completely

# PHASE 1: Core configuration constants
PROCESSING_WIDTH = 640   # Processing resolution width
PROCESSING_HEIGHT = 384  # Processing resolution height
FRAME_SKIP = 1  # Process every Nth frame (1 = every frame, 2 = every other frame, etc.)
SHOW_FPS = True  # Display FPS overlay on output frame
CONF_THRESHOLD = 0.5  # Confidence threshold for detections
JPEG_QUALITY = 85  # JPEG quality for streaming

print("=" * 60)
print("PHASE 1: YOLO-only baseline for stable real-time perception")
print("=" * 60)
print(f"✓ Depth estimation: DISABLED")
print(f"✓ ONNX Runtime: DISABLED")
print(f"✓ Processing resolution: {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}")
print(f"✓ Frame skip: {FRAME_SKIP}")
print("=" * 60)

# Initialize YOLO detector (depth disabled in PHASE 1)
print("\nInitializing YOLO detector...")
detector = YOLODetector(
    yolo_model_path="yolo11n.pt",
    depth_model_name=None,  # PHASE 1: No depth model
    depth_input_size=(256, 256),  # Not used in PHASE 1
    depth_throttle=1,  # Not used in PHASE 1
    conf_threshold=CONF_THRESHOLD,
    processing_resolution=(PROCESSING_WIDTH, PROCESSING_HEIGHT),
    enable_depth=False  # PHASE 1: Explicitly disable depth
)
print("✓ YOLO detector ready\n")

print("✓ YOLO-D detector ready (FPS optimized + PHASE 2)")

# ==============================
# CAMERA (USB CAM OR PI CAM)
# ==============================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# PHASE 1: Global stats (simplified - no depth, zones, or actions)
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
    """
    PHASE 1: Generate video frames with YOLO-only detection.
    Clean, fast, and stable real-time perception on Raspberry Pi.
    
    OPTIMIZATIONS APPLIED:
    1. Frame resizing to processing_resolution (640×384) - done once after capture
    2. torch.no_grad() used for YOLO inference
    3. Optional frame skipping (process every Nth frame)
    4. FPS measurement with rolling average
    """
    # FPS measurement setup
    fps_buffer = []
    fps_avg_len = 30  # Average over last 30 frames
    frame_counter = 0
    
    while True:
        # Measure frame processing time for FPS calculation
        frame_start_time = time.time()
        
        # Read frame from camera
        success, frame = cap.read()
        if not success:
            break
        
        # PHASE 1: Optional frame skipping (process every Nth frame)
        frame_counter += 1
        if FRAME_SKIP > 1 and (frame_counter % FRAME_SKIP != 0):
            # Skip this frame - use last frame or show empty
            continue

        # PHASE 1: Resize frame ONCE after capture for processing
        original_frame = frame.copy()
        h_orig, w_orig = frame.shape[:2]
        if (w_orig, h_orig) != (PROCESSING_WIDTH, PROCESSING_HEIGHT):
            frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # PHASE 1: YOLO inference only (no depth)
        detections, depth_map, display_frame = detector.detect(frame, update_depth=False)
        
        # PHASE 1: Simple detection processing (no filtering, no zones, no actions)
        object_count = len(detections)
        current_detections = []
        
        # PHASE 1: Draw simple bounding boxes and labels
        for detection in detections:
            cls_id = detection['class_id']
            bbox = detection['bbox']
            label = detection['class']
            conf = detection['confidence']
            color = bbox_colors[cls_id % len(bbox_colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label_text = f"{label} {int(conf * 100)}%"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), color, -1)
            cv2.putText(display_frame, label_text, (x1 + 2, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Store detection info for stats
            det_info = {
                'label': label,
                'confidence': int(conf * 100)
            }
            current_detections.append(det_info)

        # PHASE 1: Calculate FPS using time delta
        frame_end_time = time.time()
        frame_dt = frame_end_time - frame_start_time
        
        if frame_dt > 0:
            instant_fps = 1.0 / frame_dt
            fps_buffer.append(instant_fps)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
        else:
            avg_fps = 0
        
        # PHASE 1: Update global stats (simplified)
        with stats_lock:
            stats['object_count'] = object_count
            stats['fps'] = avg_fps
            stats['detections'] = current_detections
        
        # PHASE 1: Draw FPS and object count overlay (if enabled)
        if SHOW_FPS:
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Objects: {object_count}",
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2)
        
        # PHASE 1: Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YOLO-D Live Detection - Raspberry Pi</title>
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
                <h1>🎯 YOLO-D Live Detection</h1>
                <p><span class="status-indicator"></span>Real-time Depth-Aware Object Detection</p>
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
                    <div class="stat-item" id="depthModeItem" style="display: none;">
                        <span class="stat-label">Depth Mode</span>
                        <span class="stat-value" id="depthMode">Relative</span>
                    </div>
                    <div class="stat-item" id="normalizationItem" style="display: none;">
                        <span class="stat-label">Normalization</span>
                        <span class="stat-value" id="normalization">Percentile (p5..p95)</span>
                    </div>
                    
                    <!-- PHASE 3A: Robot Mode Panel -->
                    <div style="background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2); margin-top: 20px;">
                        <h2 style="color: #667eea; margin-bottom: 15px; font-size: 1.5em; border-bottom: 2px solid #667eea; padding-bottom: 10px;">🤖 Robot Mode</h2>
                        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 10px; margin-bottom: 15px; text-align: center;">
                            <div style="font-size: 2em; font-weight: 700; color: #667eea; margin-bottom: 10px;" id="robotAction">PROCEED</div>
                            <div style="font-size: 0.9em; color: #666;" id="robotReason">Initializing...</div>
                        </div>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px;">
                            <div style="font-weight: 600; color: #333; margin-bottom: 10px;">Nearest Object:</div>
                            <div id="nearestObjectInfo" style="color: #666; font-size: 0.9em;">None detected</div>
                        </div>
                        <div style="margin-top: 15px; font-size: 0.85em; color: #999;">
                            Filtered: <span id="filteredCount">0</span> / Total: <span id="totalCount">0</span>
                        </div>
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
                <p>Powered by YOLO-D (YOLO11n + Depth) on Raspberry Pi | Real-time Depth-Aware Detection</p>
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
                        
                        // PHASE 2: Update depth mode and normalization info
                        if (data.depth_mode) {
                            document.getElementById('depthMode').textContent = data.depth_mode;
                            document.getElementById('depthModeItem').style.display = 'flex';
                        }
                        if (data.normalization_method) {
                            document.getElementById('normalization').textContent = data.normalization_method;
                            document.getElementById('normalizationItem').style.display = 'flex';
                        }
                        
                        // PHASE 3A: Update robot mode panel
                        if (data.action) {
                            const actionElement = document.getElementById('robotAction');
                            actionElement.textContent = data.action;
                            // Color code actions
                            const actionColors = {
                                'STOP': '#ff0000',
                                'SLOW_DOWN': '#ff9800',
                                'PROCEED': '#4caf50',
                                'AVOID_LEFT': '#9c27b0',
                                'AVOID_RIGHT': '#9c27b0'
                            };
                            actionElement.style.color = actionColors[data.action] || '#667eea';
                        }
                        if (data.action_reason) {
                            document.getElementById('robotReason').textContent = data.action_reason;
                        }
                        if (data.nearest_object) {
                            const obj = data.nearest_object;
                            const distText = obj.distance_m ? `${obj.distance_m.toFixed(2)}m` : `Rel:${obj.normalized_depth?.toFixed(2) || 'N/A'}`;
                            document.getElementById('nearestObjectInfo').innerHTML = 
                                `<strong>${obj.class}</strong> (${obj.confidence}%)<br>` +
                                `Distance: ${distText}<br>` +
                                `Zone: <span style="color: ${obj.zone === 'Near' ? '#4caf50' : obj.zone === 'Medium' ? '#ffc107' : '#ff9800'}">${obj.zone}</span>`;
                        } else {
                            document.getElementById('nearestObjectInfo').textContent = 'None detected';
                        }
                        if (data.filtered_count !== undefined) {
                            document.getElementById('filteredCount').textContent = data.filtered_count;
                        }
                        if (data.object_count !== undefined) {
                            document.getElementById('totalCount').textContent = data.object_count;
                        }
                        
                        const detectionsList = document.getElementById('detectionsList');
                        if (data.detections && data.detections.length > 0) {
                            detectionsList.innerHTML = data.detections.map(det => {
                                // PHASE 2: Format distance display
                                let distanceText = '';
                                if (det.distance_m !== undefined) {
                                    distanceText = `📏 ${det.distance_m}m`;
                                } else if (det.depth !== undefined) {
                                    distanceText = `📏 ${det.depth}m`;
                                } else if (det.normalized_depth !== undefined) {
                                    distanceText = `📏 Rel: ${det.normalized_depth}`;
                                } else {
                                    distanceText = '📏 N/A';
                                }
                                
                                // PHASE 2: Zone display with color
                                const zoneColors = {
                                    'Near': '#4caf50',
                                    'Medium': '#ffc107',
                                    'Far': '#ff9800',
                                    'Unknown': '#999'
                                };
                                const zoneColor = zoneColors[det.zone] || '#999';
                                
                                return `<div class="detection-item">
                                    <div style="display: flex; flex-direction: column; gap: 5px;">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <span class="detection-label">${det.label}</span>
                                            <span class="detection-confidence">${det.confidence}%</span>
                                        </div>
                                        <div style="font-size: 0.85em; color: #667eea; font-weight: 600;">
                                            ${distanceText}
                                        </div>
                                        <div style="font-size: 0.85em; color: ${zoneColor}; font-weight: 600;">
                                            🎯 Zone: ${det.zone}
                                        </div>
                                    </div>
                                </div>`;
                            }).join('');
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

@app.route('/robot_state')
def get_robot_state():
    """
    PHASE 3A: JSON endpoint for robot state.
    Returns current action, nearest object, and system status.
    """
    import time
    with stats_lock:
        robot_state = {
            'timestamp': time.time(),
            'fps': stats.get('fps', 0),
            'depth_mode': stats.get('depth_mode', 'Relative'),
            'action': stats.get('action', 'PROCEED'),
            'reason': stats.get('action_reason', 'Unknown'),
            'nearest_object': stats.get('nearest_object'),
            'filtered_count': stats.get('filtered_count', 0),
            'total_count': stats.get('object_count', 0),
            'normalization_method': stats.get('normalization_method', 'Percentile (p5..p95)')
        }
        return jsonify(robot_state)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
