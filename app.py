from flask import Flask, Response, jsonify
import cv2
import numpy as np
import time
import threading
from yolo3d_detector import YOLODetector, draw_detection_with_depth

app = Flask(__name__)

# ==============================
# INITIALIZE YOLO-D DETECTOR (FPS OPTIMIZED + PHASE 2)
# ==============================
# FPS OPTIMIZATION CONSTANTS (PHASE 1)
PROCESSING_WIDTH = 640   # Processing resolution width (lower = faster)
PROCESSING_HEIGHT = 384  # Processing resolution height (lower = faster)
DEPTH_THROTTLE_INTERVAL = 4  # Run depth every N frames (higher = faster)

# PHASE 2: Depth normalization and meter estimation configuration
ENABLE_METERS = False  # Set True to enable meter conversion (requires calibration)
CALIBRATION_NEAR_M = 0.5  # Reference distance 1 (meters)
CALIBRATION_FAR_M = 2.0    # Reference distance 2 (meters)
# Calibration values will be set during calibration (if enabled)

# PHASE 3A: Distance-based filtering and robotics actions
ENABLE_DISTANCE_FILTERING = True  # Enable zone-based filtering
ALLOW_ZONES = ["Near", "Medium"]  # Zones to include in filtered detections (exclude "Far")
TARGET_CLASSES = []  # Empty = allow all classes; otherwise only these classes trigger actions
SHOW_ALL_DETECTIONS = False  # If True, show all detections in UI (debug mode); else show filtered only
ACTION_DEBOUNCE_FRAMES = 3  # Action must persist for N frames before changing (reduces flicker)

print("Initializing YOLO-D detector (FPS optimized + PHASE 2)...")
detector = YOLODetector(
    yolo_model_path="yolo11n.pt",
    depth_model_name="depth-anything/Depth-Anything-V2-Small-hf",
    depth_input_size=(384, 384),  # Depth model input size (smaller = faster)
    depth_throttle=DEPTH_THROTTLE_INTERVAL,  # Process depth every N frames
    conf_threshold=0.5,
    processing_resolution=(PROCESSING_WIDTH, PROCESSING_HEIGHT)  # FPS optimization
)

# PHASE 2: Configure meter estimation (if enabled)
if ENABLE_METERS:
    detector.enable_meters = True
    detector.calibration_near_m = CALIBRATION_NEAR_M
    detector.calibration_far_m = CALIBRATION_FAR_M
    print(f"✓ Meter estimation enabled (calibration: {CALIBRATION_NEAR_M}m - {CALIBRATION_FAR_M}m)")
else:
    print("✓ Relative depth mode (normalized [0,1])")

print("✓ YOLO-D detector ready (FPS optimized + PHASE 2)")

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
    'detections': [],
    'filtered_count': 0,
    'nearest_object': None,
    'action': 'PROCEED',
    'action_reason': 'Initializing...',
    'depth_mode': 'Relative',
    'normalization_method': 'Percentile (p5..p95)'
}
stats_lock = threading.Lock()

# PHASE 3A: Action state for debouncing
action_history = []  # Store last N actions for debouncing
action_history_max = ACTION_DEBOUNCE_FRAMES

# Global stats
stats = {
    'object_count': 0,
    'fps': 0,
    'detections': [],
    'filtered_count': 0,
    'nearest_object': None,
    'action': 'PROCEED',
    'action_reason': 'Initializing...',
    'depth_mode': 'Relative',
    'normalization_method': 'Percentile (p5..p95)'
}
stats_lock = threading.Lock()

# PHASE 3A: Action state for debouncing
action_history = []  # Store last N actions for debouncing
action_history_max = ACTION_DEBOUNCE_FRAMES

# Bounding box colors
bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133),
    (88,159,106), (96,202,231), (159,124,168),
    (169,162,241), (98,118,150), (172,176,184)
]

def generate_frames():
    """
    Generate video frames with YOLO-D (depth-aware) detection.
    FPS OPTIMIZED for Raspberry Pi CPU.
    
    OPTIMIZATIONS APPLIED:
    1. Frame resizing to processing_resolution (640x384) - done in detector
    2. Depth throttling (runs every N frames, cached otherwise)
    3. FPS measurement and overlay
    4. torch.no_grad() used in all inference operations
    """
    # FPS measurement setup
    fps_buffer = []
    fps_avg_len = 30  # Average over last 30 frames
    last_time = time.time()
    frame_counter = 0
    
    while True:
        # FPS OPTIMIZATION: Measure time delta for FPS calculation
        current_time = time.time()
        frame_start_time = current_time
        
        success, frame = cap.read()
        if not success:
            break

        # OPTIMIZATION: Detector handles frame resizing internally
        # Depth throttling is controlled by detector's frame counter
        detections, depth_map, display_frame = detector.detect(frame, update_depth=None)

        # PHASE 3A: Filter detections and compute actions
        filtered_detections = detections
        if ENABLE_DISTANCE_FILTERING:
            # Filter by zone
            filtered_detections = detector.filter_detections_by_zone(detections, ALLOW_ZONES)
            # Filter by class (if TARGET_CLASSES specified)
            if TARGET_CLASSES:
                filtered_detections = detector.filter_detections_by_class(filtered_detections, TARGET_CLASSES)
        
        # Find nearest object
        nearest_object = detector.find_nearest_object(filtered_detections)
        
        # Compute action (pass frame width for avoid direction calculation)
        h, w = display_frame.shape[:2]
        action, action_reason = detector.compute_action(filtered_detections, nearest_object, frame_width=w)
        
        # PHASE 3A: Action debouncing (prevent flicker)
        action_history.append(action)
        if len(action_history) > action_history_max:
            action_history.pop(0)
        
        # Use most common action in history (debounced)
        if len(action_history) >= action_history_max:
            from collections import Counter
            action_counts = Counter(action_history)
            debounced_action = action_counts.most_common(1)[0][0]
            # Only update reason if action changed
            if debounced_action != action:
                # Recompute reason for debounced action
                h, w = display_frame.shape[:2]
                action, action_reason = detector.compute_action(filtered_detections, nearest_object, frame_width=w)
            action = debounced_action

        # Determine which detections to display
        display_detections = filtered_detections if not SHOW_ALL_DETECTIONS else detections
        
        object_count = len(display_detections)
        filtered_count = len(filtered_detections)
        current_detections = []

        # Draw detections with depth information
        for detection in display_detections:
            cls_id = detection['class_id']
            color = bbox_colors[cls_id % len(bbox_colors)]
            
            # Draw detection with depth overlay
            draw_detection_with_depth(display_frame, detection, color)
            
            object_count += 1
            
            # PHASE 2: Include zone and distance information in stats
            det_info = {
                'label': detection['class'],
                'confidence': int(detection['confidence'] * 100),
                'zone': detection.get('zone', 'Unknown'),
                'normalized_depth': round(detection.get('normalized_depth', 0), 2) if detection.get('normalized_depth') is not None else None
            }
            
            # Add distance in meters if available
            if detection.get('distance_m') is not None:
                det_info['distance_m'] = round(detection['distance_m'], 2)
            elif detection.get('depth') is not None:  # Backward compatibility
                det_info['depth'] = round(detection['depth'], 2)
            
            current_detections.append(det_info)

        # FPS OPTIMIZATION: Calculate FPS using time delta
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
        
        # PHASE 3A: Update global stats with action and nearest object info
        with stats_lock:
            stats['object_count'] = object_count
            stats['fps'] = avg_fps
            stats['detections'] = current_detections
            stats['filtered_count'] = filtered_count
            stats['nearest_object'] = nearest_object
            stats['action'] = action
            stats['action_reason'] = action_reason
            stats['depth_mode'] = "Meters" if detector.enable_meters else "Relative"
            stats['normalization_method'] = "Percentile (p5..p95)"

        # FPS OPTIMIZATION: Draw FPS overlay on frame
        cv2.putText(display_frame, f"Objects: {object_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        
        # PHASE 2: Draw depth mode indicator
        if depth_map is not None:
            depth_status = "Depth: ON"
            depth_color = (0, 255, 0)
        else:
            depth_status = "Depth: CACHED"
            depth_color = (255, 255, 0)
        
        cv2.putText(display_frame, depth_status,
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, depth_color, 2)
        
        # PHASE 2: Display depth mode (Relative vs Calibrated)
        depth_mode = "Meters" if detector.enable_meters else "Relative"
        cv2.putText(display_frame, f"Mode: {depth_mode}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
        
        # PHASE 3A: Display action and nearest object
        action_colors = {
            'STOP': (0, 0, 255),           # Red
            'SLOW_DOWN': (0, 165, 255),    # Orange
            'PROCEED': (0, 255, 0),        # Green
            'AVOID_LEFT': (255, 0, 255),   # Magenta
            'AVOID_RIGHT': (255, 0, 255)   # Magenta
        }
        action_color = action_colors.get(action, (255, 255, 255))
        
        # Draw action prominently
        h, w = display_frame.shape[:2]
        cv2.putText(display_frame, f"ACTION: {action}",
                   (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, action_color, 3)
        cv2.putText(display_frame, action_reason,
                   (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 2)
        
        # Draw nearest object info
        if nearest_object:
            dist_text = f"{nearest_object['distance']:.2f}m" if nearest_object.get('distance_m') else f"Rel:{nearest_object.get('normalized_depth', 0):.2f}"
            cv2.putText(display_frame, f"NEAREST: {nearest_object['class']} {dist_text} ({nearest_object['zone']})",
                       (w - 300, 90), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 255), 2)

        # Encode frame as JPEG with lower quality for better performance
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        frame_counter += 1
        last_time = current_time

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
