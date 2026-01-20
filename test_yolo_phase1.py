"""
PHASE 1: YOLO-only baseline test script for stable real-time perception on Raspberry Pi.
Clean, fast, and stable YOLO object detection pipeline running on CPU only.
No depth estimation, no ONNX, no ROS, no experimental features.
"""

import cv2
import numpy as np
import time
from yolo3d_detector import YOLODetector

# PHASE 1: Configuration constants
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 384
FRAME_SKIP = 1  # Process every frame (1 = every frame, 2 = every other frame, etc.)
SHOW_FPS = True
CONF_THRESHOLD = 0.5

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

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Bounding box colors
bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133),
    (88,159,106), (96,202,231), (159,124,168),
    (169,162,241), (98,118,150), (172,176,184)
]

# FPS tracking
fps_buffer = []
fps_avg_len = 30

print("\nStarting PHASE 1 YOLO detection...")
print("Press 'q' to quit, 's' to save screenshot\n")

frame_counter = 0

while True:
    start_time = time.time()
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    # PHASE 1: Optional frame skipping (process every Nth frame)
    frame_counter += 1
    if FRAME_SKIP > 1 and (frame_counter % FRAME_SKIP != 0):
        # Skip this frame
        continue
    
    # PHASE 1: Resize frame ONCE after capture for processing
    h_orig, w_orig = frame.shape[:2]
    if (w_orig, h_orig) != (PROCESSING_WIDTH, PROCESSING_HEIGHT):
        frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # PHASE 1: YOLO inference only (no depth)
    detections, depth_map, display_frame = detector.detect(frame, update_depth=False)
    
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
    
    # PHASE 1: Calculate and display FPS
    dt = time.time() - start_time
    if dt > 0:
        fps = 1.0 / dt
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
    else:
        avg_fps = 0
    
    # PHASE 1: Draw FPS and object count overlay (if enabled)
    if SHOW_FPS:
        h, w = display_frame.shape[:2]
        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Objects: {len(detections)}", (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display frame
    cv2.imshow("PHASE 1: YOLO Detection", display_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('s'):
        filename = f"screenshot_phase1_{int(time.time())}.jpg"
        cv2.imwrite(filename, display_frame)
        print(f"Screenshot saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Done!")
