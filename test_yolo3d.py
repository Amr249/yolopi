"""
Test script for YOLO-3D visualization
Opens a window showing:
- Main view with 3D bounding boxes
- Depth map visualization (inset)
- Bird's Eye View (inset)

Similar to: https://github.com/niconielsen32/YOLO-3D
"""

import cv2
import numpy as np
import time
from yolo3d_detector import YOLODetector
from bbox3d_utils import (
    create_3d_bbox_from_2d,
    draw_3d_bbox,
    create_bird_eye_view,
    visualize_depth_map,
    draw_depth_overlay
)

# Initialize detector
print("Initializing YOLO-3D detector...")
detector = YOLODetector(
    yolo_model_path="yolo11n.pt",
    depth_model_name="depth-anything/Depth-Anything-V2-Small-hf",
    depth_input_size=(384, 256),  # Updated to match app.py (reduced for lower CPU)
    depth_throttle=6,  # Updated to match app.py (higher = lower CPU)
    conf_threshold=0.5,
    processing_resolution=(480, 288)  # Match app.py for consistency
)
print("✓ Detector ready")

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Get camera matrix
camera_matrix = detector.get_camera_matrix()

# Object dimensions database (for 3D boxes)
object_dimensions_db = {
    'person': {'width': 0.5, 'height': 1.7, 'length': 0.3},
    'car': {'width': 1.8, 'height': 1.5, 'length': 4.5},
    'bicycle': {'width': 0.6, 'height': 1.2, 'length': 1.5},
    'motorcycle': {'width': 0.7, 'height': 1.3, 'length': 2.0},
    'bus': {'width': 2.5, 'height': 3.0, 'length': 12.0},
    'truck': {'width': 2.5, 'height': 3.5, 'length': 8.0},
    'chair': {'width': 0.5, 'height': 1.0, 'length': 0.5},
    'bottle': {'width': 0.1, 'height': 0.3, 'length': 0.1},
    'potted plant': {'width': 0.3, 'height': 0.5, 'length': 0.3},
}

# Colors for different classes (white, green, yellow like reference)
bbox_colors = [
    (255, 255, 255),  # White
    (0, 255, 0),      # Green
    (0, 255, 255),    # Yellow (cyan in BGR)
    (255, 255, 0),    # Cyan
    (0, 165, 255),    # Orange
    (255, 0, 255),    # Magenta
    (128, 0, 128),    # Purple
    (255, 192, 203),  # Pink
    (0, 128, 255),    # Blue
    (255, 165, 0),    # Blue (BGR)
]

# FPS tracking
fps_buffer = []
fps_avg_len = 30

print("\nStarting YOLO-3D visualization...")
print("Press 'q' to quit, 's' to save screenshot")

frame_count = 0

while True:
    start_time = time.time()
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    # Run detection (FPS optimized - frame resizing and depth throttling handled internally)
    detections, depth_map, display_frame = detector.detect(frame, update_depth=None)
    
    # Use the display frame (already resized and optimized)
    vis_frame = display_frame.copy()
    
    # Process each detection for 3D visualization
    detections_3d = []
    
    for det in detections:
        bbox = det['bbox']
        label = det['class']
        conf = det['confidence']
        cls_id = det['class_id']
        
        # PHASE 2: Get distance for 3D box (use distance_m if available, else normalized_depth)
        depth_for_3d = det.get('distance_m', None)
        if depth_for_3d is None:
            # Use normalized_depth converted to approximate meters for 3D visualization
            normalized_depth = det.get('normalized_depth', None)
            if normalized_depth is not None:
                # Rough conversion: assume normalized [0,1] maps to [0.5, 5.0] meters
                depth_for_3d = 0.5 + normalized_depth * 4.5
            else:
                depth_for_3d = 2.0  # Default fallback
        
        # Get object dimensions
        obj_dims = object_dimensions_db.get(label, {'width': 1.0, 'height': 1.0, 'length': 1.0})
        
        # Create 3D bounding box
        try:
            corners_3d, center_3d = create_3d_bbox_from_2d(
                bbox, depth_for_3d, camera_matrix, obj_dims
            )
            
            # Draw depth overlay first (so 3D box appears on top)
            if depth_map is not None:
                vis_frame = draw_depth_overlay(vis_frame, bbox, depth_map, alpha=0.25)
            
            # Draw 3D bounding box (wireframe cube) - prominent white/green/yellow
            color = bbox_colors[cls_id % len(bbox_colors)]
            # Use thicker lines for better visibility
            vis_frame = draw_3d_bbox(vis_frame, corners_3d, camera_matrix, color, 3)
            
            # PHASE 2: Add label with zone and distance info
            x1, y1, x2, y2 = bbox
            zone = det.get('zone', 'Unknown')
            distance_m = det.get('distance_m', None)
            normalized_depth = det.get('normalized_depth', None)
            
            # Format distance display
            if distance_m is not None:
                dist_text = f"{distance_m:.2f}m"
            elif normalized_depth is not None:
                dist_text = f"Rel:{normalized_depth:.2f}"
            else:
                dist_text = "N/A"
            
            label_text = f"S:{conf:.2f} D:{dist_text} Zone:{zone} {label} ID:{cls_id}"
            
            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 5, text_h + 5)
            cv2.rectangle(vis_frame,
                         (x1, label_y - text_h - 5),
                         (x1 + text_w + 5, label_y + 2),
                         color, -1)
            cv2.putText(vis_frame, label_text, (x1 + 2, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Store for BEV
            det['center_3d'] = center_3d
            detections_3d.append(det)
            
        except Exception as e:
            print(f"Error creating 3D box for {label}: {e}")
            continue
    
    # Create depth map visualization (inset - top left)
    if depth_map is not None:
        depth_vis = visualize_depth_map(depth_map)
        if depth_vis is not None:
            # Resize for inset
            inset_size = (200, 150)
            depth_inset = cv2.resize(depth_vis, inset_size)
            
            # Place in top-left corner
            vis_frame[10:10+inset_size[1], 10:10+inset_size[0]] = depth_inset
            cv2.putText(vis_frame, "Depth Map", (10, 10+inset_size[1]+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Create Bird's Eye View (inset - bottom left)
    if detections_3d:
        bev_image = create_bird_eye_view(detections_3d, camera_matrix, 
                                        bev_size=(250, 250), view_range=10.0)
        
        # Place in bottom-left corner
        bev_size = (250, 250)
        h, w = vis_frame.shape[:2]
        bev_y = h - bev_size[1] - 10
        bev_x = 10
        
        vis_frame[bev_y:bev_y+bev_size[1], bev_x:bev_x+bev_size[0]] = bev_image
        cv2.putText(vis_frame, "Bird's Eye View", (bev_x, bev_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Calculate and display FPS
    dt = time.time() - start_time
    if dt > 0:
        fps = 1.0 / dt
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
    else:
        avg_fps = 0
    
    # Draw FPS and object count
    cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, h - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis_frame, f"Objects: {len(detections)}", (10, h - 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display frame
    cv2.imshow("YOLO-3D Detection", vis_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('s'):
        filename = f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, vis_frame)
        print(f"Screenshot saved: {filename}")
    
    frame_count += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Done!")
