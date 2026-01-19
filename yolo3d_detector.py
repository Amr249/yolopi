"""
YOLO-D (YOLO + Depth) Detector for Raspberry Pi
Optimized for CPU-only inference with depth-aware object detection.

Architecture:
- YOLO: 2D object detection
- Depth Anything V2: Monocular depth estimation
- Fusion: Extract depth at bounding box centers
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import time


class YOLODetector:
    """
    Optimized YOLO-D detector for Raspberry Pi.
    Combines YOLO 2D detection with monocular depth estimation.
    """
    
    def __init__(self, yolo_model_path="yolo11n.pt", 
                 depth_model_name="depth-anything/Depth-Anything-V2-Small-hf",
                 depth_input_size=(384, 384),
                 depth_throttle=3,
                 conf_threshold=0.5):
        """
        Initialize YOLO-D detector.
        
        Args:
            yolo_model_path: Path to YOLO model (.pt file)
            depth_model_name: HuggingFace model name for depth estimation
            depth_input_size: Input size for depth model (smaller = faster)
            depth_throttle: Process depth every N frames (1 = every frame)
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.depth_input_size = depth_input_size
        self.depth_throttle = depth_throttle
        self.frame_count = 0
        
        # Load YOLO model for 2D detection
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        self.labels = self.yolo_model.names
        print(f"✓ YOLO loaded: {len(self.labels)} classes")
        
        # Load depth estimation model
        print("Loading depth estimation model...")
        try:
            self.depth_processor = AutoImageProcessor.from_pretrained(depth_model_name)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name)
            self.depth_model.eval()  # Set to evaluation mode
            
            # Ensure CPU-only (no CUDA)
            if torch.cuda.is_available():
                print("⚠ CUDA available but using CPU for Pi compatibility")
            self.depth_model = self.depth_model.cpu()
            
            # Disable gradient computation for inference
            for param in self.depth_model.parameters():
                param.requires_grad = False
            
            self.depth_available = True
            print("✓ Depth model loaded successfully")
        except Exception as e:
            print(f"⚠ Depth model failed to load: {e}")
            print("⚠ Falling back to geometry-based depth estimation")
            self.depth_available = False
            self.depth_model = None
            self.depth_processor = None
        
        # Camera intrinsics (default values - should be calibrated)
        # For 640x480 camera with ~60 degree FOV
        self.fx = 640.0  # Focal length in pixels
        self.fy = 640.0
        self.cx = 320.0  # Principal point
        self.cy = 240.0
        
        # Known object dimensions for geometry-based depth (in meters)
        # Used as fallback when depth model is unavailable
        self.object_dimensions = {
            'person': {'height': 1.7, 'width': 0.5},
            'car': {'height': 1.5, 'width': 1.8},
            'bicycle': {'height': 1.2, 'width': 0.6},
            'motorcycle': {'height': 1.3, 'width': 0.7},
            'bus': {'height': 3.0, 'width': 2.5},
            'truck': {'height': 3.5, 'width': 2.5},
        }
        
        # Cache for depth map (reused when throttling)
        self.cached_depth_map = None
        self.cached_depth_frame = None
    
    def estimate_depth_map(self, frame):
        """
        Estimate depth map from monocular image.
        Optimized for Raspberry Pi CPU.
        
        Args:
            frame: Input BGR image (numpy array)
            
        Returns:
            depth_map: Depth map (numpy array, same size as input)
        """
        if not self.depth_available:
            return None
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for faster processing (depth model input size)
            h_orig, w_orig = frame.shape[:2]
            frame_resized = cv2.resize(frame_rgb, self.depth_input_size)
            
            # Preprocess for depth model
            inputs = self.depth_processor(images=frame_resized, return_tensors="pt")
            
            # Inference with no gradient computation (CPU optimized)
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                depth_pred = outputs.predicted_depth
            
            # Post-process depth map
            depth_pred = depth_pred.squeeze().cpu().numpy()
            
            # Resize back to original frame size
            depth_map = cv2.resize(depth_pred, (w_orig, h_orig), 
                                 interpolation=cv2.INTER_LINEAR)
            
            # Normalize depth (Depth Anything outputs relative depth)
            # Scale to approximate meters (adjust based on your scene)
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                # Scale to 0.5-10 meters range (adjustable)
                depth_map = 0.5 + depth_map * 9.5
            
            return depth_map
            
        except Exception as e:
            print(f"⚠ Depth estimation error: {e}")
            return None
    
    def get_depth_at_bbox_center(self, depth_map, bbox):
        """
        Extract depth value at the center of a bounding box.
        Uses median of a small region for robustness.
        
        Args:
            depth_map: Depth map (numpy array)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            depth: Depth value in meters (or None if unavailable)
        """
        if depth_map is None:
            return None
        
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Sample a small region around center (3x3 pixels)
        h, w = depth_map.shape
        x_start = max(0, center_x - 1)
        x_end = min(w, center_x + 2)
        y_start = max(0, center_y - 1)
        y_end = min(h, center_y + 2)
        
        region = depth_map[y_start:y_end, x_start:x_end]
        depth = np.median(region)
        
        return float(depth)
    
    def estimate_depth_geometry(self, bbox, class_name):
        """
        Fallback: Estimate depth using geometry and known object sizes.
        Used when depth model is unavailable.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            class_name: Object class name
            
        Returns:
            depth: Estimated depth in meters
        """
        x1, y1, x2, y2 = bbox
        bbox_height_px = y2 - y1
        
        # Get known object height
        if class_name in self.object_dimensions:
            real_height = self.object_dimensions[class_name]['height']
        else:
            # Default assumption for unknown objects
            real_height = 1.0
        
        # Depth = (real_height * focal_length) / pixel_height
        if bbox_height_px > 0:
            depth = (real_height * self.fy) / bbox_height_px
            # Clamp to reasonable range
            depth = np.clip(depth, 0.5, 20.0)
        else:
            depth = 5.0  # Default fallback
        
        return depth
    
    def detect(self, frame, update_depth=True):
        """
        Run YOLO-D detection pipeline.
        
        Args:
            frame: Input BGR frame
            update_depth: Whether to update depth map (throttled internally)
            
        Returns:
            detections: List of detections with depth info
            depth_map: Current depth map (may be cached)
        """
        # Step 1: YOLO 2D detection
        results = self.yolo_model(frame, verbose=False)
        detections_2d = results[0].boxes
        
        # Step 2: Depth estimation (throttled for performance)
        depth_map = None
        should_update_depth = (self.frame_count % self.depth_throttle == 0) or update_depth
        
        if should_update_depth and self.depth_available:
            depth_map = self.estimate_depth_map(frame)
            self.cached_depth_map = depth_map
            self.cached_depth_frame = frame.copy()
        elif self.cached_depth_map is not None:
            # Reuse cached depth map
            depth_map = self.cached_depth_map
        
        # Step 3: Fuse 2D detections with depth
        detections = []
        
        for det in detections_2d:
            conf = det.conf.item()
            if conf < self.conf_threshold:
                continue
            
            # Extract 2D bounding box
            x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy().squeeze())
            cls_id = int(det.cls.item())
            label = self.labels[cls_id]
            
            # Get depth at bounding box center
            if depth_map is not None:
                depth = self.get_depth_at_bbox_center(depth_map, (x1, y1, x2, y2))
            else:
                # Fallback to geometry-based estimation
                depth = self.estimate_depth_geometry((x1, y1, x2, y2), label)
            
            # Create detection with depth
            detection = {
                'bbox': (x1, y1, x2, y2),
                'class': label,
                'confidence': conf,
                'depth': depth,
                'class_id': cls_id
            }
            
            detections.append(detection)
        
        self.frame_count += 1
        return detections, depth_map


def draw_detection_with_depth(frame, detection, color):
    """
    Draw detection with depth information overlay.
    
    Args:
        frame: Image frame to draw on
        detection: Detection dictionary with bbox, class, confidence, depth
        color: BGR color tuple for bounding box
    """
    x1, y1, x2, y2 = detection['bbox']
    label = detection['class']
    conf = detection['confidence']
    depth = detection['depth']
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare text labels
    label_text = f"{label} {int(conf*100)}%"
    depth_text = f"{depth:.1f}m"
    
    # Calculate text size for background
    (label_w, label_h), _ = cv2.getTextSize(label_text, 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.5, 2)
    (depth_w, depth_h), _ = cv2.getTextSize(depth_text,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.4, 1)
    
    # Draw label background
    label_y = max(y1 - 10, label_h + 5)
    cv2.rectangle(frame, 
                 (x1, label_y - label_h - 5),
                 (x1 + label_w + 5, label_y + 2),
                 color, -1)
    
    # Draw label text
    cv2.putText(frame, label_text, (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw depth text below label
    depth_y = label_y + depth_h + 8
    if depth_y < y2:
        cv2.rectangle(frame,
                     (x1, depth_y - depth_h - 3),
                     (x1 + depth_w + 5, depth_y + 2),
                     (0, 0, 0), -1)
        cv2.putText(frame, depth_text, (x1 + 2, depth_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw depth indicator (line from center to bottom)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    cv2.circle(frame, (center_x, center_y), 3, color, -1)
    
    # Optional: Draw depth visualization bar
    if y2 < frame.shape[0] - 15:
        bar_width = int((depth / 10.0) * 50)  # Scale to 0-50 pixels
        bar_width = min(bar_width, 50)
        cv2.rectangle(frame,
                     (x1, y2 + 5),
                     (x1 + bar_width, y2 + 10),
                     color, -1)
