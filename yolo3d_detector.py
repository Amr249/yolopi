"""
YOLO3D (YOLO + Depth) Detector for Raspberry Pi
Optimized for CPU-only inference with depth-aware object detection.

Architecture:
- YOLO: 2D object detection
- Depth Anything V2: Monocular depth estimation
- Fusion: Extract depth at bounding box centers

PHASE 2: RESEARCH NOTES - Depth Normalization and Distance Estimation
======================================================================

MONOCULAR DEPTH LIMITATIONS:
----------------------------
Monocular depth estimation models (e.g., Depth Anything V2) output scale-ambiguous
depth values. The raw output does not directly correspond to metric distances.
Key limitations:
1. Scale ambiguity: Output is relative, not absolute (no inherent meter scale)
2. Scene-dependent: Depth values vary based on scene content and lighting
3. Model-dependent: Different models use different output scales/ranges
4. No calibration: Without reference points, absolute distances are unknown

ROBUST NORMALIZATION (Percentile Method):
-----------------------------------------
We use percentile-based normalization (p5..p95) instead of min/max:
- Reduces sensitivity to outliers (noisy pixels, depth artifacts)
- More stable across frames (reduces flicker)
- Handles varying scene content better
- Uses exponential moving average to smooth percentile transitions

Output: Normalized depth [0, 1] where:
- 0 = nearest objects in scene
- 1 = farthest objects in scene
- Values are relative to current frame's depth distribution

PATCH SAMPLING FOR OBJECT DEPTH:
--------------------------------
Single-pixel depth sampling is noisy and causes flicker. We use:
- 7x7 pixel patch around bbox center (49 samples)
- Median aggregation (robust to outliers)
- Center region avoids edge artifacts from bbox boundaries
- Lightweight computation (no per-pixel loops)

DISTANCE ZONES (Robotics Applications):
--------------------------------------
Three zones derived from normalized depth:
- Near: normalized_depth < 0.33 (closest third of scene)
- Medium: 0.33 <= normalized_depth < 0.66 (middle third)
- Far: normalized_depth >= 0.66 (farthest third)

These zones are useful for:
- Collision avoidance (Near zone alerts)
- Navigation planning (Medium zone for path planning)
- Long-range detection (Far zone for awareness)

OPTIONAL METER CONVERSION (Calibration Required):
-------------------------------------------------
Meter conversion requires calibration with known reference distances:
1. User stands at two known distances (e.g., 0.5m and 2.0m)
2. System records median depth values at those positions
3. Linear interpolation maps normalized depth to meters

IMPORTANT LIMITATIONS:
- Calibration is approximate (assumes linear depth scale)
- Accuracy depends on calibration quality and scene similarity
- Meter values should include error bounds (±10-20% typical)
- Re-calibration needed if camera/scene changes significantly

RESEARCH-GRADE DISCLAIMERS:
---------------------------
- Monocular depth is fundamentally scale-ambiguous
- Percentile normalization is robust but still relative
- Meter conversion is approximate and requires calibration
- Results are suitable for relative distance estimation and zone classification
- For absolute distance measurement, use stereo vision or LiDAR
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
                 depth_throttle=4,
                 conf_threshold=0.5,
                 processing_resolution=(640, 384)):
        """
        Initialize YOLO-D detector with FPS optimizations for Raspberry Pi.
        
        Args:
            yolo_model_path: Path to YOLO model (.pt file)
            depth_model_name: HuggingFace model name for depth estimation
            depth_input_size: Input size for depth model (smaller = faster)
            depth_throttle: Process depth every N frames (higher = faster, default 4)
            conf_threshold: Confidence threshold for detections
            processing_resolution: Target resolution for processing (width, height)
                                  Lower resolution = faster inference (default 640x384)
        """
        self.conf_threshold = conf_threshold
        self.depth_input_size = depth_input_size
        self.depth_throttle = depth_throttle
        self.processing_resolution = processing_resolution  # (width, height)
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
        
        # PHASE 2: Depth normalization and calibration
        # Normalization cache for stable percentile values
        self.norm_p5 = None  # 5th percentile (near)
        self.norm_p95 = None  # 95th percentile (far)
        
        # Optional meter calibration (disabled by default)
        self.enable_meters = False
        self.calibration_near_m = 0.5  # Reference distance in meters
        self.calibration_far_m = 2.0    # Reference distance in meters
        self.calibration_depth_near = None  # Depth value at near reference
        self.calibration_depth_far = None   # Depth value at far reference
        
        # Zone thresholds (normalized depth)
        self.zone_near_threshold = 0.33
        self.zone_far_threshold = 0.66
    
    def estimate_depth_map(self, frame):
        """
        Estimate depth map from monocular image.
        Optimized for Raspberry Pi CPU with torch.no_grad().
        
        Args:
            frame: Input BGR image (numpy array) - should already be resized
            
        Returns:
            depth_map: Depth map (numpy array, same size as input)
        """
        if not self.depth_available:
            return None
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for faster processing (depth model input size)
            # Frame is already resized to processing_resolution, so resize to depth_input_size
            h_orig, w_orig = frame.shape[:2]
            frame_resized = cv2.resize(frame_rgb, self.depth_input_size)
            
            # Preprocess for depth model
            inputs = self.depth_processor(images=frame_resized, return_tensors="pt")
            
            # CRITICAL: Use torch.no_grad() for CPU inference (no gradient computation)
            # This significantly reduces memory usage and speeds up inference
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                depth_pred = outputs.predicted_depth
            
            # Post-process depth map
            depth_pred = depth_pred.squeeze().cpu().numpy()
            
            # Resize back to processing resolution (not original, for consistency)
            depth_map = cv2.resize(depth_pred, (w_orig, h_orig), 
                                 interpolation=cv2.INTER_LINEAR)
            
            # PHASE 2: Robust depth normalization using percentile method
            # This reduces flicker and handles outliers better than min/max
            depth_map = self.normalize_depth_robust(depth_map)
            
            return depth_map
            
        except Exception as e:
            print(f"⚠ Depth estimation error: {e}")
            return None
    
    def normalize_depth_robust(self, depth_map):
        """
        PHASE 2: Robust depth normalization using percentile method.
        
        RESEARCH NOTES:
        - Monocular depth models output scale-ambiguous values
        - Min/max normalization is sensitive to outliers and causes flicker
        - Percentile normalization (p5..p95) is more robust and stable
        - Output: normalized depth in [0, 1] where 0=near, 1=far
        
        Args:
            depth_map: Raw depth map from model
            
        Returns:
            normalized_depth: Normalized depth map [0, 1]
        """
        if depth_map is None or depth_map.size == 0:
            return None
        
        # Compute percentiles (robust to outliers)
        p5 = np.percentile(depth_map, 5)
        p95 = np.percentile(depth_map, 95)
        
        # Cache percentiles for stability (smooth transitions)
        if self.norm_p5 is None:
            self.norm_p5 = p5
            self.norm_p95 = p95
        else:
            # Exponential moving average to reduce sudden jumps
            alpha = 0.3  # Smoothing factor
            self.norm_p5 = alpha * p5 + (1 - alpha) * self.norm_p5
            self.norm_p95 = alpha * p95 + (1 - alpha) * self.norm_p95
        
        # Normalize using cached percentiles
        if self.norm_p95 > self.norm_p5:
            normalized = (depth_map - self.norm_p5) / (self.norm_p95 - self.norm_p5)
            # Clamp to [0, 1]
            normalized = np.clip(normalized, 0.0, 1.0)
        else:
            # Fallback if percentiles are too close
            normalized = np.zeros_like(depth_map)
        
        return normalized
    
    def get_depth_at_bbox_center(self, depth_map, bbox):
        """
        PHASE 2: Extract normalized depth using robust patch sampling.
        
        RESEARCH NOTES:
        - Single pixel sampling is noisy and flickers
        - Median of a patch (7x7 or 9x9) reduces artifacts
        - Center region sampling avoids edge artifacts from bbox
        - Returns normalized depth [0, 1] where 0=near, 1=far
        
        Args:
            depth_map: Normalized depth map [0, 1]
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            normalized_depth: Normalized depth value [0, 1] or None
        """
        if depth_map is None:
            return None
        
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # PHASE 2: Use larger patch (7x7) for better robustness
        patch_size = 7
        half_patch = patch_size // 2
        
        h, w = depth_map.shape
        x_start = max(0, center_x - half_patch)
        x_end = min(w, center_x + half_patch + 1)
        y_start = max(0, center_y - half_patch)
        y_end = min(h, center_y + half_patch + 1)
        
        # Extract patch and compute median (robust to outliers)
        region = depth_map[y_start:y_end, x_start:x_end]
        if region.size == 0:
            return None
        
        normalized_depth = float(np.median(region))
        return normalized_depth
    
    def classify_distance_zone(self, normalized_depth):
        """
        PHASE 2: Classify object into Near/Medium/Far zone.
        
        Args:
            normalized_depth: Normalized depth [0, 1] where 0=near, 1=far
            
        Returns:
            zone: "Near", "Medium", or "Far"
        """
        if normalized_depth is None:
            return "Unknown"
        
        if normalized_depth < self.zone_near_threshold:
            return "Near"
        elif normalized_depth < self.zone_far_threshold:
            return "Medium"
        else:
            return "Far"
    
    def convert_to_meters(self, normalized_depth):
        """
        PHASE 2: Optional conversion from normalized depth to approximate meters.
        
        RESEARCH NOTES:
        - Monocular depth is scale-ambiguous; requires calibration
        - Linear mapping assumes constant depth scale (approximation)
        - Calibration requires user to stand at known distances
        - Meter values are approximate and should include error bounds
        - If not calibrated, returns None
        
        Args:
            normalized_depth: Normalized depth [0, 1]
            
        Returns:
            distance_m: Approximate distance in meters, or None if not calibrated
        """
        if not self.enable_meters:
            return None
        
        if (self.calibration_depth_near is None or 
            self.calibration_depth_far is None):
            return None
        
        # Linear interpolation between calibration points
        if self.calibration_depth_far != self.calibration_depth_near:
            # Map normalized_depth to meters using linear interpolation
            depth_range = self.calibration_depth_far - self.calibration_depth_near
            meter_range = self.calibration_far_m - self.calibration_near_m
            
            # Inverse mapping: normalized_depth -> meters
            # normalized_depth = 0 -> near_m, normalized_depth = 1 -> far_m
            distance_m = (self.calibration_near_m + 
                         (normalized_depth * meter_range))
        else:
            # Fallback if calibration points are identical
            distance_m = self.calibration_near_m
        
        return max(0.1, distance_m)  # Clamp to minimum 0.1m
    
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
    
    def detect(self, frame, update_depth=None):
        """
        Run YOLO-D detection pipeline with FPS optimizations.
        
        OPTIMIZATION: Frame is resized once to processing_resolution for faster inference.
        OPTIMIZATION: Depth estimation is throttled (runs every N frames).
        OPTIMIZATION: Uses torch.no_grad() for all inference operations.
        
        Args:
            frame: Input BGR frame (will be resized internally)
            update_depth: If None, uses throttling. If True/False, overrides throttling.
            
        Returns:
            detections: List of detections with depth info
            depth_map: Current depth map (may be cached from previous frame)
            original_frame: Original frame before resizing (for display)
        """
        # OPTIMIZATION 1: Resize frame ONCE to processing resolution
        # This reduces computation for both YOLO and depth estimation
        original_frame = frame.copy()
        h_orig, w_orig = frame.shape[:2]
        target_w, target_h = self.processing_resolution
        
        # Only resize if different from target
        if (w_orig, h_orig) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Step 1: YOLO 2D detection (runs every frame for smooth tracking)
        # YOLO is fast enough to run every frame on Pi
        # NOTE: Ultralytics YOLO automatically uses torch.no_grad() during inference
        results = self.yolo_model(frame, verbose=False)
        detections_2d = results[0].boxes
        
        # Step 2: Depth estimation (THROTTLED for performance)
        # OPTIMIZATION 2: Depth runs every N frames, cached depth map reused
        depth_map = None
        if update_depth is None:
            # Use throttling: run depth every depth_throttle frames
            should_update_depth = (self.frame_count % self.depth_throttle == 0)
        else:
            # Override throttling if explicitly requested
            should_update_depth = update_depth
        
        if should_update_depth and self.depth_available:
            # Run depth estimation (expensive operation)
            depth_map = self.estimate_depth_map(frame)
            self.cached_depth_map = depth_map
            self.cached_depth_frame = frame.copy()
        elif self.cached_depth_map is not None:
            # OPTIMIZATION: Reuse cached depth map (no computation)
            depth_map = self.cached_depth_map
        
        # Step 3: Fuse 2D detections with depth
        detections = []
        
        for det in detections_2d:
            conf = det.conf.item()
            if conf < self.conf_threshold:
                continue
            
            # Extract 2D bounding box (in processing resolution coordinates)
            x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy().squeeze())
            cls_id = int(det.cls.item())
            label = self.labels[cls_id]
            
            # PHASE 2: Get normalized depth using robust patch sampling
            normalized_depth = None
            distance_m = None
            zone = "Unknown"
            
            if depth_map is not None:
                # Get normalized depth [0, 1] where 0=near, 1=far
                normalized_depth = self.get_depth_at_bbox_center(depth_map, (x1, y1, x2, y2))
                
                if normalized_depth is not None:
                    # Classify into zone
                    zone = self.classify_distance_zone(normalized_depth)
                    
                    # Optional: Convert to meters if calibrated
                    distance_m = self.convert_to_meters(normalized_depth)
            else:
                # Fallback to geometry-based estimation (returns approximate meters)
                distance_m = self.estimate_depth_geometry((x1, y1, x2, y2), label)
                if distance_m is not None:
                    # Convert to normalized depth for consistency (rough approximation)
                    # Assume geometry-based depth is in reasonable range
                    normalized_depth = min(1.0, max(0.0, (distance_m - 0.5) / 5.0))
                    zone = self.classify_distance_zone(normalized_depth)
            
            # Scale bbox coordinates back to original frame size if needed
            if (w_orig, h_orig) != (target_w, target_h):
                scale_x = w_orig / target_w
                scale_y = h_orig / target_h
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
            
            # Get object dimensions for 3D box
            object_dims = self.object_dimensions.get(label, None)
            
            # PHASE 2: Create detection with normalized depth, zone, and optional meters
            detection = {
                'bbox': (x1, y1, x2, y2),  # In original frame coordinates
                'class': label,
                'confidence': conf,
                'depth': distance_m,  # Keep for backward compatibility (meters or None)
                'normalized_depth': normalized_depth,  # PHASE 2: Normalized [0, 1]
                'zone': zone,  # PHASE 2: "Near", "Medium", or "Far"
                'distance_m': distance_m,  # PHASE 2: Approximate meters (if calibrated)
                'class_id': cls_id,
                'object_dimensions': object_dims
            }
            
            detections.append(detection)
        
        # Scale depth map back to original size if needed (for visualization)
        if depth_map is not None and (w_orig, h_orig) != (target_w, target_h):
            depth_map = cv2.resize(depth_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        
        self.frame_count += 1
        return detections, depth_map, original_frame
    
    def get_camera_matrix(self):
        """Get camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)


def draw_detection_with_depth(frame, detection, color):
    """
    PHASE 2: Draw detection with depth information, zone, and distance overlay.
    
    Args:
        frame: Image frame to draw on
        detection: Detection dictionary with bbox, class, confidence, zone, distance
        color: BGR color tuple for bounding box
    """
    x1, y1, x2, y2 = detection['bbox']
    label = detection['class']
    conf = detection['confidence']
    
    # PHASE 2: Get zone and distance information
    zone = detection.get('zone', 'Unknown')
    distance_m = detection.get('distance_m', None)
    normalized_depth = detection.get('normalized_depth', None)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # PHASE 2: Prepare text labels with zone and distance
    label_text = f"{label} {int(conf*100)}%"
    
    # Format distance display
    if distance_m is not None:
        # Show meters if calibrated
        distance_text = f"{distance_m:.2f}m"
    elif normalized_depth is not None:
        # Show relative depth if not calibrated
        distance_text = f"Rel: {normalized_depth:.2f}"
    else:
        distance_text = "N/A"
    
    # Zone text with color coding
    zone_colors = {
        'Near': (0, 255, 0),      # Green
        'Medium': (0, 255, 255),  # Yellow
        'Far': (0, 165, 255),     # Orange
        'Unknown': (128, 128, 128) # Gray
    }
    zone_color = zone_colors.get(zone, (128, 128, 128))
    zone_text = f"Zone: {zone}"
    
    # Calculate text sizes
    (label_w, label_h), _ = cv2.getTextSize(label_text, 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.5, 2)
    (distance_w, distance_h), _ = cv2.getTextSize(distance_text,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.4, 1)
    (zone_w, zone_h), _ = cv2.getTextSize(zone_text,
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.4, 1)
    
    # Draw label background
    label_y = max(y1 - 10, label_h + 5)
    cv2.rectangle(frame, 
                 (x1, label_y - label_h - 5),
                 (x1 + label_w + 5, label_y + 2),
                 color, -1)
    cv2.putText(frame, label_text, (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw distance text
    distance_y = label_y + distance_h + 8
    if distance_y < y2:
        cv2.rectangle(frame,
                     (x1, distance_y - distance_h - 3),
                     (x1 + distance_w + 5, distance_y + 2),
                     (0, 0, 0), -1)
        cv2.putText(frame, distance_text, (x1 + 2, distance_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw zone text with color coding
    zone_y = distance_y + zone_h + 8
    if zone_y < y2:
        cv2.rectangle(frame,
                     (x1, zone_y - zone_h - 3),
                     (x1 + zone_w + 5, zone_y + 2),
                     (0, 0, 0), -1)
        cv2.putText(frame, zone_text, (x1 + 2, zone_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, zone_color, 1)
    
    # Draw depth indicator (line from center to bottom)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    cv2.circle(frame, (center_x, center_y), 3, color, -1)
    
    # Optional: Draw depth visualization bar (using normalized depth)
    if normalized_depth is not None and y2 < frame.shape[0] - 15:
        bar_width = int(normalized_depth * 50)  # Scale normalized [0,1] to 0-50 pixels
        bar_width = min(bar_width, 50)
        cv2.rectangle(frame,
                     (x1, y2 + 5),
                     (x1 + bar_width, y2 + 10),
                     zone_color, -1)
