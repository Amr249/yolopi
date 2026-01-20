"""
ONNX Runtime-based YOLO detector for Raspberry Pi.
This is an optional alternative to PyTorch YOLO inference.

Usage:
    from onnx.onnx_yolo_detector import OnnxYoloDetector
    
    detector = OnnxYoloDetector("yolo11n.onnx")
    results = detector.detect(frame)
    
Returns:
    List of detections in format compatible with yolo3d_detector.py
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional


class OnnxYoloDetector:
    """
    ONNX Runtime-based YOLO detector.
    Optimized for CPU-only inference on Raspberry Pi.
    """
    
    def __init__(self, onnx_model_path: str, conf_threshold: float = 0.5, 
                 input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize ONNX Runtime YOLO detector.
        
        Args:
            onnx_model_path: Path to ONNX model file (.onnx)
            conf_threshold: Confidence threshold for detections
            input_size: Input image size (width, height) - YOLO11n uses 640x640
        """
        self.onnx_model_path = onnx_model_path
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.input_width, self.input_height = input_size
        
        # Load ONNX model with CPU execution provider
        print(f"📦 Loading ONNX model from {onnx_model_path}...")
        try:
            # Create session with CPU execution provider only (no GPU)
            providers = ['CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                onnx_model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Get input/output names and shapes
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            print(f"✓ ONNX model loaded successfully")
            print(f"   Input shape: {input_shape}")
            print(f"   Input name: {self.input_name}")
            
            # Get output names
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"   Output names: {self.output_names}")
            
            # YOLO class names (standard COCO classes)
            self.labels = self._get_coco_labels()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def _get_coco_labels(self) -> Dict[int, str]:
        """
        Get COCO class labels (YOLO11n uses COCO dataset).
        Returns dict mapping class_id to class_name.
        """
        coco_labels = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        return {i: label for i, label in enumerate(coco_labels)}
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float, int, int, int, int]:
        """
        Preprocess frame for ONNX inference.
        
        Args:
            frame: Input BGR image (numpy array)
            
        Returns:
            preprocessed: Preprocessed image (1, 3, H, W) normalized to [0, 1]
            scale_x: Scale factor for x coordinates
            scale_y: Scale factor for y coordinates
            pad_x: X padding offset
            pad_y: Y padding offset
            orig_w: Original frame width
            orig_h: Original frame height
        """
        orig_h, orig_w = frame.shape[:2]
        
        # Resize to model input size with letterbox padding (maintains aspect ratio)
        # This matches YOLO's preprocessing
        scale = min(self.input_width / orig_w, self.input_height / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)  # Gray padding
        
        # Calculate padding offsets (center)
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        
        # Place resized image in padded image
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose from HWC to CHW and add batch dimension: (1, 3, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        # Calculate scale factors (accounting for padding)
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
        
        return batched, scale_x, scale_y, pad_x, pad_y, orig_w, orig_h
    
    def _postprocess(self, outputs: List[np.ndarray], scale_x: float, scale_y: float,
                    pad_x: int, pad_y: int, orig_w: int, orig_h: int) -> List[Dict]:
        """
        Postprocess ONNX outputs to extract bounding boxes.
        
        Args:
            outputs: ONNX model outputs
            scale_x: Scale factor for x coordinates
            scale_y: Scale factor for y coordinates
            pad_x: X padding offset
            pad_y: Y padding offset
            orig_w: Original frame width
            orig_h: Original frame height
            
        Returns:
            detections: List of detection dicts compatible with yolo3d_detector.py format
        """
        detections = []
        
        # YOLO ONNX output format: Ultralytics exports can vary
        # Format 1: (batch, num_detections, 85) - raw format
        # Format 2: (batch, num_detections, 6) - post-processed (x1,y1,x2,y2,conf,cls)
        # Format 3: Multiple outputs (boxes, scores, classes)
        if len(outputs) == 0:
            return detections
        
        # Get output tensor (usually first output)
        output = outputs[0]  # Shape can vary
        
        # Handle empty output
        if output.size == 0 or output.shape[0] == 0:
            return detections
        
        # Check output format
        if len(output.shape) == 3 and output.shape[2] == 6:
            # Post-processed format: (batch, num_detections, 6) - (x1,y1,x2,y2,conf,cls)
            predictions = output[0]  # Remove batch dimension
            boxes_xyxy = predictions[:, :4]  # x1, y1, x2, y2
            confidences = predictions[:, 4]  # confidence
            class_ids = predictions[:, 5].astype(int)  # class ID
            
            # Filter by confidence
            mask = confidences >= self.conf_threshold
            boxes_xyxy = boxes_xyxy[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            
            # Process each detection
            for box, cls_id, conf in zip(boxes_xyxy, class_ids, confidences):
                # Box is already in x1,y1,x2,y2 format
                # Remove padding and scale to original size
                x1, y1, x2, y2 = box
                
                # Remove padding offset
                x1 -= pad_x
                y1 -= pad_y
                x2 -= pad_x
                y2 -= pad_y
                
                # Scale to original image size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Clamp to original image bounds
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                # Get class label
                label = self.labels.get(int(cls_id), f"class_{int(cls_id)}")
                
                # Create detection dict
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'class': label,
                    'class_id': int(cls_id),
                    'confidence': float(conf)
                }
                detections.append(detection)
            
            return detections
        
        # Raw format: (batch, num_detections, 85) or similar
        # Remove batch dimension if present
        if len(output.shape) == 3:
            predictions = output[0]  # Remove batch dimension: (num_detections, 85)
        else:
            predictions = output  # Already 2D: (num_detections, 85)
        
        # Extract bounding boxes, objectness, and class scores
        boxes = predictions[:, :4]  # (num_detections, 4) - x_center, y_center, width, height
        objectness = predictions[:, 4:5]  # (num_detections, 1)
        class_scores = predictions[:, 5:]  # (num_detections, 80)
        
        # Combine objectness and class scores
        scores = objectness * class_scores  # (num_detections, 80)
        
        # Get class IDs and max confidence
        class_ids = np.argmax(scores, axis=1)  # (num_detections,)
        confidences = np.max(scores, axis=1)  # (num_detections,)
        
        # Filter by confidence threshold
        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        # Convert center+width/height to x1,y1,x2,y2 format
        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            # Box format: (x_center, y_center, width, height) in model input coordinates
            x_center, y_center, width, height = box
            
            # Remove padding offset
            x_center -= pad_x
            y_center -= pad_y
            
            # Convert to x1, y1, x2, y2
            x1 = int((x_center - width / 2) * scale_x)
            y1 = int((y_center - height / 2) * scale_y)
            x2 = int((x_center + width / 2) * scale_x)
            y2 = int((y_center + height / 2) * scale_y)
            
            # Clamp to original image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            # Get class label
            label = self.labels.get(int(cls_id), f"class_{int(cls_id)}")
            
            # Create detection dict (compatible with yolo3d_detector.py format)
            detection = {
                'bbox': (x1, y1, x2, y2),
                'class': label,
                'class_id': int(cls_id),
                'confidence': float(conf)
            }
            detections.append(detection)
        
        return detections
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLO detection on input frame.
        
        Args:
            frame: Input BGR image (numpy array)
            
        Returns:
            detections: List of detection dicts with format:
                {
                    'bbox': (x1, y1, x2, y2),
                    'class': str,
                    'class_id': int,
                    'confidence': float
                }
        """
        # Preprocess
        preprocessed, scale_x, scale_y, pad_x, pad_y, orig_w, orig_h = self._preprocess(frame)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
        
        # Postprocess
        detections = self._postprocess(outputs, scale_x, scale_y, pad_x, pad_y, orig_w, orig_h)
        
        return detections
