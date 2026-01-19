# YOLO3D: Monocular Depth-Aware Object Detection for Robotics

## Abstract

This document describes the YOLO3D system, a real-time object detection and depth estimation pipeline designed for resource-constrained platforms such as the Raspberry Pi. The system combines YOLO11n for 2D object detection with Depth Anything V2 for monocular depth estimation, implementing robust normalization, distance-based zone classification, and rule-based action policies for robotics applications.

## 1. Introduction

YOLO3D addresses the challenge of providing depth-aware object detection on embedded systems with limited computational resources. The system processes video streams in real-time, providing both relative and optionally calibrated metric distance estimates for detected objects, enabling collision avoidance and navigation decision-making.

## 2. System Architecture

### 2.1 Pipeline Overview

The YOLO3D pipeline consists of the following stages:

1. **Frame Capture**: Input frames are captured from a USB camera at 640×480 resolution
2. **Frame Resizing**: Frames are resized to 640×384 for processing efficiency
3. **2D Object Detection**: YOLO11n performs object detection on every frame
4. **Depth Estimation**: Depth Anything V2 generates depth maps (throttled every N frames)
5. **Depth Normalization**: Raw depth values are normalized using percentile-based method
6. **Per-Object Depth Sampling**: 7×7 patch median sampling extracts depth for each detection
7. **Zone Classification**: Objects are classified into Near/Medium/Far zones
8. **Action Policy**: Rule-based policy generates robotics actions (STOP/SLOW_DOWN/PROCEED)

### 2.2 Performance Optimizations

To achieve real-time performance on Raspberry Pi (CPU-only), the following optimizations are applied:

- **Resolution Reduction**: Processing at 640×384 instead of full resolution reduces computation by ~20%
- **Depth Throttling**: Depth estimation runs every N frames (default: 4), with cached depth maps reused
- **Gradient Disabling**: All inference operations use `torch.no_grad()` to reduce memory and computation
- **Efficient Sampling**: Per-object depth uses small patch sampling (7×7) rather than full bbox processing

## 3. Depth Normalization

### 3.1 Percentile-Based Normalization

Monocular depth models output scale-ambiguous values that vary with scene content. To create stable, relative depth maps, we employ percentile-based normalization instead of min/max normalization.

Given a raw depth map $D_{raw}$ of size $H \times W$, we compute:

$$p_5 = \text{percentile}(D_{raw}, 5)$$
$$p_{95} = \text{percentile}(D_{raw}, 95)$$

The normalized depth map is then:

$$D_{norm} = \frac{D_{raw} - p_5}{p_{95} - p_5}$$

Values are clamped to $[0, 1]$ where:
- $0$ represents the nearest objects in the scene
- $1$ represents the farthest objects in the scene

### 3.2 Exponential Moving Average Smoothing

To reduce flicker between frames, percentile values are smoothed using exponential moving average (EMA):

$$p_5^{(t)} = \alpha \cdot p_5^{(raw)} + (1 - \alpha) \cdot p_5^{(t-1)}$$
$$p_{95}^{(t)} = \alpha \cdot p_{95}^{(raw)} + (1 - \alpha) \cdot p_{95}^{(t-1)}$$

where $\alpha = 0.3$ is the smoothing factor and $t$ denotes the frame index.

**Advantages of Percentile Normalization:**
- Robust to outliers (noisy pixels, depth artifacts)
- Stable across frames (reduces flicker)
- Adapts to varying scene content
- More reliable than min/max for real-world scenes

## 4. Per-Object Depth Estimation

### 4.1 Patch-Based Median Sampling

For each detected object with bounding box coordinates $(x_1, y_1, x_2, y_2)$, we extract depth using a patch-based approach:

1. Compute bounding box center: $(c_x, c_y) = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$
2. Extract a $7 \times 7$ pixel patch centered at $(c_x, c_y)$ from the normalized depth map
3. Compute median depth: $d_{obj} = \text{median}(\text{Patch})$

**Pseudo-code:**
```
patch_size = 7
half_patch = patch_size // 2
x_start = max(0, c_x - half_patch)
x_end = min(W, c_x + half_patch + 1)
y_start = max(0, c_y - half_patch)
y_end = min(H, c_y + half_patch + 1)
patch = D_norm[y_start:y_end, x_start:x_end]
d_obj = median(patch)
```

**Rationale:**
- Single-pixel sampling is noisy and causes flicker
- Median aggregation is robust to outliers
- Center region avoids edge artifacts from bbox boundaries
- 7×7 patch (49 samples) provides good balance between robustness and computation

### 4.2 Distance Zone Classification

Objects are classified into three zones based on normalized depth:

- **Near**: $d_{obj} < 0.33$ (closest third of scene)
- **Medium**: $0.33 \leq d_{obj} < 0.66$ (middle third)
- **Far**: $d_{obj} \geq 0.66$ (farthest third)

These zones are used for robotics decision-making (see Section 6).

## 5. Optional Meter Calibration

### 5.1 Calibration Process

To convert normalized depth to approximate metric distances, a two-point calibration is performed:

1. User stands at known distance $d_1$ (e.g., 0.5 m) from camera
2. System records median normalized depth value: $\delta_1$
3. User stands at known distance $d_2$ (e.g., 2.0 m) from camera
4. System records median normalized depth value: $\delta_2$

### 5.2 Linear Interpolation

Assuming linear depth scale (approximation), the mapping from normalized depth $\delta$ to meters $d$ is:

$$d = d_1 + \frac{\delta - \delta_1}{\delta_2 - \delta_1} \cdot (d_2 - d_1)$$

### 5.3 Limitations

**Important Limitations:**
- Calibration assumes linear depth scale, which may not hold for all scenes
- Accuracy is approximate (typical error ±10-20%)
- Calibration is scene-dependent and may require re-calibration if camera/scene changes
- Monocular depth is fundamentally scale-ambiguous; calibration provides relative scaling only

**When to Use:**
- Suitable for approximate distance estimation
- Useful for relative distance comparisons
- Not suitable for high-precision measurement (use stereo vision or LiDAR instead)

## 6. Action Policy

### 6.1 Rule-Based Policy

The system implements a deterministic rule-based action policy:

```
IF any object in Near zone:
    IF object center_x < frame_center - 50:
        RETURN AVOID_RIGHT
    ELIF object center_x > frame_center + 50:
        RETURN AVOID_LEFT
    ELSE:
        RETURN STOP
ELIF any object in Medium zone:
    RETURN SLOW_DOWN
ELSE:
    RETURN PROCEED
```

### 6.2 Action Debouncing

To prevent action flicker, actions are debounced using a frame history buffer:

1. Store last $N$ actions (default: $N = 3$)
2. Use most common action in buffer as output
3. Only update action if it persists for $N$ consecutive frames

This provides hysteresis and reduces rapid state changes.

## 7. Platform Constraints and Optimizations

### 7.1 Raspberry Pi Constraints

- **CPU-only**: No GPU acceleration available
- **Limited memory**: ~1-4 GB RAM
- **Thermal limits**: Sustained high CPU usage causes throttling
- **ARM architecture**: Different instruction set than x86

### 7.2 Implemented Optimizations

1. **Frame Resizing**: 640×384 reduces processing by ~20% vs 640×480
2. **Depth Throttling**: Depth runs every 4 frames, reducing computation by 75%
3. **Gradient Disabling**: `torch.no_grad()` reduces memory and speeds inference
4. **Caching**: Depth maps cached and reused between frames
5. **Efficient Sampling**: Small patch sampling avoids per-pixel loops

### 7.3 Performance Characteristics

- **YOLO Inference**: ~30-50 ms per frame (640×384)
- **Depth Estimation**: ~200-400 ms per frame (384×384 input, throttled)
- **Overall Pipeline**: ~100-200 ms per frame (with throttling)
- **Achieved FPS**: 5-15 FPS depending on Pi model and scene complexity

## 8. Outputs

The system provides multiple outputs:

1. **Detections**: List of objects with class, confidence, bbox, zone, and distance
2. **Filtered Detections**: Subset based on zone and class filters (for robotics)
3. **Nearest Object**: Closest object with full metadata
4. **Action**: Current robotics action (STOP/SLOW_DOWN/PROCEED/AVOID_LEFT/AVOID_RIGHT)
5. **JSON Endpoint**: `/robot_state` provides structured state for integration

## 9. Conclusion

YOLO3D provides a practical solution for depth-aware object detection on resource-constrained platforms. The combination of robust normalization, efficient sampling, and rule-based action policies enables real-time robotics applications while maintaining acceptable accuracy for relative distance estimation and zone-based decision-making.

## References

- YOLO11n: Ultralytics YOLO (https://github.com/ultralytics/ultralytics)
- Depth Anything V2: Microsoft Research
- Percentile normalization: Robust statistics methods
- Patch sampling: Common practice in depth estimation literature
