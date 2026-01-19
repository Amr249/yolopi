# PHASE 2: Research Notes - Depth Normalization and Distance Estimation

## Overview

This document explains the depth normalization and distance estimation methods used in YOLO3D, including research-grade disclaimers and limitations.

## Monocular Depth Estimation Limitations

### Scale Ambiguity

Monocular depth models (e.g., Depth Anything V2) output **scale-ambiguous** depth values. This means:

- Raw output does **not** directly correspond to metric distances (meters)
- Depth values are **relative** to the scene content
- The same object at different distances may produce similar depth values
- **No inherent meter scale** without calibration

### Why This Matters

For robotics applications:
- ✅ **Relative distances work**: "Object A is closer than Object B"
- ✅ **Zone classification works**: "Object is in Near/Medium/Far zone"
- ❌ **Absolute distances unreliable**: "Object is exactly 1.5 meters away" (requires calibration)

## Robust Depth Normalization

### Percentile Method (p5..p95)

Instead of min/max normalization, we use **percentile-based normalization**:

```python
p5 = np.percentile(depth_map, 5)   # 5th percentile (near)
p95 = np.percentile(depth_map, 95) # 95th percentile (far)
normalized = (depth - p5) / (p95 - p5)
```

**Advantages:**
- **Robust to outliers**: Ignores noisy pixels and depth artifacts
- **Stable across frames**: Reduces flicker compared to min/max
- **Handles varying scenes**: Adapts to different scene content
- **Smooth transitions**: Uses exponential moving average for percentile values

**Output:**
- Normalized depth in range [0, 1]
- 0 = nearest objects in current frame
- 1 = farthest objects in current frame
- Values are **relative** to frame's depth distribution

### Why Not Min/Max?

Min/max normalization is sensitive to:
- Single noisy pixel (outlier) can shift entire range
- Sudden changes cause flicker between frames
- Doesn't handle varying scene content well

## Patch Sampling for Object Depth

### Method

For each detected object, we extract depth using:
- **7x7 pixel patch** around bounding box center (49 samples)
- **Median aggregation** (robust to outliers)
- **Center region** (avoids edge artifacts)

### Why Not Single Pixel?

Single-pixel sampling:
- ❌ Noisy (depth artifacts, quantization)
- ❌ Flickers between frames
- ❌ Sensitive to bbox alignment

Patch sampling:
- ✅ Robust to noise
- ✅ Stable across frames
- ✅ Lightweight computation (no per-pixel loops)

## Distance Zones (Robotics Applications)

### Three-Zone Classification

Based on normalized depth [0, 1]:
- **Near**: < 0.33 (closest third of scene)
- **Medium**: 0.33 - 0.66 (middle third)
- **Far**: ≥ 0.66 (farthest third)

### Use Cases

- **Collision avoidance**: Alert on Near zone objects
- **Navigation planning**: Use Medium zone for path planning
- **Long-range awareness**: Monitor Far zone for approaching objects

## Optional Meter Conversion

### Calibration Process

To convert normalized depth to approximate meters:

1. **User stands at known distance 1** (e.g., 0.5m from camera)
2. System records median depth value → `calibration_depth_near`
3. **User stands at known distance 2** (e.g., 2.0m from camera)
4. System records median depth value → `calibration_depth_far`
5. Linear interpolation maps normalized depth → meters

### Limitations

- **Approximate**: Assumes linear depth scale (may not hold for all scenes)
- **Scene-dependent**: Calibration valid for similar scenes/lighting
- **Error bounds**: Typical accuracy ±10-20% (depends on calibration quality)
- **Re-calibration needed**: If camera/scene changes significantly

### When to Use

- ✅ **Relative distances sufficient**: Use normalized depth + zones
- ✅ **Approximate distances needed**: Use meter conversion with calibration
- ❌ **Precise distances required**: Use stereo vision or LiDAR instead

## Research-Grade Disclaimers

### What This System Provides

1. **Relative distance estimation**: "Object A is closer than Object B" ✅
2. **Zone classification**: "Object is in Near/Medium/Far zone" ✅
3. **Approximate meters** (if calibrated): "Object is ~1.5m away" ⚠️

### What This System Does NOT Provide

1. **Absolute metric distances** (without calibration) ❌
2. **High-precision measurements** (typical error ±10-20%) ❌
3. **Real-time calibration** (requires manual setup) ❌

### Recommended Use Cases

- ✅ Collision avoidance (zone-based)
- ✅ Relative distance sorting
- ✅ Navigation planning (zone-aware)
- ✅ Object tracking (relative depth)
- ⚠️ Approximate distance measurement (with calibration)
- ❌ Precise distance measurement (use stereo/LiDAR)

## Technical Implementation Details

### Normalization Cache

To reduce flicker, percentile values are cached and smoothed:
```python
alpha = 0.3  # Smoothing factor
norm_p5 = alpha * p5 + (1 - alpha) * cached_p5
norm_p95 = alpha * p95 + (1 - alpha) * cached_p95
```

### Patch Sampling

```python
patch_size = 7  # 7x7 = 49 samples
half_patch = 3
region = depth_map[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
depth = np.median(region)  # Robust to outliers
```

### Zone Classification

```python
if normalized_depth < 0.33:
    zone = "Near"
elif normalized_depth < 0.66:
    zone = "Medium"
else:
    zone = "Far"
```

## References

- Depth Anything V2: [Paper/Repository]
- Monocular depth estimation limitations: Standard computer vision literature
- Percentile normalization: Robust statistics methods
- Patch sampling: Common practice in depth estimation

## Conclusion

YOLO3D provides **robust relative distance estimation** suitable for robotics applications requiring zone-based decision making. For absolute distance measurement, calibration is available but results are approximate. For high-precision requirements, consider stereo vision or LiDAR systems.
