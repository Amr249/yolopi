# YOLO3D Performance Characteristics

## Performance Metrics

| Metric | Before Phase 1 | After Phase 1 | After Phase 2 | Notes |
|--------|---------------|---------------|---------------|-------|
| **Processing Resolution** | 640×480 | 640×384 | 640×384 | 20% reduction in pixels |
| **Depth Throttle** | Every frame (1:1) | Every 4 frames (1:4) | Every 4 frames (1:4) | 75% reduction in depth computation |
| **YOLO Inference** | Every frame | Every frame | Every frame | No change |
| **Depth Input Size** | 384×384 | 384×384 | 384×384 | Depth model input |
| **Estimated FPS** | 3-8 FPS | 8-15 FPS | 8-15 FPS | 2-3x improvement |
| **Frame Latency** | 200-300 ms | 100-200 ms | 100-200 ms | Reduced by ~40% |
| **Memory Usage** | ~800 MB | ~600 MB | ~600 MB | Reduced by frame resizing |

## Component Timing Breakdown

| Component | Time per Frame | Frequency | Notes |
|-----------|---------------|-----------|-------|
| **Frame Capture** | ~10 ms | Every frame | USB camera I/O |
| **Frame Resize** | ~5 ms | Every frame | 640×480 → 640×384 |
| **YOLO Inference** | 30-50 ms | Every frame | 640×384 input |
| **Depth Estimation** | 200-400 ms | Every 4 frames | 384×384 input, throttled |
| **Depth Normalization** | ~5 ms | Every 4 frames | Percentile + EMA |
| **Patch Sampling** | ~1 ms | Every frame | 7×7 median per detection |
| **Zone Classification** | <1 ms | Every frame | Simple threshold |
| **Action Policy** | <1 ms | Every frame | Rule-based |
| **Action Debouncing** | <1 ms | Every frame | History buffer |

## Performance vs Configuration

### Resolution Impact

| Resolution | Pixels | Relative Speed | FPS Estimate |
|------------|--------|----------------|-------------|
| 640×480 | 307,200 | 1.0x (baseline) | 3-8 FPS |
| 640×384 | 245,760 | 1.25x | 8-15 FPS |
| 480×288 | 138,240 | 2.2x | 15-25 FPS |
| 320×192 | 61,440 | 5.0x | 25-40 FPS |

**Note**: Lower resolutions reduce accuracy but increase speed.

### Depth Throttle Impact

| Throttle Interval | Depth Compute Frequency | Relative Speed | FPS Estimate |
|-------------------|------------------------|----------------|--------------|
| 1 (every frame) | 100% | 1.0x | 3-8 FPS |
| 2 (every 2 frames) | 50% | 1.5x | 5-12 FPS |
| 4 (every 4 frames) | 25% | 2.0x | 8-15 FPS |
| 8 (every 8 frames) | 12.5% | 2.5x | 10-18 FPS |

**Note**: Higher throttle intervals reduce depth update rate but increase overall FPS.

### Depth Input Size Impact

| Depth Input Size | Pixels | Relative Speed | Quality Impact |
|------------------|--------|----------------|---------------|
| 256×256 | 65,536 | 2.25x | Lower accuracy |
| 384×384 | 147,456 | 1.0x (baseline) | Balanced |
| 512×512 | 262,144 | 0.56x | Higher accuracy |

**Note**: Smaller depth input reduces computation but may reduce depth map quality.

## Raspberry Pi Model Performance

| Pi Model | CPU | RAM | Estimated FPS | Notes |
|----------|-----|-----|---------------|-------|
| **Pi 3B+** | 4×1.4 GHz | 1 GB | 5-8 FPS | May need lower resolution |
| **Pi 4 (2GB)** | 4×1.5 GHz | 2 GB | 8-12 FPS | Recommended minimum |
| **Pi 4 (4GB)** | 4×1.5 GHz | 4 GB | 10-15 FPS | Optimal |
| **Pi 5** | 4×2.4 GHz | 4-8 GB | 12-20 FPS | Best performance |

## Memory Usage Breakdown

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| **YOLO Model** | ~50 MB | yolo11n.pt loaded |
| **Depth Model** | ~200-300 MB | Depth Anything V2 Small |
| **Frame Buffers** | ~10 MB | 640×384×3 bytes × buffers |
| **Depth Maps** | ~5 MB | Cached depth map |
| **Python Runtime** | ~100 MB | Base Python + libraries |
| **Total** | ~400-500 MB | Typical usage |

## Optimization Trade-offs

| Optimization | Speed Gain | Accuracy Impact | When to Use |
|--------------|-----------|-----------------|-------------|
| **Frame Resize (640×384)** | +25% | Minimal | Always recommended |
| **Depth Throttle (N=4)** | +100% | Slight (cached) | Default setting |
| **Depth Input (384×384)** | Baseline | Baseline | Balanced |
| **Depth Input (256×256)** | +125% | Moderate | If FPS critical |
| **Lower Resolution (480×288)** | +100% | Moderate | If FPS critical |

## Recommended Configurations

### High Performance (Pi 4/5)
```python
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 384
DEPTH_THROTTLE_INTERVAL = 4
depth_input_size = (384, 384)
```
**Expected**: 10-15 FPS

### Balanced (Pi 4)
```python
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 384
DEPTH_THROTTLE_INTERVAL = 4
depth_input_size = (384, 384)
```
**Expected**: 8-12 FPS

### Low Resource (Pi 3B+)
```python
PROCESSING_WIDTH = 480
PROCESSING_HEIGHT = 288
DEPTH_THROTTLE_INTERVAL = 6
depth_input_size = (256, 256)
```
**Expected**: 5-8 FPS

## Notes

- **FPS measurements** are approximate and depend on scene complexity, number of detections, and thermal throttling
- **Depth throttling** trades temporal resolution for speed; cached depth maps provide continuity
- **Resolution reduction** primarily affects YOLO accuracy at distance, not depth estimation quality
- **Memory usage** may vary based on number of detections and frame buffer management
- **Thermal throttling** on Pi can reduce sustained performance; adequate cooling recommended
