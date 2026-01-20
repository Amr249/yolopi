# ONNX Runtime Experiment for YOLO3D

## Overview

This document describes the experimental ONNX Runtime optimization path for YOLO object detection in the YOLO3D pipeline. ONNX Runtime is tested as an optional alternative to PyTorch for YOLO inference to potentially reduce CPU usage on Raspberry Pi.

## Motivation

### Why ONNX Runtime?

1. **Optimized Execution**: ONNX Runtime includes CPU-specific optimizations that may outperform PyTorch's CPU inference on ARM architectures (Raspberry Pi).
2. **Lower Memory Footprint**: ONNX models typically have a smaller memory footprint than PyTorch models in runtime.
3. **Hardware Acceleration**: While this experiment uses CPU-only execution, ONNX Runtime supports various execution providers for future acceleration (e.g., TensorRT, OpenVINO).
4. **Performance Testing**: Comparing ONNX Runtime vs PyTorch provides empirical data on which backend performs better on the target hardware.

### Expected Benefits on ARM CPU

- **Potential 10-30% CPU reduction** (experimental, needs measurement)
- **Lower memory usage** (smaller runtime footprint)
- **Faster inference latency** (optimized graph execution)

**Note**: These are theoretical expectations. Actual performance depends on:
- Raspberry Pi model (Pi 4, Pi 5, etc.)
- Python version and dependencies
- ONNX Runtime version
- Model complexity and input size

## Architecture

### System Design

```
┌─────────────────────────────────────────────┐
│         YOLO3D Detector (app.py)            │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
   ┌───▼────┐     ┌───▼──────────┐
   │ PyTorch│ OR  │ ONNX Runtime │  (YOLO only)
   │  YOLO  │     │    YOLO      │
   └───┬────┘     └───┬──────────┘
       │               │
       └───────┬───────┘
               │
       ┌───────▼────────┐
       │ Depth Model    │  (PyTorch - unchanged)
       │ (PyTorch only) │
       └────────────────┘
```

### Key Components

1. **`onnx/export_yolo_to_onnx.py`**: Script to export YOLO PyTorch model to ONNX format.
2. **`onnx/onnx_yolo_detector.py`**: ONNX Runtime-based YOLO detector class.
3. **`yolo3d_detector.py`**: Modified to support both PyTorch and ONNX Runtime backends with automatic fallback.

## Implementation Details

### 1. Model Export

**File**: `onnx/export_yolo_to_onnx.py`

Exports `yolo11n.pt` to `yolo11n.onnx` with:
- Dynamic batch size support
- ONNX opset 12 (compatible with ONNX Runtime)
- Graph simplification for optimization

**Usage**:
```bash
python onnx/export_yolo_to_onnx.py
```

**Output**: `yolo11n.onnx` in the root directory.

### 2. ONNX Runtime Detector

**File**: `onnx/onnx_yolo_detector.py`

`OnnxYoloDetector` class:
- Uses `CPUExecutionProvider` only (no GPU)
- Preprocesses frames with letterbox padding (matches YOLO preprocessing)
- Postprocesses ONNX outputs to bounding boxes
- Returns detections in format compatible with `yolo3d_detector.py`

### 3. Integration with YOLO3D

**File**: `yolo3d_detector.py`

**Configuration Flag**:
```python
USE_ONNX_YOLO = False  # Set True to enable ONNX Runtime
ONNX_MODEL_PATH = "yolo11n.onnx"
```

**Behavior**:
- If `USE_ONNX_YOLO = True` and ONNX model exists: uses ONNX Runtime
- If ONNX fails to load or inference fails: automatically falls back to PyTorch
- If `USE_ONNX_YOLO = False`: uses PyTorch (default)

**Timing Comparison**:
- Logs inference timing every 100 frames
- Compares PyTorch vs ONNX Runtime performance
- Prints speedup ratio when both backends are tested

### 4. Graceful Fallback

The system includes multiple fallback layers:

1. **Import Fallback**: If `onnxruntime` is not installed, ONNX path is disabled.
2. **Model File Fallback**: If ONNX model file is missing, falls back to PyTorch.
3. **Runtime Fallback**: If ONNX inference fails during execution, falls back to PyTorch.

All fallbacks are automatic and logged with warnings. The system never crashes due to ONNX issues.

## Limitations

### GPU Acceleration NOT Used

- This experiment uses **CPU-only** execution (`CPUExecutionProvider`)
- No CUDA, TensorRT, OpenVINO, or other GPU acceleration
- ONNX Runtime GPU providers are not tested (Raspberry Pi GPU is VideoCore, not CUDA-compatible)

### Depth Model Unchanged

- Depth estimation remains PyTorch-based (not converted to ONNX)
- Only YOLO object detection uses ONNX Runtime
- Future work could explore ONNX Runtime for depth estimation

### Compatibility

- ONNX Runtime must be installed: `pip install onnxruntime`
- ONNX model must be exported first (run export script)
- Model format: ONNX opset 12 (compatible with most ONNX Runtime versions)

### Performance Claims

- No performance guarantees without measurement
- Results vary by hardware and workload
- This is an **experimental optimization** for testing purposes

## Usage

### Step 1: Install ONNX Runtime

```bash
pip install onnxruntime
```

### Step 2: Export YOLO Model to ONNX

```bash
python onnx/export_yolo_to_onnx.py
```

This creates `yolo11n.onnx` in the root directory.

### Step 3: Enable ONNX Runtime

Edit `yolo3d_detector.py`:

```python
USE_ONNX_YOLO = True  # Enable ONNX Runtime
```

### Step 4: Run Application

```bash
python app.py
```

The system will:
1. Attempt to load ONNX Runtime detector
2. Log timing comparisons every 100 frames
3. Fall back to PyTorch if ONNX fails

### Step 5: Monitor Performance

Watch the console output for timing logs:

```
⏱️  Inference timing (last 100 frames):
   PyTorch: 45.23 ms/frame
   ONNX Runtime: 38.12 ms/frame
   Speedup: 1.19x
```

## Testing on Raspberry Pi

### Recommended Test Procedure

1. **Baseline Measurement**:
   - Set `USE_ONNX_YOLO = False`
   - Run application for 1-2 minutes
   - Note average CPU usage and FPS

2. **ONNX Runtime Measurement**:
   - Set `USE_ONNX_YOLO = True`
   - Export ONNX model first
   - Run application for 1-2 minutes
   - Note average CPU usage and FPS

3. **Compare Results**:
   - Compare CPU usage: `htop` or `top`
   - Compare FPS: Check application FPS overlay
   - Compare inference timing: Check console logs

### Expected Observations

- **CPU Usage**: May decrease by 10-30% (needs verification)
- **FPS**: May increase by 5-15% (needs verification)
- **Memory**: May decrease slightly (needs verification)
- **Latency**: Inference time may decrease (check console logs)

**Note**: Actual results vary by Pi model, Python version, and workload.

## Troubleshooting

### ONNX Model Not Found

**Error**: `⚠ ONNX model not found at yolo11n.onnx`

**Solution**: Run the export script first:
```bash
python onnx/export_yolo_to_onnx.py
```

### ONNX Runtime Not Installed

**Error**: `⚠ Warning: ONNX Runtime not available. Falling back to PyTorch YOLO.`

**Solution**: Install ONNX Runtime:
```bash
pip install onnxruntime
```

### ONNX Inference Fails

**Error**: `⚠ ONNX Runtime inference failed: ...`

**Solution**: The system automatically falls back to PyTorch. Check error message for details. Common issues:
- Model format incompatible (re-export with export script)
- Input shape mismatch (check preprocessing)
- ONNX Runtime version incompatible (update: `pip install --upgrade onnxruntime`)

## Future Work

### Potential Enhancements

1. **ONNX Runtime for Depth Estimation**: Convert depth model to ONNX for consistency.
2. **Quantization**: Explore INT8 quantization for further CPU reduction.
3. **Execution Providers**: Test other execution providers (if available on Pi).
4. **Automatic Selection**: Choose backend based on performance measurements.
5. **Benchmarking Tool**: Create automated benchmarking script for performance comparison.

### Not Planned

- GPU acceleration (Raspberry Pi GPU is VideoCore, not CUDA-compatible)
- ROS2 integration changes
- Depth model ONNX conversion (Phase 1 scope: YOLO only)

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Ultralytics YOLO Export Guide](https://docs.ultralytics.com/modes/export/)
- [ONNX Model Format](https://onnx.ai/)

## Conclusion

This ONNX Runtime experiment provides an optional optimization path for YOLO inference on Raspberry Pi. It is designed as a **non-breaking addition** with automatic fallback to PyTorch, ensuring the system remains stable and functional regardless of ONNX availability or performance.

**Remember**: This is experimental. Results should be measured on actual hardware before making deployment decisions.
