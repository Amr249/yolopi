# YOLO-D (YOLO + Depth) Detection for Raspberry Pi

Real-time depth-aware object detection using YOLO11n + Depth Anything V2.

## Architecture

- **YOLO11n**: Fast 2D object detection
- **Depth Anything V2**: Monocular depth estimation
- **Fusion**: Extract depth at bounding box centers for each detection

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
python app.py
```

The web interface will be available at: `http://<pi-ip>:5000`

## Files

- `yolo3d_detector.py`: YOLO-D detector module (optimized for Pi CPU)
- `app.py`: Flask web application with video streaming
- `yolo11n.pt`: YOLO model file (should be in repo)

## Performance Optimizations

1. **Depth Throttling**: Processes depth every 3 frames (configurable)
2. **Reduced Input Size**: Depth model uses 384x384 input (faster)
3. **CPU-Only**: All inference runs on CPU with `torch.no_grad()`
4. **Caching**: Depth maps are cached and reused between frames

## Configuration

Edit `app.py` to adjust:

```python
detector = YOLODetector(
    yolo_model_path="yolo11n.pt",
    depth_model_name="depth-anything/Depth-Anything-V2-Small-hf",
    depth_input_size=(384, 384),  # Smaller = faster
    depth_throttle=3,  # Process depth every N frames
    conf_threshold=0.5
)
```

## Expected Performance

- **FPS**: 5-15 FPS (depending on Pi model and scene complexity)
- **Latency**: ~100-200ms per frame
- **Memory**: ~500MB-1GB RAM usage

## Troubleshooting

### Depth model fails to load
- Falls back to geometry-based depth estimation
- Uses known object sizes to estimate distance
- Less accurate but still functional

### Low FPS
- Increase `depth_throttle` (e.g., 5 or 10)
- Reduce `depth_input_size` (e.g., 256x256)
- Reduce camera resolution in `app.py`

### Out of memory
- Reduce `depth_input_size`
- Increase `depth_throttle`
- Close other applications

## Notes

- Depth values are relative (0.5-10 meters range)
- For absolute depth, calibrate camera intrinsics
- First run will download depth model (~100-200MB)
