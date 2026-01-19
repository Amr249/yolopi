# YOLO3D Data Flow Diagram

## System Data Flow (ASCII Diagram)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOLO3D Pipeline                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Camera     │
│  (640×480)   │
└──────┬───────┘
       │ Frame
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FRAME PREPROCESSING                                   │
│  ┌──────────────┐                                                       │
│  │ Resize to    │                                                       │
│  │ 640×384      │                                                       │
│  └──────┬───────┘                                                       │
└─────────┼───────────────────────────────────────────────────────────────┘
          │
          ├─────────────────────────────────────┐
          │                                     │
          ▼                                     ▼
┌─────────────────────┐              ┌──────────────────────┐
│   YOLO11n           │              │  Depth Anything V2   │
│   (Every Frame)     │              │  (Throttled: N=4)    │
│                     │              │                      │
│  Input: 640×384    │              │  Input: 384×384      │
│  Output: 2D Boxes   │              │  Output: Depth Map    │
│  + Classes          │              │  (Raw)               │
└──────┬──────────────┘              └──────┬───────────────┘
       │                                     │
       │                                     │
       │                                     ▼
       │                          ┌──────────────────────┐
       │                          │  Percentile Norm     │
       │                          │  (p5, p95) + EMA     │
       │                          │                      │
       │                          │  Output: D_norm      │
       │                          │  [0, 1]              │
       │                          └──────┬───────────────┘
       │                                   │
       │                                   │ Cache
       │                                   │ (if throttled)
       │                                   ▼
       │                          ┌──────────────────────┐
       │                          │  Cached Depth Map    │
       │                          │  (Reused if skip)    │
       │                          └──────┬───────────────┘
       │                                   │
       └───────────────────────────────────┘
                    │
                    ▼
       ┌────────────────────────────┐
       │   FUSION & PROCESSING      │
       │                            │
       │  For each 2D detection:    │
       │  ┌──────────────────────┐  │
       │  │ Extract 7×7 patch    │  │
       │  │ at bbox center       │  │
       │  │                      │  │
       │  │ Compute median       │  │
       │  │ depth: d_obj         │  │
       │  └──────┬───────────────┘  │
       │         │                  │
       │         ▼                  │
       │  ┌──────────────────────┐  │
       │  │ Zone Classification  │  │
       │  │                      │  │
       │  │ Near:  d < 0.33      │  │
       │  │ Med:   0.33 ≤ d < 0.66│ │
       │  │ Far:   d ≥ 0.66      │  │
       │  └──────┬───────────────┘  │
       │         │                  │
       │         ▼                  │
       │  ┌──────────────────────┐  │
       │  │ Optional: Meter      │  │
       │  │ Conversion           │  │
       │  │ (if calibrated)      │  │
       │  └──────┬───────────────┘  │
       └─────────┼──────────────────┘
                 │
                 ▼
       ┌────────────────────────────┐
       │   DETECTION OUTPUTS        │
       │                            │
       │  • Full detections list    │
       │  • Filtered detections     │
       │    (by zone/class)         │
       │  • Nearest object          │
       │  • Zone per object         │
       │  • Distance (norm/meters)  │
       └─────────┬──────────────────┘
                 │
                 ▼
       ┌────────────────────────────┐
       │   ACTION POLICY            │
       │                            │
       │  ┌──────────────────────┐ │
       │  │ Rule Engine:          │ │
       │  │ • Near → STOP         │ │
       │  │ • Medium → SLOW_DOWN  │ │
       │  │ • Else → PROCEED      │ │
       │  │ • Avoid direction     │ │
       │  │   (if Near + side)    │ │
       │  └──────┬───────────────┘ │
       │         │                  │
       │         ▼                  │
       │  ┌──────────────────────┐ │
       │  │ Action Debouncing    │ │
       │  │ (N-frame history)    │ │
       │  │                      │ │
       │  │ Most common action   │ │
       │  │ in buffer            │ │
       │  └──────┬───────────────┘ │
       └─────────┼──────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT INTERFACES                                 │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │  Video Stream    │  │  Web UI          │  │  JSON Endpoint   │      │
│  │  (MJPEG)         │  │  (Flask)         │  │  /robot_state    │      │
│  │                  │  │                  │  │                  │      │
│  │  • 3D Boxes       │  │  • Stats Panel   │  │  • Action        │      │
│  │  • Depth Overlay  │  │  • Robot Mode    │  │  • Nearest Obj   │      │
│  │  • Action Overlay │  │  • Detections    │  │  • FPS           │      │
│  │  • FPS            │  │  • Zones         │  │  • Timestamp     │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
Camera
  │
  ├─→ Frame Resize (640×384)
  │
  ├─→ YOLO11n Detector
  │     │
  │     └─→ 2D Detections (bbox, class, conf)
  │
  └─→ Depth Anything V2 (throttled)
        │
        ├─→ Raw Depth Map
        │
        └─→ Percentile Normalization
              │
              ├─→ EMA Smoothing
              │
              └─→ Normalized Depth Map [0,1]
                    │
                    └─→ Cache (if throttled)
                          │
                          └─→ Reused on skipped frames

2D Detections + Normalized Depth
  │
  └─→ Fusion Module
        │
        ├─→ Per-Object Patch Sampling (7×7 median)
        │
        ├─→ Zone Classification
        │
        ├─→ Optional Meter Conversion
        │
        └─→ Detection Outputs
              │
              ├─→ Filtering (zone/class)
              │
              ├─→ Nearest Object Selection
              │
              └─→ Action Policy
                    │
                    └─→ Debouncing
                          │
                          └─→ Final Action
```

## Data Structures

### Detection Object
```
{
  'bbox': (x1, y1, x2, y2),
  'class': str,
  'confidence': float,
  'normalized_depth': float [0,1],
  'zone': 'Near' | 'Medium' | 'Far',
  'distance_m': float | None,
  'class_id': int
}
```

### Nearest Object
```
{
  'class': str,
  'confidence': float,
  'zone': str,
  'distance': float,
  'distance_m': float | None,
  'normalized_depth': float | None,
  'bbox': (x1, y1, x2, y2)
}
```

### Robot State (JSON)
```
{
  'timestamp': float,
  'fps': float,
  'depth_mode': 'Relative' | 'Meters',
  'action': 'STOP' | 'SLOW_DOWN' | 'PROCEED' | 'AVOID_LEFT' | 'AVOID_RIGHT',
  'reason': str,
  'nearest_object': dict | null,
  'filtered_count': int,
  'total_count': int,
  'normalization_method': str
}
```

## Caching Strategy

```
Frame N:     [Depth Compute] → Cache
Frame N+1:   [Depth Skip] → Use Cache
Frame N+2:   [Depth Skip] → Use Cache
Frame N+3:   [Depth Skip] → Use Cache
Frame N+4:   [Depth Compute] → Update Cache
```

**Throttle Factor**: Default N=4 (depth computed every 4 frames, cached for 3)
