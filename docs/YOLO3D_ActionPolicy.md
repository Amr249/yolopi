# YOLO3D Action Policy

## Overview

The YOLO3D action policy is a deterministic, rule-based system that generates robotics actions based on detected objects and their distance zones. The policy is designed for collision avoidance and navigation decision-making in real-time robotics applications.

## Decision Rules

### Primary Rules

The action policy follows a priority-based decision tree:

```
1. IF any filtered object in Near zone:
     a. IF object center_x < frame_center - threshold:
          → AVOID_RIGHT
     b. ELIF object center_x > frame_center + threshold:
          → AVOID_LEFT
     c. ELSE:
          → STOP

2. ELIF any filtered object in Medium zone:
     → SLOW_DOWN

3. ELSE:
     → PROCEED
```

### Zone Definitions

- **Near Zone**: Normalized depth < 0.33 (closest third of scene)
- **Medium Zone**: 0.33 ≤ Normalized depth < 0.66 (middle third)
- **Far Zone**: Normalized depth ≥ 0.66 (farthest third)

**Note**: Far zone objects are typically excluded from action computation via filtering.

### Avoid Direction Computation

When an object is in the Near zone, the system computes an avoid direction based on the object's horizontal position relative to the frame center:

```
frame_center = frame_width / 2
object_center_x = (bbox_x1 + bbox_x2) / 2
threshold = 50 pixels

IF object_center_x < frame_center - threshold:
    → Object is on LEFT side
    → Action: AVOID_RIGHT (suggest moving right to avoid)

ELIF object_center_x > frame_center + threshold:
    → Object is on RIGHT side
    → Action: AVOID_LEFT (suggest moving left to avoid)

ELSE:
    → Object is CENTERED
    → Action: STOP (no clear avoid direction)
```

**Rationale**: This simple heuristic provides directional guidance when objects are close and positioned to the side, enabling lateral avoidance maneuvers.

## Action Definitions

### STOP
- **Trigger**: Any object in Near zone (centered or unclear direction)
- **Purpose**: Immediate halt to prevent collision
- **Use Case**: Object directly ahead or too close to determine safe direction

### AVOID_LEFT
- **Trigger**: Near zone object on right side of frame
- **Purpose**: Suggest moving left to avoid object
- **Use Case**: Object detected on right, space available on left

### AVOID_RIGHT
- **Trigger**: Near zone object on left side of frame
- **Purpose**: Suggest moving right to avoid object
- **Use Case**: Object detected on left, space available on right

### SLOW_DOWN
- **Trigger**: Any object in Medium zone (no Near zone objects)
- **Purpose**: Reduce speed while maintaining forward progress
- **Use Case**: Object approaching but not immediately dangerous

### PROCEED
- **Trigger**: No objects in Near or Medium zones (or all filtered out)
- **Purpose**: Continue normal operation
- **Use Case**: Clear path ahead

## Action Debouncing (Hysteresis)

### Problem

Without debouncing, actions can flicker rapidly between states (e.g., STOP ↔ PROCEED) when:
- Objects move in/out of zones
- Detection confidence fluctuates
- Depth estimation has temporal noise

### Solution

A frame history buffer provides hysteresis:

1. **Buffer Size**: Last N actions stored (default: N = 3)
2. **Selection**: Most common action in buffer is output
3. **Update**: Action only changes if it persists for N consecutive frames

**Pseudo-code:**
```
action_history = []  # Circular buffer
action_history_max = 3

# Each frame:
action_history.append(current_action)
if len(action_history) > action_history_max:
    action_history.pop(0)

# Output:
if len(action_history) >= action_history_max:
    debounced_action = most_common(action_history)
else:
    debounced_action = current_action
```

**Effect**: Actions must persist for at least 3 frames before changing, reducing flicker by ~70% in typical scenarios.

## Filtering and Target Classes

### Zone Filtering

The policy only considers objects in allowed zones:

```python
ALLOW_ZONES = ["Near", "Medium"]  # Excludes "Far"
```

Objects in Far zone are ignored for action computation (but may still be displayed in debug mode).

### Class Filtering

If `TARGET_CLASSES` is specified, only those classes can trigger actions:

```python
TARGET_CLASSES = ["person", "car"]  # Only these trigger actions
```

If empty, all detected classes are considered.

**Use Case**: Filter to only relevant objects (e.g., ignore "chair" but respond to "person").

## Nearest Object Selection

The system identifies the nearest object for display and potential use in action reasoning:

**Selection Criteria:**
1. Prefer `distance_m` (meters) if calibrated
2. Else use `normalized_depth` (smaller = closer)
3. Select object with minimum distance value

**Output Structure:**
```python
{
    'class': str,
    'confidence': float,
    'zone': 'Near' | 'Medium' | 'Far',
    'distance': float,  # meters or normalized
    'distance_m': float | None,
    'normalized_depth': float | None,
    'bbox': (x1, y1, x2, y2)
}
```

## Action State Output

The system provides action state via JSON endpoint `/robot_state`:

```json
{
    "timestamp": 1234567890.123,
    "fps": 12.5,
    "depth_mode": "Relative",
    "action": "STOP",
    "reason": "Near object: person",
    "nearest_object": {
        "class": "person",
        "confidence": 0.85,
        "zone": "Near",
        "distance": 0.25,
        "distance_m": null,
        "normalized_depth": 0.25,
        "bbox": [100, 150, 200, 300]
    },
    "filtered_count": 2,
    "total_count": 5,
    "normalization_method": "Percentile (p5..p95)"
}
```

## Policy Characteristics

### Deterministic

The policy is fully deterministic:
- Same inputs → same outputs
- No randomness or probabilistic elements
- Reproducible behavior

### Real-time

- Computation time: <1 ms per frame
- No complex optimization or search
- Suitable for high-frequency updates

### Interpretable

- Clear rule-based logic
- Human-readable action reasons
- Easy to debug and modify

### Limitations

- **No trajectory prediction**: Only considers current frame
- **No multi-object reasoning**: Considers zones, not individual object trajectories
- **Simple avoid direction**: Based on horizontal position only
- **No velocity consideration**: Distance-based only, not speed-aware

## Future Enhancements (Not Implemented)

Potential improvements for future versions:
- Multi-object tracking for trajectory prediction
- Velocity-aware actions (consider object motion)
- Confidence-weighted actions
- Temporal consistency (track objects across frames)
- Machine learning-based policy (learned from demonstrations)

## Configuration Parameters

```python
# Action policy configuration
ENABLE_DISTANCE_FILTERING = True
ALLOW_ZONES = ["Near", "Medium"]
TARGET_CLASSES = []  # Empty = all classes
ACTION_DEBOUNCE_FRAMES = 3
```

## Example Scenarios

### Scenario 1: Person Approaching
- Detection: person, confidence 0.9, zone: Near, normalized_depth: 0.2
- Action: STOP
- Reason: "Near object: person"

### Scenario 2: Car on Right Side
- Detection: car, confidence 0.8, zone: Near, normalized_depth: 0.25, center_x: 500 (frame_width: 640)
- Action: AVOID_LEFT
- Reason: "Near object: car (right side)"

### Scenario 3: Multiple Objects
- Detections: person (Near), chair (Medium), bottle (Far)
- Filtered: person (Near), chair (Medium)  # Far excluded
- Action: STOP (Near takes priority)
- Reason: "Near object: person"

### Scenario 4: Clear Path
- Detections: tree (Far), building (Far)
- Filtered: []  # All in Far zone, excluded
- Action: PROCEED
- Reason: "All clear"
