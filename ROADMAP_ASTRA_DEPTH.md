# 🚀 Roadmap: From YOLO-Only → Depth-Camera-Based AI Perception

This roadmap assumes:

- You already finished Phase 1 (YOLO-only, stable)
- You are using test_yolo3d.py
- You want to use Astra Pro hardware depth, NOT ML depth

## ✅ PHASE 1 (DONE / BASELINE)

**Status:** ✔ Completed or almost done

### Goal
Stable YOLO object detection on Raspberry Pi.

### Output
- YOLO detections
- FPS overlay
- Low CPU usage
- No depth logic

This is your baseline.

## 🟡 PHASE 2 — Camera Depth Bring-Up (NO AI yet)

### Goal
Access and visualize raw Astra depth reliably.

### Steps
1. Detect Astra Pro device
2. Verify RGB + Depth streams
3. Read depth frames
4. Display:
   - RGB stream
   - Depth stream (grayscale / colormap)
5. Verify:
   - Depth frame size
   - Depth units (mm or meters)
   - Handle invalid depth values (0 / NaN)

### Output
- RGB window
- Depth window
- No YOLO yet

📌 **No fusion, no AI logic here**

## 🟡 PHASE 3 — RGB + Depth Synchronization

### Goal
Ensure RGB and depth frames correspond correctly.

### Steps
1. Capture RGB and depth in same loop
2. Align depth to RGB (if required)
3. Resize depth to match RGB
4. Confirm pixel-to-pixel correspondence

### Output
- Single RGB frame
- Matching depth frame
- Verified alignment

📌 **Still no YOLO fusion**

## 🟢 PHASE 4 — YOLO + Depth Fusion (Core Perception)

### Goal
Compute object distance using hardware depth.

### Steps
1. Run YOLO on RGB frame
2. For each bounding box:
   - Compute bbox center
   - Extract depth value at center
   - Sample small patch (e.g. 7×7)
   - Use median depth
3. Handle edge cases:
   - Invalid depth
   - Out-of-range values
4. Convert depth to meters
5. Attach depth to each detection

### Output
Each detection has:
- Class
- Confidence
- Distance (meters)

## 🟢 PHASE 5 — Distance Zones (Robotics-Friendly)

### Goal
Convert raw distance → semantic zones.

### Steps
1. Define thresholds:
   - Near
   - Medium
   - Far
2. Classify each object
3. Color-code bounding boxes
4. Display distance + zone label

### Output
```
Person | 0.85 | 1.2 m | NEAR
```

## 🟢 PHASE 6 — Nearest-Obstacle Logic

### Goal
Find the most critical object.

### Steps
1. Among all detections:
   - Select nearest valid object
2. Track:
   - Distance
   - Bounding box position
3. Visualize:
   - Highlight nearest object
   - Display "NEAREST OBJECT"

### Output
One "primary obstacle" per frame

## 🟢 PHASE 7 — Action Signals (Still NO ROS)

### Goal
Generate high-level robot actions.

### Steps
1. Define rules:
   - STOP (near)
   - SLOW (medium)
   - PROCEED (far)
2. Optional:
   - LEFT / RIGHT decision based on bbox x-position
3. Display action on screen

### Output
```
ACTION: STOP
```

📌 **Still no motors, no ROS**

## 🟢 PHASE 8 — Stability & Performance Tuning

### Goal
Make it demo-ready.

### Steps
1. Optional frame skipping
2. Optional depth sampling throttle
3. FPS measurement
4. CPU usage measurement
5. Confirm no crashes over long run

## 🟣 PHASE 9 — Documentation & Experiments

### Goal
Prepare graduation-ready material.

### Steps
1. Capture screenshots
2. Record FPS and CPU usage
3. Describe pipeline
4. Compare:
   - YOLO-only
   - YOLO + ML depth (old)
   - YOLO + Astra depth (final)
