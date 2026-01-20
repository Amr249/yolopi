"""
PHASE 2: Camera Depth Bring-Up (NO AI yet)
Goal: Access and visualize raw Astra Pro depth reliably using V4L2.

This script:
1. Opens RGB stream from /dev/video0
2. Opens depth stream from /dev/video1 (V4L2)
3. Reads frames from both streams in the same loop
4. Processes depth (uint16 mm -> meters)
5. Displays RGB and depth streams separately
6. Verifies depth frame properties (size, units, invalid values)

Requirements:
- OpenCV only (no external SDKs)
- Astra Pro camera with RGB and depth devices
- V4L2 support on Raspberry Pi
"""

import cv2
import numpy as np
import time
import sys

print("=" * 60)
print("PHASE 2: Camera Depth Bring-Up (V4L2)")
print("=" * 60)
print("Goal: Access and visualize raw Astra Pro depth reliably")
print("Using: OpenCV with V4L2 backend")
print("=" * 60)

# Camera device paths
RGB_DEVICE = "/dev/video0"
DEPTH_DEVICE = "/dev/video1"

# Stream configuration
RGB_WIDTH = 640
RGB_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480

print(f"\nOpening cameras:")
print(f"  RGB:  {RGB_DEVICE}")
print(f"  Depth: {DEPTH_DEVICE} (V4L2)")

# Open RGB camera
rgb_cap = cv2.VideoCapture(RGB_DEVICE)
if not rgb_cap.isOpened():
    print(f"\n❌ Error: Failed to open RGB camera at {RGB_DEVICE}")
    print("Please check:")
    print(f"  1. Device exists: ls -l {RGB_DEVICE}")
    print(f"  2. Permissions: sudo chmod 666 {RGB_DEVICE}")
    print(f"  3. Device is not in use by another process")
    sys.exit(1)

# Configure RGB stream
rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, RGB_WIDTH)
rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB_HEIGHT)
rgb_cap.set(cv2.CAP_PROP_FPS, 30)

# Get actual RGB properties
actual_rgb_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_rgb_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
print(f"✓ RGB camera opened: {actual_rgb_width}x{actual_rgb_height} @ {actual_rgb_fps:.1f}fps")

# Open depth camera using V4L2 backend
depth_cap = cv2.VideoCapture(DEPTH_DEVICE, cv2.CAP_V4L2)
if not depth_cap.isOpened():
    print(f"\n❌ Error: Failed to open depth camera at {DEPTH_DEVICE}")
    print("Please check:")
    print(f"  1. Device exists: ls -l {DEPTH_DEVICE}")
    print(f"  2. Permissions: sudo chmod 666 {DEPTH_DEVICE}")
    print(f"  3. V4L2 support: v4l2-ctl --list-devices")
    print(f"  4. Depth format: v4l2-ctl --device={DEPTH_DEVICE} --list-formats-ext")
    rgb_cap.release()
    sys.exit(1)

# Configure depth stream
depth_cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEPTH_WIDTH)
depth_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEPTH_HEIGHT)
depth_cap.set(cv2.CAP_PROP_FPS, 30)

# Set depth format to Y16 (16-bit grayscale) if supported
# Some cameras may need this for proper depth reading
try:
    depth_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
except:
    pass  # Not all cameras support this

# Get actual depth properties
actual_depth_width = int(depth_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_depth_height = int(depth_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_depth_fps = depth_cap.get(cv2.CAP_PROP_FPS)
print(f"✓ Depth camera opened: {actual_depth_width}x{actual_depth_height} @ {actual_depth_fps:.1f}fps")

# Depth processing parameters
DEPTH_SCALE_MM_TO_M = 1000.0  # Convert millimeters to meters
INVALID_DEPTH_VALUE = 0  # Depth values of 0 are invalid

print("\n" + "=" * 60)
print("PHASE 2: RGB + Depth Streams")
print("=" * 60)
print("Controls:")
print("  'q' - Quit")
print("  's' - Save screenshots")
print("  'i' - Print depth info to console")
print("=" * 60)

# FPS tracking
fps_buffer = []
fps_avg_len = 30
frame_count = 0
start_time = time.time()

# Depth statistics
depth_stats = {
    'min_m': float('inf'),
    'max_m': float('-inf'),
    'valid_pixels': 0,
    'invalid_pixels': 0,
    'total_pixels': 0
}

try:
    while True:
        frame_start = time.time()
        
        # Read RGB frame
        rgb_ret, rgb_frame = rgb_cap.read()
        if not rgb_ret or rgb_frame is None:
            print("Warning: Failed to read RGB frame, skipping...")
            continue
        
        # Read depth frame
        depth_ret, depth_frame_raw = depth_cap.read()
        if not depth_ret or depth_frame_raw is None:
            print("Warning: Failed to read depth frame, skipping...")
            continue
        
        # Process depth frame
        # Depth frame is uint16 (millimeters)
        # If OpenCV reads it as 3-channel, extract first channel
        if len(depth_frame_raw.shape) == 3:
            # Convert BGR to grayscale if needed, or use first channel
            depth_frame_uint16 = depth_frame_raw[:, :, 0].astype(np.uint16)
            # If it's actually BGR format, reconstruct 16-bit value
            # Some cameras output depth in BGR format where B and G form the 16-bit value
            if depth_frame_raw.dtype == np.uint8:
                depth_frame_uint16 = depth_frame_raw[:, :, 0].astype(np.uint16) * 256 + depth_frame_raw[:, :, 1].astype(np.uint16)
        else:
            depth_frame_uint16 = depth_frame_raw.astype(np.uint16)
        
        # Create mask for invalid depth values (0 = invalid)
        invalid_mask = (depth_frame_uint16 == INVALID_DEPTH_VALUE)
        valid_mask = ~invalid_mask
        
        # Convert depth from mm to meters
        depth_frame_m = depth_frame_uint16.astype(np.float32) / DEPTH_SCALE_MM_TO_M
        
        # Calculate depth statistics
        if np.any(valid_mask):
            valid_depths = depth_frame_m[valid_mask]
            depth_stats['min_m'] = float(np.min(valid_depths))
            depth_stats['max_m'] = float(np.max(valid_depths))
            depth_stats['valid_pixels'] = int(np.sum(valid_mask))
        else:
            depth_stats['min_m'] = 0.0
            depth_stats['max_m'] = 0.0
            depth_stats['valid_pixels'] = 0
        
        depth_stats['invalid_pixels'] = int(np.sum(invalid_mask))
        depth_stats['total_pixels'] = depth_frame_uint16.size
        
        # Prepare depth for visualization (normalize for colormap)
        # Clamp depth to reasonable range for visualization (0-5m)
        depth_viz = np.clip(depth_frame_m, 0, 5.0)
        depth_normalized = (depth_viz / 5.0 * 255).astype(np.uint8)
        
        # Mark invalid pixels as 0 (black) in visualization
        depth_normalized[invalid_mask] = 0
        
        # Apply colormap for visualization (JET colormap)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Add depth info overlay to depth visualization
        info_y = 30
        cv2.putText(depth_colormap, f"Depth: {actual_depth_width}x{actual_depth_height}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(depth_colormap, f"Format: uint16 (mm -> m)",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30
        
        if depth_stats['valid_pixels'] > 0:
            cv2.putText(depth_colormap, 
                       f"Range: {depth_stats['min_m']:.2f}m - {depth_stats['max_m']:.2f}m",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 30
        else:
            cv2.putText(depth_colormap, "Range: No valid depth",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            info_y += 30
        
        cv2.putText(depth_colormap,
                   f"Valid: {depth_stats['valid_pixels']} / {depth_stats['total_pixels']}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add RGB info overlay
        h, w = rgb_frame.shape[:2]
        cv2.putText(rgb_frame, f"RGB: {w}x{h}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Calculate and display FPS
        frame_end = time.time()
        frame_dt = frame_end - frame_start
        if frame_dt > 0:
            fps = 1.0 / frame_dt
            fps_buffer.append(fps)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
        else:
            avg_fps = 0
        
        # Add FPS to RGB frame
        cv2.putText(rgb_frame, f"FPS: {avg_fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display both streams
        cv2.imshow("RGB Stream", rgb_frame)
        cv2.imshow("Depth Stream (Colormap)", depth_colormap)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            rgb_filename = f"phase2_rgb_{timestamp}.jpg"
            depth_filename = f"phase2_depth_{timestamp}.jpg"
            cv2.imwrite(rgb_filename, rgb_frame)
            cv2.imwrite(depth_filename, depth_colormap)
            print(f"Screenshots saved:")
            print(f"  RGB:   {rgb_filename}")
            print(f"  Depth: {depth_filename}")
        elif key == ord('i'):
            # Print depth info to console
            print("\n" + "=" * 60)
            print("Depth Frame Info:")
            print("=" * 60)
            print(f"Resolution: {actual_depth_width}x{actual_depth_height}")
            print(f"Depth Format: uint16 (millimeters)")
            print(f"Depth Scale: {DEPTH_SCALE_MM_TO_M} (mm -> meters)")
            print(f"Valid Pixels: {depth_stats['valid_pixels']} / {depth_stats['total_pixels']}")
            print(f"Invalid Pixels: {depth_stats['invalid_pixels']}")
            if depth_stats['valid_pixels'] > 0:
                print(f"Depth Range: {depth_stats['min_m']:.2f}m - {depth_stats['max_m']:.2f}m")
            else:
                print(f"Depth Range: No valid depth data")
            print(f"Frame Count: {frame_count}")
            if avg_fps > 0:
                print(f"Average FPS: {avg_fps:.1f}")
            print("=" * 60)
        
        frame_count += 1

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\n❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup: Release both cameras
    print("\nCleaning up...")
    if rgb_cap.isOpened():
        rgb_cap.release()
        print("✓ RGB camera released")
    if depth_cap.isOpened():
        depth_cap.release()
        print("✓ Depth camera released")
    cv2.destroyAllWindows()
    print("\n✓ Phase 2 test completed")
    
    # Print final statistics
    total_time = time.time() - start_time
    if total_time > 0:
        print(f"\nTotal frames processed: {frame_count}")
        print(f"Total time: {total_time:.1f}s")
        if frame_count > 0:
            print(f"Average FPS: {frame_count / total_time:.1f}")
