"""
PHASE 2: Camera Depth Bring-Up (NO AI yet)
Goal: Access and visualize raw Astra Pro depth reliably.

This script:
1. Detects Astra Pro device
2. Verifies RGB + Depth streams
3. Reads depth frames
4. Displays RGB and Depth streams separately
5. Verifies depth frame properties (size, units, invalid values)
"""

import cv2
import numpy as np
import time
import sys

print("=" * 60)
print("PHASE 2: Camera Depth Bring-Up")
print("=" * 60)
print("Goal: Access and visualize raw Astra Pro depth reliably")
print("=" * 60)

# Try to import Orbbec SDK
try:
    import pyorbbecsdk as obs
    ORBBEC_SDK_AVAILABLE = True
    print("✓ pyorbbecsdk available")
except ImportError:
    ORBBEC_SDK_AVAILABLE = False
    print("⚠ pyorbbecsdk not available - will try OpenCV fallback")
    print("  Install with: pip install pyorbbecsdk")

# Configuration
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
COLOR_WIDTH = 640
COLOR_HEIGHT = 480


def test_opencv_astra():
    """
    Try to access Astra camera using OpenCV (fallback method).
    This may work if the camera is recognized as a standard UVC device.
    """
    print("\n[OpenCV Method] Attempting to access camera via OpenCV...")
    
    # Try different camera indices
    for cam_idx in range(5):
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            print(f"  Found camera at index {cam_idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, COLOR_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, COLOR_HEIGHT)
            
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ Successfully read frame: {frame.shape}")
                cap.release()
                return cap_idx, None  # No depth support in basic OpenCV
            cap.release()
    
    print("  ⚠ No working camera found via OpenCV")
    return None, None


def test_orbbec_sdk():
    """
    Access Astra Pro camera using Orbbec SDK.
    This is the proper method for accessing depth data.
    """
    print("\n[Orbbec SDK Method] Attempting to access Astra Pro via Orbbec SDK...")
    
    try:
        # Initialize pipeline
        ctx = obs.Context()
        
        # Query devices
        device_list = ctx.query_devices()
        device_count = device_list.get_count()
        
        if device_count == 0:
            print("  ⚠ No Orbbec devices found")
            return None, None
        
        print(f"  ✓ Found {device_count} Orbbec device(s)")
        
        # Connect to first device
        device = device_list.get_device_by_index(0)
        
        # Get device info
        device_info = device.get_device_info()
        print(f"  Device: {device_info.get_name()}")
        print(f"  Serial: {device_info.get_serial_number()}")
        
        # Create pipeline
        pipeline = obs.Pipeline(device)
        
        # Configure streams
        config = obs.Config()
        
        # Try to enable depth stream
        depth_profile_list = pipeline.get_stream_profile_list(obs.OBSensorType.DEPTH_SENSOR)
        if depth_profile_list.get_count() > 0:
            depth_profile = depth_profile_list.get_video_stream_profile(640, 0, obs.OBFormat.D16, 30)
            if depth_profile:
                config.enable_stream(depth_profile)
                print(f"  ✓ Enabled depth stream: 640x480 @ 30fps")
            else:
                # Try default profile
                depth_profile = depth_profile_list.get_default_video_stream_profile(obs.OBSensorType.DEPTH_SENSOR)
                config.enable_stream(depth_profile)
                print(f"  ✓ Enabled default depth stream")
        else:
            print("  ⚠ No depth stream profiles available")
            return None, None
        
        # Try to enable color stream
        color_profile_list = pipeline.get_stream_profile_list(obs.OBSensorType.COLOR_SENSOR)
        if color_profile_list.get_count() > 0:
            color_profile = color_profile_list.get_video_stream_profile(640, 0, obs.OBFormat.RGB, 30)
            if color_profile:
                config.enable_stream(color_profile)
                print(f"  ✓ Enabled color stream: 640x480 @ 30fps")
            else:
                color_profile = color_profile_list.get_default_video_stream_profile(obs.OBSensorType.COLOR_SENSOR)
                config.enable_stream(color_profile)
                print(f"  ✓ Enabled default color stream")
        
        # Start pipeline
        pipeline.start(config)
        print("  ✓ Pipeline started successfully")
        
        return pipeline, device
        
    except Exception as e:
        print(f"  ❌ Error accessing Orbbec SDK: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def visualize_depth_colormap(depth_frame, scale=1000.0):
    """
    Convert depth frame to colormap visualization.
    
    Args:
        depth_frame: Depth frame in mm (16-bit)
        scale: Scale factor to convert to meters (default 1000.0 for mm->m)
    
    Returns:
        Colormap visualization (BGR for OpenCV display)
    """
    if depth_frame is None:
        return None
    
    # Convert to numpy array
    if isinstance(depth_frame, np.ndarray):
        depth_data = depth_frame
    else:
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        height = depth_frame.get_height()
        width = depth_frame.get_width()
        depth_data = depth_data.reshape((height, width))
    
    # Handle invalid values (0 or NaN)
    depth_data = depth_data.astype(np.float32)
    depth_data[depth_data == 0] = np.nan
    
    # Convert to meters if needed (assuming input is in mm)
    depth_meters = depth_data / scale
    
    # Normalize for visualization (0-5m range)
    depth_normalized = np.clip(depth_meters / 5.0, 0, 1)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)
    
    # Apply colormap (JET colormap for depth visualization)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # Mark invalid pixels as black
    invalid_mask = np.isnan(depth_data)
    depth_colormap[invalid_mask] = [0, 0, 0]
    
    return depth_colormap


def main():
    """
    Main function for PHASE 2: Camera Depth Bring-Up
    """
    pipeline = None
    device = None
    
    # Try Orbbec SDK first (proper method)
    if ORBBEC_SDK_AVAILABLE:
        pipeline, device = test_orbbec_sdk()
    
    # If Orbbec SDK fails or unavailable, try OpenCV fallback
    if pipeline is None:
        print("\n⚠ Orbbec SDK method failed or unavailable")
        print("⚠ Switching to OpenCV fallback (RGB only, no depth)")
        print("\nTo enable depth support, install pyorbbecsdk:")
        print("  pip install pyorbbecsdk")
        print("\nContinuing with OpenCV fallback for RGB visualization...")
        
        cam_idx, _ = test_opencv_astra()
        if cam_idx is None:
            print("\n❌ Failed to access any camera")
            print("Please check:")
            print("  1. Camera is connected")
            print("  2. Camera drivers are installed")
            print("  3. For Astra Pro depth: Install pyorbbecsdk")
            sys.exit(1)
        
        # OpenCV fallback loop (RGB only)
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, COLOR_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, COLOR_HEIGHT)
        
        print("\n" + "=" * 60)
        print("PHASE 2: RGB Stream (Depth not available)")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("=" * 60)
        
        while True:
            ret, color_frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Display RGB
            cv2.imshow("RGB Stream", color_frame)
            
            # Display info overlay
            h, w = color_frame.shape[:2]
            info_text = f"RGB: {w}x{h} | Depth: Not Available"
            cv2.putText(color_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("RGB Stream", color_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"phase2_rgb_{int(time.time())}.jpg"
                cv2.imwrite(filename, color_frame)
                print(f"Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Orbbec SDK pipeline (with depth support)
    print("\n" + "=" * 60)
    print("PHASE 2: RGB + Depth Streams")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'i' - Print depth info")
    print("=" * 60)
    
    frame_count = 0
    depth_stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'valid_pixels': 0,
        'invalid_pixels': 0,
        'total_pixels': 0
    }
    
    try:
        while True:
            # Wait for frames
            frameset = pipeline.wait_for_frames(100)  # 100ms timeout
            if frameset is None:
                continue
            
            color_frame = None
            depth_frame = None
            
            # Extract color frame
            try:
                color_frame_data = frameset.get_color_frame()
                if color_frame_data:
                    # Convert to numpy array
                    color_data = np.frombuffer(
                        color_frame_data.get_data(),
                        dtype=np.uint8
                    )
                    height = color_frame_data.get_height()
                    width = color_frame_data.get_width()
                    color_data = color_data.reshape((height, width, 3))
                    # Convert RGB to BGR for OpenCV
                    color_frame = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Warning: Failed to get color frame: {e}")
            
            # Extract depth frame
            try:
                depth_frame_data = frameset.get_depth_frame()
                if depth_frame_data:
                    depth_frame = depth_frame_data
                    
                    # Get depth properties
                    depth_width = depth_frame.get_width()
                    depth_height = depth_frame.get_height()
                    depth_scale = device.get_depth_scale()  # Usually 1000.0 (mm)
                    
                    # Convert to numpy for analysis
                    depth_data = np.frombuffer(
                        depth_frame.get_data(),
                        dtype=np.uint16
                    ).reshape((depth_height, depth_width))
                    
                    # Analyze depth frame
                    valid_mask = (depth_data > 0)
                    depth_stats['valid_pixels'] = np.sum(valid_mask)
                    depth_stats['invalid_pixels'] = np.sum(~valid_mask)
                    depth_stats['total_pixels'] = depth_height * depth_width
                    
                    if np.any(valid_mask):
                        valid_depths = depth_data[valid_mask]
                        depth_stats['min'] = np.min(valid_depths) / depth_scale
                        depth_stats['max'] = np.max(valid_depths) / depth_scale
                    
                    # Visualize depth
                    depth_colormap = visualize_depth_colormap(depth_frame, depth_scale)
                    
                    # Display depth info overlay
                    if depth_colormap is not None:
                        info_y = 30
                        cv2.putText(depth_colormap, 
                                   f"Depth: {depth_width}x{depth_height}",
                                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        info_y += 30
                        cv2.putText(depth_colormap,
                                   f"Scale: {depth_scale:.1f} (mm->m)",
                                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        info_y += 30
                        cv2.putText(depth_colormap,
                                   f"Range: {depth_stats['min']:.2f}m - {depth_stats['max']:.2f}m",
                                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        info_y += 30
                        cv2.putText(depth_colormap,
                                   f"Valid: {depth_stats['valid_pixels']} / {depth_stats['total_pixels']}",
                                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        cv2.imshow("Depth Stream (Colormap)", depth_colormap)
                        
            except Exception as e:
                print(f"Warning: Failed to get depth frame: {e}")
            
            # Display color frame
            if color_frame is not None:
                h, w = color_frame.shape[:2]
                cv2.putText(color_frame, f"RGB: {w}x{h}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("RGB Stream", color_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                if color_frame is not None:
                    cv2.imwrite(f"phase2_rgb_{timestamp}.jpg", color_frame)
                    print(f"Screenshot saved: phase2_rgb_{timestamp}.jpg")
                if depth_colormap is not None:
                    cv2.imwrite(f"phase2_depth_{timestamp}.jpg", depth_colormap)
                    print(f"Screenshot saved: phase2_depth_{timestamp}.jpg")
            elif key == ord('i'):
                # Print depth info
                print("\n" + "=" * 60)
                print("Depth Frame Info:")
                print("=" * 60)
                print(f"Resolution: {depth_frame.get_width()}x{depth_frame.get_height()}")
                print(f"Depth Scale: {depth_scale:.1f} (converts mm to meters)")
                print(f"Valid Pixels: {depth_stats['valid_pixels']} / {depth_stats['total_pixels']}")
                print(f"Invalid Pixels: {depth_stats['invalid_pixels']}")
                print(f"Depth Range: {depth_stats['min']:.2f}m - {depth_stats['max']:.2f}m")
                print("=" * 60)
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("\n✓ Phase 2 test completed")


if __name__ == "__main__":
    main()
