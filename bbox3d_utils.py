"""
3D Bounding Box Utilities for YOLO-3D
Functions for creating and visualizing 3D bounding boxes from 2D detections + depth.
"""

import cv2
import numpy as np


def project_3d_to_2d(points_3d, camera_matrix, dist_coeffs=None):
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: Nx3 array of 3D points (X, Y, Z)
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients (optional)
    
    Returns:
        points_2d: Nx2 array of 2D image points
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))
    
    points_2d, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        dist_coeffs
    )
    return points_2d.reshape(-1, 2)


def create_3d_bbox_from_2d(bbox_2d, depth, camera_matrix, 
                          object_dimensions=None, default_dimensions=(1.0, 1.0, 1.0)):
    """
    Create a 3D bounding box from 2D detection and depth.
    Dimensions are calculated from 2D bbox size to ensure proper fit.
    
    Args:
        bbox_2d: 2D bounding box (x1, y1, x2, y2)
        depth: Depth value at bbox center (in meters)
        camera_matrix: 3x3 camera intrinsic matrix
        object_dimensions: Dict with 'width', 'height', 'length' ratios (optional)
        default_dimensions: Default (width, height, length) if object_dimensions is None
    
    Returns:
        corners_3d: 8x3 array of 3D box corners
        center_3d: 3D center point
    """
    x1, y1, x2, y2 = bbox_2d
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Calculate 2D bounding box dimensions in pixels
    bbox_width_px = x2 - x1
    bbox_height_px = y2 - y1
    
    # Calculate 3D dimensions from 2D bbox and depth
    # Real size = (pixel size * depth) / focal_length
    width_3d = (bbox_width_px * depth) / fx
    height_3d = (bbox_height_px * depth) / fy
    
    # For length (depth dimension), use aspect ratio from known dimensions if available
    # Otherwise estimate from width
    if object_dimensions:
        # Use aspect ratio from known dimensions
        known_width = object_dimensions.get('width', width_3d)
        known_length = object_dimensions.get('length', width_3d * 0.5)
        length_ratio = known_length / known_width if known_width > 0 else 0.5
        length_3d = width_3d * length_ratio
    else:
        # Default: assume object is roughly as deep as it is wide
        length_3d = width_3d * 0.5
    
    # Ensure minimum dimensions to avoid tiny boxes
    width_3d = max(width_3d, 0.1)
    height_3d = max(height_3d, 0.1)
    length_3d = max(length_3d, 0.1)
    
    # Calculate 3D center from 2D center and depth
    center_x_2d = (x1 + x2) / 2
    center_y_2d = (y1 + y2) / 2
    
    # Back-project to 3D
    center_3d_x = (center_x_2d - cx) * depth / fx
    center_3d_y = (center_y_2d - cy) * depth / fy
    center_3d_z = depth
    
    center_3d = np.array([center_3d_x, center_3d_y, center_3d_z])
    
    # Create 3D box corners (assuming upright orientation, facing camera)
    # Box is centered at center_3d with calculated dimensions
    half_w = width_3d / 2
    half_h = height_3d / 2
    half_l = length_3d / 2
    
    # 8 corners of the 3D box
    corners_3d = np.array([
        # Front face (closer to camera)
        [center_3d_x - half_w, center_3d_y - half_h, center_3d_z - half_l],
        [center_3d_x + half_w, center_3d_y - half_h, center_3d_z - half_l],
        [center_3d_x + half_w, center_3d_y + half_h, center_3d_z - half_l],
        [center_3d_x - half_w, center_3d_y + half_h, center_3d_z - half_l],
        # Back face (farther from camera)
        [center_3d_x - half_w, center_3d_y - half_h, center_3d_z + half_l],
        [center_3d_x + half_w, center_3d_y - half_h, center_3d_z + half_l],
        [center_3d_x + half_w, center_3d_y + half_h, center_3d_z + half_l],
        [center_3d_x - half_w, center_3d_y + half_h, center_3d_z + half_l],
    ])
    
    return corners_3d, center_3d


def draw_3d_bbox(frame, corners_3d, camera_matrix, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box on 2D image with enhanced 3D visualization.
    
    Args:
        frame: Image to draw on
        corners_3d: 8x3 array of 3D box corners
        camera_matrix: 3x3 camera intrinsic matrix
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        frame: Image with 3D box drawn
    """
    # Project 3D corners to 2D
    corners_2d = project_3d_to_2d(corners_3d, camera_matrix)
    corners_2d = corners_2d.astype(int)
    
    h, w = frame.shape[:2]
    
    # Define edges of the 3D box (12 edges of a cube)
    # Front face (closer to camera) - draw with full color and thickness
    front_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    # Back face (farther from camera) - draw with lighter/dashed style
    back_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    # Connecting edges (depth edges)
    connecting_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    
    # Draw front face (thicker, brighter)
    for edge in front_edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(frame, pt1, pt2, color, thickness + 1)
    
    # Draw back face (thinner, slightly darker to show depth)
    back_color = tuple(int(c * 0.6) for c in color)  # Darker version
    for edge in back_edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(frame, pt1, pt2, back_color, max(1, thickness - 1))
    
    # Draw connecting edges (depth lines)
    for edge in connecting_edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            # Draw dashed line for depth edges
            draw_dashed_line(frame, pt1, pt2, color, thickness)
    
    return frame


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """
    Draw a dashed line between two points.
    """
    dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    if dist < dash_length:
        cv2.line(img, pt1, pt2, color, thickness)
        return
    
    num_dashes = int(dist / dash_length)
    dx = (pt2[0] - pt1[0]) / num_dashes
    dy = (pt2[1] - pt1[1]) / num_dashes
    
    for i in range(0, num_dashes, 2):
        start = (int(pt1[0] + i * dx), int(pt1[1] + i * dy))
        end = (int(pt1[0] + (i + 1) * dx), int(pt1[1] + (i + 1) * dy))
        cv2.line(img, start, end, color, thickness)


def draw_depth_overlay(frame, bbox_2d, depth_map, alpha=0.3):
    """
    Draw semi-transparent depth overlay inside bounding box.
    Blue color indicates depth information.
    
    Args:
        frame: Image to draw on
        bbox_2d: 2D bounding box (x1, y1, x2, y2)
        depth_map: Depth map (numpy array)
        alpha: Transparency (0.0 to 1.0)
    
    Returns:
        frame: Image with depth overlay
    """
    if depth_map is None:
        return frame
    
    x1, y1, x2, y2 = bbox_2d
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    # Extract depth region
    depth_region = depth_map[y1:y2, x1:x2]
    if depth_region.size == 0:
        return frame
    
    # Normalize depth to 0-255 for visualization
    depth_norm = cv2.normalize(depth_region, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    
    # Create blue overlay (BGR format: blue channel)
    overlay = frame.copy()
    overlay_region = overlay[y1:y2, x1:x2]
    
    # Create blue tint based on depth
    # Closer objects = brighter blue, farther = darker blue
    blue_channel = depth_norm
    green_channel = (depth_norm * 0.3).astype(np.uint8)  # Slight green tint
    red_channel = np.zeros_like(depth_norm)
    
    # Combine channels
    blue_overlay = np.stack([blue_channel, green_channel, red_channel], axis=2)
    
    # Blend with original image
    overlay_region[:] = cv2.addWeighted(overlay_region, 1 - alpha, blue_overlay, alpha, 0)
    overlay[y1:y2, x1:x2] = overlay_region
    
    return overlay


def create_bird_eye_view(detections, camera_matrix, bev_size=(400, 400), 
                        view_range=10.0, camera_height=1.5):
    """
    Create Bird's Eye View (BEV) visualization.
    
    Args:
        detections: List of detections with 3D info
        camera_matrix: Camera intrinsic matrix
        bev_size: Size of BEV image (width, height)
        view_range: Range of view in meters (front/back/left/right)
        camera_height: Height of camera above ground (meters)
    
    Returns:
        bev_image: Bird's eye view image
    """
    bev_w, bev_h = bev_size
    bev_image = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    
    # Draw grid
    grid_spacing = 1.0  # 1 meter
    center_x, center_y = bev_w // 2, bev_h // 2
    pixels_per_meter = bev_w / (2 * view_range)
    
    # Draw grid lines
    for i in range(-int(view_range), int(view_range) + 1):
        x = int(center_x + i * pixels_per_meter)
        if 0 <= x < bev_w:
            cv2.line(bev_image, (x, 0), (x, bev_h), (50, 50, 50), 1)
        
        y = int(center_y + i * pixels_per_meter)
        if 0 <= y < bev_h:
            cv2.line(bev_image, (0, y), (bev_w, y), (50, 50, 50), 1)
    
    # Draw center (camera position)
    cv2.circle(bev_image, (center_x, center_y), 5, (0, 255, 255), -1)
    cv2.line(bev_image, (center_x, center_y), 
            (center_x, center_y - 20), (0, 255, 255), 2)  # Forward direction
    
    # Draw detections
    colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
        (88, 159, 106), (96, 202, 231), (159, 124, 168),
        (169, 162, 241), (98, 118, 150), (172, 176, 184)
    ]
    
    for i, det in enumerate(detections):
        if 'center_3d' not in det:
            continue
        
        center_3d = det['center_3d']
        x_3d, y_3d, z_3d = center_3d
        
        # Project to BEV (X-Z plane, Y is up)
        # In BEV: X is left-right, Z is forward-backward
        bev_x = int(center_x + x_3d * pixels_per_meter)
        bev_y = int(center_y - z_3d * pixels_per_meter)  # Negative Z is forward
        
        if 0 <= bev_x < bev_w and 0 <= bev_y < bev_h:
            color = colors[det.get('class_id', 0) % len(colors)]
            cv2.circle(bev_image, (bev_x, bev_y), 8, color, -1)
            
            # Draw label
            label = f"{det['class']}"
            cv2.putText(bev_image, label, (bev_x + 10, bev_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Add axis labels
    cv2.putText(bev_image, "Forward", (center_x - 30, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bev_image, f"Range: {view_range}m", (10, bev_h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return bev_image


def visualize_depth_map(depth_map, colormap=cv2.COLORMAP_INFERNO):
    """
    Visualize depth map as color image.
    
    Args:
        depth_map: Depth map (numpy array)
        colormap: OpenCV colormap
    
    Returns:
        depth_vis: Colored depth visualization
    """
    if depth_map is None:
        return None
    
    # Normalize depth map to 0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # Apply colormap
    depth_vis = cv2.applyColorMap(depth_normalized, colormap)
    
    return depth_vis
