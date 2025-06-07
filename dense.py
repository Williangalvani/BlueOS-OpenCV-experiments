import numpy as np
import cv2 as cv
import time
import math

# Decimation factor - reduce image size by this factor for better performance
DECIMATION_FACTOR = 5

cap = cv.VideoCapture(cv.samples.findFile("video.mp4"))

# Get original video properties
original_fps = cap.get(cv.CAP_PROP_FPS)
total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
print(f"Original video FPS: {original_fps:.2f}")
print(f"Total frames: {int(total_frames)}")

ret, frame1 = cap.read()
original_height, original_width = frame1.shape[:2]
center = (original_width // 2, original_height // 2)

# Decimate the first frame
frame1_small = cv.resize(frame1, None, fx=1/DECIMATION_FACTOR, fy=1/DECIMATION_FACTOR, interpolation=cv.INTER_NEAREST)
prvs = cv.cvtColor(frame1_small, cv.COLOR_BGR2GRAY)

# Pre-allocate arrays to avoid memory allocation overhead
h, w = prvs.shape
flow = np.zeros((h, w, 2), dtype=np.float32)
mag = np.zeros((h, w), dtype=np.float32)
ang = np.zeros((h, w), dtype=np.float32)

# Pre-allocate HSV array for visualization
hsv = np.zeros((h, w, 3), dtype=np.uint8)
hsv[..., 1] = 255

# Create grid of points for motion estimation
step = 8  # Sample every 8th pixel for efficiency
y_coords, x_coords = np.mgrid[step//2:h-step//2:step, step//2:w-step//2:step]
points1 = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).astype(np.float32)

# Variables for FPS calculation
frame_count = 0
start_time = time.time()
fps_display = 0

# Motion tracking variables
pan_x_smooth = 0
pan_y_smooth = 0
rotation_smooth = 0
alpha = 0.5  # Smoothing factor

# Accumulation variables for stabilization
accumulated_rotation = 0  # Total rotation accumulation for stabilization
accumulated_pan_x = 0
accumulated_pan_y = 0

# Motion vector visualization settings
vector_scale = 10  # Scale factor to make motion vector visible

def extract_global_motion(flow_field, points):
    """Extract global pan and rotation from optical flow field"""
    # Sample flow vectors at grid points
    flow_vectors = []
    source_points = []
    
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < flow_field.shape[1] and 0 <= y < flow_field.shape[0]:
            fx, fy = flow_field[y, x]
            # Only use points with significant motion
            if np.sqrt(fx*fx + fy*fy) > 0.5:
                source_points.append([x, y])
                flow_vectors.append([x + fx, y + fy])
    
    if len(source_points) < 10:  # Need minimum points for robust estimation
        return 0, 0, 0
    
    source_points = np.array(source_points, dtype=np.float32)
    flow_points = np.array(flow_vectors, dtype=np.float32)
    
    # Get image center for proper rotation decomposition
    h, w = flow_field.shape[:2]
    center_x, center_y = w / 2.0, h / 2.0
    
    # Estimate affine transformation using RANSAC
    try:
        transform, mask = cv.estimateAffinePartial2D(source_points, flow_points, 
                                                    method=cv.RANSAC, 
                                                    ransacReprojThreshold=2.0)
        if transform is not None:
            # The transform is: [cos(θ) -sin(θ) tx]
            #                   [sin(θ)  cos(θ) ty]
            # But this assumes rotation around origin. We need rotation around center.
            
            # Extract rotation angle
            rotation_rad = math.atan2(transform[1, 0], transform[0, 0])
            rotation_deg = math.degrees(rotation_rad)
            
            # Extract translation - but this includes the apparent translation due to 
            # rotation around origin instead of center. We need to correct this.
            tx_raw = transform[0, 2]
            ty_raw = transform[1, 2]
            
            # Calculate the apparent translation caused by rotation around origin vs center
            cos_r = math.cos(rotation_rad)
            sin_r = math.sin(rotation_rad)
            
            # The rotation around center can be decomposed as:
            # 1. Translate to origin: (-center_x, -center_y)
            # 2. Rotate around origin
            # 3. Translate back: (+center_x, +center_y)
            # The net translation due to this rotation is:
            rotation_translation_x = center_x * (cos_r - 1) + center_y * sin_r
            rotation_translation_y = -center_x * sin_r + center_y * (cos_r - 1)
            
            # The pure translation (pan) is the raw translation minus the rotation-induced translation
            pure_pan_x = tx_raw - rotation_translation_x
            pure_pan_y = ty_raw - rotation_translation_y
            
            return pure_pan_x, pure_pan_y, rotation_deg
    except:
        pass
    
    return 0, 0, 0

def apply_rotation_correction(image, rotation_angle, center):
    """Apply inverse rotation to stabilize the image"""
    if abs(rotation_angle) < 0.1:  # Skip tiny rotations
        return image
    
    # Create rotation matrix for inverse rotation
    rotation_matrix = cv.getRotationMatrix2D(center, 0, 1.0)
    
    # Apply rotation
    corrected = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return corrected

def update_visualization_points(points, colors, pan_x, pan_y, rotation_deg, width, height):
    """Update visualization points based on detected pan and rotation motion"""
    # Scale pan from decimated resolution to original resolution
    scaled_pan_x = pan_x * DECIMATION_FACTOR
    scaled_pan_y = pan_y * DECIMATION_FACTOR
    
    # For visualization: if camera rotates clockwise (+), scene appears to rotate counter-clockwise (-)
    # So we negate the rotation to show how the scene appears to move
    scene_rotation_deg = 0 #-rotation_deg
    
    # Apply rotation around image center
    center_x, center_y = width // 2, height // 2
    rotation_rad = math.radians(scene_rotation_deg)
    cos_rot = math.cos(rotation_rad)
    sin_rot = math.sin(rotation_rad)
    
    # Apply incremental motion to each point (not accumulating on the points themselves)
    new_points = np.copy(points)
    
    for i in range(len(points)):
        # Get current position relative to center
        rel_x = points[i, 0] - center_x
        rel_y = points[i, 1] - center_y
        
        # Apply rotation
        rotated_rel_x = rel_x * cos_rot - rel_y * sin_rot
        rotated_rel_y = rel_x * sin_rot + rel_y * cos_rot
        
        # Apply translation and convert back to absolute coordinates
        new_points[i, 0] = rotated_rel_x + center_x + scaled_pan_x
        new_points[i, 1] = rotated_rel_y + center_y + scaled_pan_y
    
    # Update the points array
    points[:] = new_points
    
    # Regenerate points that have moved off screen
    for i in range(len(points)):
        if (points[i, 0] < 0 or points[i, 0] >= width or 
            points[i, 1] < 0 or points[i, 1] >= height):
            # Regenerate point at random location
            points[i] = np.random.rand(2) * [width, height]
            colors[i] = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
    
    return points, colors

def draw_motion_vector(image, pan_x, pan_y, rotation_deg):
    """Draw motion vector arrow showing pan direction and rotation indicator"""
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Draw center cross to show reference point
    cv.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
    cv.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)
    cv.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)
    
    # Scale pan motion for visibility
    scaled_pan_x = pan_x * DECIMATION_FACTOR * vector_scale
    scaled_pan_y = pan_y * DECIMATION_FACTOR * vector_scale
    
    # Calculate end point of motion vector
    end_x = int(center_x + scaled_pan_x)
    end_y = int(center_y + scaled_pan_y)
    
    # Draw motion vector arrow
    if abs(scaled_pan_x) > 1 or abs(scaled_pan_y) > 1:  # Only draw if motion is significant
        # Main arrow line
        cv.arrowedLine(image, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
        
        # Add magnitude text
        magnitude = math.sqrt(scaled_pan_x**2 + scaled_pan_y**2)
        cv.putText(image, f'Pan: {magnitude/vector_scale/DECIMATION_FACTOR:.1f}px', 
                   (center_x + 30, center_y - 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw rotation indicator (arc around center)
    if abs(rotation_deg) > 0.1:
        # Draw rotation arc
        radius = 80
        start_angle = -90  # Start from top
        arc_angle = int(rotation_deg * 3)  # Scale rotation for visibility
        
        # Choose color based on rotation direction
        rot_color = (0, 0, 255) if rotation_deg > 0 else (255, 0, 0)  # Red for CCW, Blue for CW
        
        cv.ellipse(image, (center_x, center_y), (radius, radius), 0, 
                   start_angle, start_angle + arc_angle, rot_color, 3)
        
        # Add rotation text
        cv.putText(image, f'Rot: {rotation_deg:.2f}°', 
                   (center_x + 30, center_y + 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, rot_color, 2)
        
        # Draw rotation direction indicator
        if rotation_deg > 0:
            cv.putText(image, 'CW', (center_x + 90, center_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, rot_color, 2)
        else:
            cv.putText(image, 'CCW', (center_x + 90, center_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, rot_color, 2)
    
    return image

while(1):
    loop_start = time.time()
    
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    # Decimate the current frame
    frame2_small = cv.resize(frame2, None, fx=1/DECIMATION_FACTOR, fy=1/DECIMATION_FACTOR, interpolation=cv.INTER_NEAREST)
    next = cv.cvtColor(frame2_small, cv.COLOR_BGR2GRAY)
    
    # Optimized Farneback parameters for speed:
    cv.calcOpticalFlowFarneback(prvs, next, flow, 0.5, 2, 10, 2, 5, 1.2, 0)
    
    # Extract global motion parameters
    pan_x, pan_y, rotation = extract_global_motion(flow, points1)
    
    # Apply smoothing to reduce noise
    pan_x_smooth = alpha * pan_x + (1 - alpha) * pan_x_smooth
    pan_y_smooth = alpha * pan_y + (1 - alpha) * pan_y_smooth  
    rotation_smooth = alpha * rotation + (1 - alpha) * rotation_smooth
    
    # Accumulate motion deltas for stabilization
    accumulated_rotation -= rotation_smooth
    accumulated_pan_x += pan_x_smooth
    accumulated_pan_y += pan_y_smooth
    
    # Apply rotation correction using accumulated rotation
    frame2_unrotated = apply_rotation_correction(frame2, accumulated_rotation, center)
    
    # Update visualization every 3 frames for better performance
    if frame_count % 3 == 0:
        cv.cartToPolar(flow[..., 0], flow[..., 1], mag, ang)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        
        # Add motion information to the optical flow image
        cv.putText(bgr, f'FPS: {fps_display:.1f}', (10, 25), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(bgr, f'Pan X: {pan_x_smooth:.1f}px', (10, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(bgr, f'Pan Y: {pan_y_smooth:.1f}px', (10, 75), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(bgr, f'Rotation: {rotation_smooth:.2f}°', (10, 100), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(bgr, f'Size: {w}x{h}', (10, 125), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create copies for drawing points (so we don't modify originals)
        frame2_with_points = frame2.copy()
        frame2_unrotated_with_points = frame2_unrotated.copy()
        
        # Draw motion vector on both frames
        frame2_with_points = draw_motion_vector(frame2_with_points, pan_x_smooth, pan_y_smooth, rotation_smooth)
        frame2_unrotated_with_points = draw_motion_vector(frame2_unrotated_with_points, pan_x_smooth, pan_y_smooth, rotation_smooth)
        
        # Add text to original and unrotated frames
        cv.putText(frame2_with_points, 'Original + Motion Vector', (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.putText(frame2_unrotated_with_points, f'Unrotated ({-accumulated_rotation:.2f}°)', (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display all three windows
        cv.imshow('Optical Flow', bgr)
        cv.imshow('Original Frame', frame2_with_points)
        cv.imshow('Rotation Corrected', frame2_unrotated_with_points)
    
    # Calculate and display FPS
    frame_count += 1
    if frame_count % 10 == 0:  # Update FPS display every 10 frames
        elapsed_time = time.time() - start_time
        fps_display = frame_count / elapsed_time
        print(f"Processing FPS: {fps_display:.1f} | Instant Rot: {rotation_smooth:.2f}° | Accumulated: {accumulated_rotation:.2f}°")
    
    # Calculate loop time
    loop_time = time.time() - loop_start
    
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('p'):  # Press 'p' to print detailed stats
        print(f"Current processing FPS: {fps_display:.2f}")
        print(f"Loop time: {loop_time*1000:.2f}ms")
        print(f"Speedup vs original: {fps_display/original_fps:.2f}x")
        print(f"Frame size: {w}x{h}")
        print(f"Instantaneous motion - Pan: ({pan_x_smooth:.2f}, {pan_y_smooth:.2f}), Rotation: {rotation_smooth:.3f}°")
        print(f"Accumulated motion - Pan: ({accumulated_pan_x:.2f}, {accumulated_pan_y:.2f}), Rotation: {accumulated_rotation:.3f}°")
    elif k == ord('r'):  # Press 'r' to reset accumulation
        accumulated_rotation = 0
        accumulated_pan_x = 0
        accumulated_pan_y = 0
        print("Reset accumulation values to zero")
    elif k == ord('s'):  # Press 's' to save current frame
        cv.imwrite('optical_flow_output.png', bgr)
        cv.imwrite('original_frame.png', frame2_with_points)
        cv.imwrite('unrotated_frame.png', frame2_unrotated_with_points)
        print("Saved optical_flow_output.png, original_frame.png, and unrotated_frame.png")
    prvs = next

# Final statistics
total_time = time.time() - start_time
final_fps = frame_count / total_time
print(f"\nFinal Statistics:")
print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
print(f"Average processing FPS: {final_fps:.2f}")
print(f"Speedup vs original video: {final_fps/original_fps:.2f}x")
print(f"Processing resolution: {w}x{h} pixels")

cv.destroyAllWindows()
