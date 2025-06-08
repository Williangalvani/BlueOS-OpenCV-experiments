import numpy as np
import cv2 as cv
import time
import math
import argparse
import os
import csv

# Set up command line arguments
parser = argparse.ArgumentParser(description='Dense Optical Flow with global motion estimation (headless version)')
parser.add_argument('--video', type=str, help='Path to video file (default: use video.mp4)')
parser.add_argument('--output-dir', type=str, default='/app/output', help='Output directory for videos')
parser.add_argument('--max-frames', type=int, default=300, help='Maximum number of frames to process (default: 300)')
parser.add_argument('--decimation', type=int, default=5, help='Decimation factor for performance (default: 5)')
parser.add_argument('--save-videos', action='store_true', help='Save output videos (default: only print motion data)')
parser.add_argument('--save-reference', type=str, help='Save pan/tilt reference data to specified file (CSV format)')

# Optical Flow Parameters (Farneback algorithm)
parser.add_argument('--pyr-scale', type=float, default=0.5, help='Pyramid scale factor (default: 0.5)')
parser.add_argument('--levels', type=int, default=2, help='Number of pyramid levels (default: 2)')
parser.add_argument('--winsize', type=int, default=10, help='Averaging window size (default: 10)')
parser.add_argument('--iterations', type=int, default=2, help='Number of iterations at each pyramid level (default: 2)')
parser.add_argument('--poly-n', type=int, default=5, help='Size of pixel neighborhood for polynomial expansion (default: 5)')
parser.add_argument('--poly-sigma', type=float, default=1.2, help='Standard deviation for Gaussian kernel (default: 1.2)')

# Motion Estimation Parameters
parser.add_argument('--grid-step', type=int, default=8, help='Grid sampling step for motion estimation (default: 8)')
parser.add_argument('--motion-threshold', type=float, default=0.5, help='Minimum motion magnitude to consider (default: 0.5)')
parser.add_argument('--min-points', type=int, default=10, help='Minimum points required for robust estimation (default: 10)')
parser.add_argument('--ransac-threshold', type=float, default=2.0, help='RANSAC reprojection threshold (default: 2.0)')

# Smoothing Parameters
parser.add_argument('--smoothing-alpha', type=float, default=0.5, help='Motion smoothing factor (0-1, default: 0.5)')

args = parser.parse_args()

# Only create output directory if saving videos
if args.save_videos:
    os.makedirs(args.output_dir, exist_ok=True)

# Initialize reference data storage if --save-reference is specified
reference_data = []

# Decimation factor - reduce image size by this factor for better performance
DECIMATION_FACTOR = args.decimation

# Open video file
video_path = args.video if args.video else "/app/videos/input.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Get original video properties
original_fps = cap.get(cv.CAP_PROP_FPS)
total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
print(f"Using video file: {video_path}")
print(f"Original video FPS: {original_fps:.2f}")
print(f"Total frames: {int(total_frames)}")
print(f"Processing up to {args.max_frames} frames")
if args.save_videos:
    print(f"Video output enabled - saving to {args.output_dir}")
else:
    print("Video output disabled - showing motion analysis only")

ret, frame1 = cap.read()
original_height, original_width = frame1.shape[:2]
center = (original_width // 2, original_height // 2)

# Set up video writers only if saving videos
optical_flow_writer = None
original_writer = None
corrected_writer = None

if args.save_videos:
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    optical_flow_writer = cv.VideoWriter(
        os.path.join(args.output_dir, 'dense_optical_flow.mp4'), 
        fourcc, original_fps, (original_width//DECIMATION_FACTOR, original_height//DECIMATION_FACTOR)
    )
    original_writer = cv.VideoWriter(
        os.path.join(args.output_dir, 'original_with_motion.mp4'), 
        fourcc, original_fps, (original_width, original_height)
    )
    corrected_writer = cv.VideoWriter(
        os.path.join(args.output_dir, 'rotation_corrected.mp4'), 
        fourcc, original_fps, (original_width, original_height)
    )

# Decimate the first frame
frame1_small = cv.resize(frame1, None, fx=1/DECIMATION_FACTOR, fy=1/DECIMATION_FACTOR, interpolation=cv.INTER_NEAREST)
prvs = cv.cvtColor(frame1_small, cv.COLOR_BGR2GRAY)

# Pre-allocate arrays to avoid memory allocation overhead
h, w = prvs.shape
flow = np.zeros((h, w, 2), dtype=np.float32)
mag = np.zeros((h, w), dtype=np.float32)
ang = np.zeros((h, w), dtype=np.float32)

# Pre-allocate HSV array for visualization (only if saving videos)
if args.save_videos:
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

# Create grid of points for motion estimation
step = args.grid_step  # Use configurable grid step
y_coords, x_coords = np.mgrid[step//2:h-step//2:step, step//2:w-step//2:step]
points1 = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).astype(np.float32)

# Variables for FPS calculation
frame_count = 0
processed_frames = 0
start_time = time.time()
fps_display = 0

# Motion tracking variables
pan_x_smooth = 0
pan_y_smooth = 0
rotation_smooth = 0
alpha = args.smoothing_alpha  # Use configurable smoothing factor

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
            if np.sqrt(fx*fx + fy*fy) > args.motion_threshold:
                source_points.append([x, y])
                flow_vectors.append([x + fx, y + fy])
    
    if len(source_points) < args.min_points:  # Need minimum points for robust estimation
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
                                                    ransacReprojThreshold=args.ransac_threshold)
        if transform is not None:
            # Extract rotation angle
            rotation_rad = math.atan2(transform[1, 0], transform[0, 0])
            rotation_deg = math.degrees(rotation_rad)
            
            # Extract translation - correct for rotation around center vs origin
            tx_raw = transform[0, 2]
            ty_raw = transform[1, 2]
            
            cos_r = math.cos(rotation_rad)
            sin_r = math.sin(rotation_rad)
            
            rotation_translation_x = center_x * (cos_r - 1) + center_y * sin_r
            rotation_translation_y = -center_x * sin_r + center_y * (cos_r - 1)
            
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

print("Processing frames...")
print("Frame | Pan X | Pan Y | Rotation | Acc.Rot | FPS")
print("------|-------|-------|----------|---------|----")

while processed_frames < args.max_frames:
    loop_start = time.time()
    
    ret, frame2 = cap.read()
    if not ret:
        print(f'\nFinished processing video at frame {frame_count}')
        break
    
    frame_count += 1
    
    # Decimate the current frame
    frame2_small = cv.resize(frame2, None, fx=1/DECIMATION_FACTOR, fy=1/DECIMATION_FACTOR, interpolation=cv.INTER_NEAREST)
    next = cv.cvtColor(frame2_small, cv.COLOR_BGR2GRAY)
    
    # Configurable Farneback parameters for optical flow:
    cv.calcOpticalFlowFarneback(prvs, next, flow, 
                               args.pyr_scale, args.levels, args.winsize, 
                               args.iterations, args.poly_n, args.poly_sigma, 0)
    
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
    
    # Store reference data if --save-reference is specified
    if args.save_reference:
        reference_data.append([
            processed_frames,
            pan_x_smooth,
            pan_y_smooth,
            rotation_smooth,
            accumulated_rotation,
            accumulated_pan_x,
            accumulated_pan_y
        ])
    
    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps_display = processed_frames / elapsed_time if elapsed_time > 0 else 0
    
    # Print motion data (every 10 frames for readability)
    if processed_frames % 10 == 0 or not args.save_videos:
        print(f"{processed_frames:5d} | {pan_x_smooth:5.1f} | {pan_y_smooth:5.1f} | {rotation_smooth:8.2f} | {accumulated_rotation:7.1f} | {fps_display:3.1f}")
    
    # Only process videos if saving is enabled
    if args.save_videos:
        # Apply rotation correction using accumulated rotation
        frame2_unrotated = apply_rotation_correction(frame2, accumulated_rotation, center)
        
        # Create optical flow visualization
        cv.cartToPolar(flow[..., 0], flow[..., 1], mag, ang)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        
        # Add motion information to the optical flow image
        cv.putText(bgr, f'Frame: {processed_frames}/{args.max_frames}', (10, 25), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(bgr, f'Pan X: {pan_x_smooth:.1f}px', (10, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(bgr, f'Pan Y: {pan_y_smooth:.1f}px', (10, 75), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(bgr, f'Rotation: {rotation_smooth:.2f}°', (10, 100), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(bgr, f'Size: {w}x{h}', (10, 125), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create copies for drawing motion vectors
        frame2_with_motion = frame2.copy()
        frame2_unrotated_with_motion = frame2_unrotated.copy()
        
        # Draw motion vector on both frames
        frame2_with_motion = draw_motion_vector(frame2_with_motion, pan_x_smooth, pan_y_smooth, rotation_smooth)
        frame2_unrotated_with_motion = draw_motion_vector(frame2_unrotated_with_motion, pan_x_smooth, pan_y_smooth, rotation_smooth)
        
        # Add text to frames
        cv.putText(frame2_with_motion, 'Original + Motion Vector', (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.putText(frame2_unrotated_with_motion, f'Rotation Corrected ({-accumulated_rotation:.2f}°)', (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Write frames to output videos
        optical_flow_writer.write(bgr)
        original_writer.write(frame2_with_motion)
        corrected_writer.write(frame2_unrotated_with_motion)
    
    processed_frames += 1
    prvs = next

# Clean up
cap.release()
if args.save_videos:
    optical_flow_writer.release()
    original_writer.release()
    corrected_writer.release()

# Final statistics
total_time = time.time() - start_time
final_fps = processed_frames / total_time
print(f"\nProcessing complete!")
print(f"Processed {processed_frames} frames in {total_time:.2f} seconds")
print(f"Average processing FPS: {final_fps:.2f}")
print(f"Speedup vs original video: {final_fps/original_fps:.2f}x")
print(f"Processing resolution: {w}x{h} pixels")
print(f"Final accumulated rotation: {accumulated_rotation:.2f}°")
print(f"Final accumulated pan: ({accumulated_pan_x:.2f}, {accumulated_pan_y:.2f})")

if args.save_videos:
    print(f"\nOutput files saved to {args.output_dir}:")
    print(f"  - dense_optical_flow.mp4 (optical flow visualization)")
    print(f"  - original_with_motion.mp4 (original video with motion vectors)")
    print(f"  - rotation_corrected.mp4 (rotation-stabilized video)")
else:
    print(f"\nTo save video outputs, use --save-videos flag")

# Save reference data if --save-reference is specified
if args.save_reference:
    with open(args.save_reference, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write CSV header
        csvwriter.writerow([
            'frame',
            'pan_x',
            'pan_y', 
            'rotation_deg',
            'accumulated_rotation_deg',
            'accumulated_pan_x',
            'accumulated_pan_y'
        ])
        # Write all collected data
        csvwriter.writerows(reference_data)
    print(f"\nReference data saved to {args.save_reference}")
    print(f"Saved {len(reference_data)} frames of pan/tilt data") 