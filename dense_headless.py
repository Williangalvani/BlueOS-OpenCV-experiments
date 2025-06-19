import numpy as np
import cv2 as cv
import time
import math
import argparse
import os
import csv
import cProfile
import pstats
from collections import defaultdict

# Set up command line arguments
parser = argparse.ArgumentParser(description='Dense Optical Flow with global motion estimation (headless version)')
parser.add_argument('--video', type=str, help='Path to video file (default: use video.mp4)')
parser.add_argument('--output-dir', type=str, default='/app/output', help='Output directory for videos')
parser.add_argument('--max-frames', type=int, default=300, help='Maximum number of frames to process (default: 300)')
parser.add_argument('--decimation', type=int, default=5, help='Decimation factor for performance (default: 5)')
parser.add_argument('--save-videos', action='store_true', help='Save output videos (default: only print motion data)')
parser.add_argument('--save-reference', type=str, help='Save pan/tilt reference data to specified file (CSV format)')

# Profiling arguments
parser.add_argument('--profile-method', choices=['manual', 'cprofile'], default='manual', help='Profiling method (default: manual)')
parser.add_argument('--profile-output', type=str, help='Save cProfile output to file')
parser.add_argument('--enable-profiling', action='store_true', help='Enable performance profiling (default: disabled)')

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

class PerformanceProfiler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
        self.current_timers = {}
        
    def start_timer(self, operation):
        if not self.enabled:
            return
        self.current_timers[operation] = time.perf_counter()
        
    def end_timer(self, operation):
        if not self.enabled:
            return 0
        if operation in self.current_timers:
            elapsed = time.perf_counter() - self.current_timers[operation]
            self.timings[operation].append(elapsed)
            self.counters[operation] += 1
            del self.current_timers[operation]
            return elapsed
        return 0
    
    def get_stats(self):
        if not self.enabled:
            return {}
        stats = {}
        total_time = sum(sum(times) for times in self.timings.values())
        
        for operation, times in self.timings.items():
            if times:
                stats[operation] = {
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'count': len(times),
                    'percentage': (sum(times) / total_time * 100) if total_time > 0 else 0
                }
        return stats
    
    def print_report(self):
        if not self.enabled:
            return
            
        stats = self.get_stats()
        total_time = sum(stat['total_time'] for stat in stats.values())
        
        print(f"\n{'='*80}")
        print(f"PERFORMANCE PROFILING REPORT")
        print(f"{'='*80}")
        print(f"Total Processing Time: {total_time:.3f}s")
        print(f"{'='*80}")
        print(f"{'Operation':<25} {'Total(s)':<10} {'Avg(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10} {'Count':<8} {'%':<8}")
        print(f"{'-'*80}")
        
        # Sort by total time (descending)
        for operation, stat in sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True):
            print(f"{operation:<25} {stat['total_time']:<10.3f} {stat['avg_time']*1000:<10.2f} "
                  f"{stat['min_time']*1000:<10.2f} {stat['max_time']*1000:<10.2f} "
                  f"{stat['count']:<8} {stat['percentage']:<8.1f}")
        
        print(f"{'='*80}")

# Initialize profiler
profiler = PerformanceProfiler(enabled=args.enable_profiling)

# Only create output directory if saving videos
if args.save_videos:
    os.makedirs(args.output_dir, exist_ok=True)

# Initialize reference data storage if --save-reference is specified
reference_data = []

# Decimation factor - reduce image size by this factor for better performance
DECIMATION_FACTOR = args.decimation

# Open video file
profiler.start_timer('video_setup')
video_path = args.video if args.video else "/app/videos/input.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Get original video properties
original_fps = cap.get(cv.CAP_PROP_FPS)
total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
profiler.end_timer('video_setup')

print(f"Using video file: {video_path}")
print(f"Original video FPS: {original_fps:.2f}")
print(f"Total frames: {int(total_frames)}")
print(f"Processing up to {args.max_frames} frames")
if args.enable_profiling:
    print(f"Performance profiling: ENABLED ({args.profile_method})")
if args.save_videos:
    print(f"Video output enabled - saving to {args.output_dir}")
else:
    print("Video output disabled - showing motion analysis only")

profiler.start_timer('frame_read_initial')
ret, frame1 = cap.read()
profiler.end_timer('frame_read_initial')

original_height, original_width = frame1.shape[:2]
center = (original_width // 2, original_height // 2)

# Set up video writers only if saving videos
optical_flow_writer = None
original_writer = None
corrected_writer = None

if args.save_videos:
    profiler.start_timer('video_writers_setup')
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
    profiler.end_timer('video_writers_setup')

# Decimate the first frame
profiler.start_timer('frame_decimation')
frame1_small = cv.resize(frame1, None, fx=1/DECIMATION_FACTOR, fy=1/DECIMATION_FACTOR, interpolation=cv.INTER_NEAREST)
prvs = cv.cvtColor(frame1_small, cv.COLOR_BGR2GRAY)
profiler.end_timer('frame_decimation')

# Pre-allocate arrays to avoid memory allocation overhead
h, w = prvs.shape
profiler.start_timer('array_allocation')
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
profiler.end_timer('array_allocation')

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
    profiler.start_timer('motion_sampling')
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
    profiler.end_timer('motion_sampling')
    
    if len(source_points) < args.min_points:  # Need minimum points for robust estimation
        return 0, 0, 0
    
    profiler.start_timer('motion_estimation')
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
            
            profiler.end_timer('motion_estimation')
            return pure_pan_x, pure_pan_y, rotation_deg
    except:
        pass
    
    profiler.end_timer('motion_estimation')
    return 0, 0, 0

def apply_rotation_correction(image, rotation_angle, center):
    """Apply inverse rotation to stabilize the image"""
    if abs(rotation_angle) < 0.1:  # Skip tiny rotations
        return image
    
    profiler.start_timer('rotation_correction')
    # Create rotation matrix for inverse rotation
    rotation_matrix = cv.getRotationMatrix2D(center, 0, 1.0)
    
    # Apply rotation
    corrected = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    profiler.end_timer('rotation_correction')
    return corrected

def draw_motion_vector(image, pan_x, pan_y, rotation_deg):
    """Draw motion vector arrow showing pan direction and rotation indicator"""
    profiler.start_timer('motion_vector_drawing')
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
    
    profiler.end_timer('motion_vector_drawing')
    return image

print("Processing frames...")
if args.enable_profiling:
    print("Frame | Pan X | Pan Y | Rotation | Acc.Rot | FPS | Frame Time(ms)")
    print("------|-------|-------|----------|---------|-----|---------------")
else:
    print("Frame | Pan X | Pan Y | Rotation | Acc.Rot | FPS")
    print("------|-------|-------|----------|---------|----")

# Start cProfile if requested
if args.enable_profiling and args.profile_method == 'cprofile':
    pr = cProfile.Profile()
    pr.enable()

processing_start_time = time.perf_counter()

while processed_frames < args.max_frames:
    frame_start_time = time.perf_counter()
    
    profiler.start_timer('frame_read')
    ret, frame2 = cap.read()
    profiler.end_timer('frame_read')
    
    if not ret:
        print(f'\nFinished processing video at frame {frame_count}')
        break
    
    frame_count += 1
    
    # Decimate the current frame
    profiler.start_timer('frame_decimation')
    frame2_small = cv.resize(frame2, None, fx=1/DECIMATION_FACTOR, fy=1/DECIMATION_FACTOR, interpolation=cv.INTER_NEAREST)
    next = cv.cvtColor(frame2_small, cv.COLOR_BGR2GRAY)
    profiler.end_timer('frame_decimation')
    
    # Configurable Farneback parameters for optical flow:
    profiler.start_timer('optical_flow_calculation')
    cv.calcOpticalFlowFarneback(prvs, next, flow, 
                               args.pyr_scale, args.levels, args.winsize, 
                               args.iterations, args.poly_n, args.poly_sigma, 0)
    profiler.end_timer('optical_flow_calculation')
    
    # Extract global motion parameters
    profiler.start_timer('global_motion_extraction')
    pan_x, pan_y, rotation = extract_global_motion(flow, points1)
    profiler.end_timer('global_motion_extraction')
    
    # Apply smoothing to reduce noise
    profiler.start_timer('motion_smoothing')
    pan_x_smooth = alpha * pan_x + (1 - alpha) * pan_x_smooth
    pan_y_smooth = alpha * pan_y + (1 - alpha) * pan_y_smooth  
    rotation_smooth = alpha * rotation + (1 - alpha) * rotation_smooth
    
    # Accumulate motion deltas for stabilization
    accumulated_rotation -= rotation_smooth
    accumulated_pan_x += pan_x_smooth
    accumulated_pan_y += pan_y_smooth
    profiler.end_timer('motion_smoothing')
    
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
    
    # Calculate FPS and frame time
    elapsed_time = time.time() - start_time
    fps_display = processed_frames / elapsed_time if elapsed_time > 0 else 0
    frame_time = (time.perf_counter() - frame_start_time) * 1000
    
    # Print motion data (every 10 frames for readability)
    if processed_frames % 10 == 0 or not args.save_videos:
        if args.enable_profiling:
            print(f"{processed_frames:5d} | {pan_x_smooth:5.1f} | {pan_y_smooth:5.1f} | {rotation_smooth:8.2f} | {accumulated_rotation:7.1f} | {fps_display:3.1f} | {frame_time:13.2f}")
        else:
            print(f"{processed_frames:5d} | {pan_x_smooth:5.1f} | {pan_y_smooth:5.1f} | {rotation_smooth:8.2f} | {accumulated_rotation:7.1f} | {fps_display:3.1f}")
    
    # Only process videos if saving is enabled
    if args.save_videos:
        profiler.start_timer('video_processing')
        
        # Apply rotation correction using accumulated rotation
        frame2_unrotated = apply_rotation_correction(frame2, accumulated_rotation, center)
        
        # Create optical flow visualization
        profiler.start_timer('flow_visualization')
        cv.cartToPolar(flow[..., 0], flow[..., 1], mag, ang)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        profiler.end_timer('flow_visualization')
        
        # Add motion information to the optical flow image
        profiler.start_timer('text_overlay')
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
        profiler.end_timer('text_overlay')
        
        # Create copies for drawing motion vectors
        profiler.start_timer('frame_copying')
        frame2_with_motion = frame2.copy()
        frame2_unrotated_with_motion = frame2_unrotated.copy()
        profiler.end_timer('frame_copying')
        
        # Draw motion vector on both frames
        frame2_with_motion = draw_motion_vector(frame2_with_motion, pan_x_smooth, pan_y_smooth, rotation_smooth)
        frame2_unrotated_with_motion = draw_motion_vector(frame2_unrotated_with_motion, pan_x_smooth, pan_y_smooth, rotation_smooth)
        
        # Add text to frames
        profiler.start_timer('text_overlay')
        cv.putText(frame2_with_motion, 'Original + Motion Vector', (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.putText(frame2_unrotated_with_motion, f'Rotation Corrected ({-accumulated_rotation:.2f}°)', (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        profiler.end_timer('text_overlay')
        
        # Write frames to output videos
        profiler.start_timer('video_writing')
        optical_flow_writer.write(bgr)
        original_writer.write(frame2_with_motion)
        corrected_writer.write(frame2_unrotated_with_motion)
        profiler.end_timer('video_writing')
        
        profiler.end_timer('video_processing')
    
    processed_frames += 1
    prvs = next

processing_end_time = time.perf_counter()

# Stop cProfile if used
if args.enable_profiling and args.profile_method == 'cprofile':
    pr.disable()

# Clean up
cap.release()
if args.save_videos:
    optical_flow_writer.release()
    original_writer.release()
    corrected_writer.release()

# Final statistics
total_time = time.time() - start_time
final_fps = processed_frames / total_time
actual_processing_time = processing_end_time - processing_start_time

print(f"\nProcessing complete!")
print(f"Processed {processed_frames} frames in {total_time:.2f} seconds")
if args.enable_profiling:
    print(f"Pure processing time: {actual_processing_time:.2f} seconds")
print(f"Average processing FPS: {final_fps:.2f}")
print(f"Target FPS for real-time: {original_fps:.2f}")
if args.enable_profiling:
    print(f"Real-time performance: {'✅ YES' if final_fps >= original_fps else '❌ NO'}")
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

# Print performance profiling report
if args.enable_profiling:
    profiler.print_report()
    
    # Handle cProfile output
    if args.profile_method == 'cprofile':
        if args.profile_output:
            pr.dump_stats(args.profile_output)
            print(f"\ncProfile data saved to {args.profile_output}")
            print(f"View with: python -m pstats {args.profile_output}")
        else:
            print(f"\nTop 20 functions by cumulative time:")
            stats = pstats.Stats(pr)
            stats.sort_stats('cumulative')
            stats.print_stats(20)

    # Optimization recommendations
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*80}")

    stats = profiler.get_stats()
    if 'optical_flow_calculation' in stats:
        flow_time = stats['optical_flow_calculation']['avg_time'] * 1000
        print(f"1. Optical Flow Calculation: {flow_time:.1f}ms/frame")
        if flow_time > 20:  # > 20ms means can't reach 50fps
            print(f"   - Consider increasing decimation factor (current: {DECIMATION_FACTOR})")
            print(f"   - Try Farneback parameter tuning (pyramid levels, iterations)")

    if 'global_motion_extraction' in stats:
        motion_time = stats['global_motion_extraction']['avg_time'] * 1000
        print(f"2. Motion Estimation: {motion_time:.1f}ms/frame")
        if motion_time > 10:
            print(f"   - Reduce grid step size or use sparse sampling")
            print(f"   - Consider simpler motion estimation algorithms")

    target_frame_time = 1000 / original_fps
    avg_frame_time = actual_processing_time / processed_frames * 1000
    print(f"3. Target frame time for real-time: {target_frame_time:.1f}ms")
    print(f"   Current average frame time: {avg_frame_time:.1f}ms")

    if avg_frame_time > target_frame_time:
        print(f"   - Need {target_frame_time/avg_frame_time:.1f}x speedup for real-time")
    else:
        print(f"   - ✅ Already achieving real-time performance!")

    print(f"{'='*80}") 