#!/usr/bin/env python3
"""
Optimization script for dense_headless.py parameters
Optimizes for both processing speed and accuracy against reference motion data
"""

import subprocess
import time
import csv
import numpy as np
import pandas as pd
import argparse
import os
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings("ignore")

@dataclass
class OptimizationResult:
    """Results from a single optimization run"""
    params: Dict[str, float]
    processing_time: float
    accuracy_error: float
    combined_score: float
    success: bool
    output_file: str

def run_single_evaluation(args_tuple):
    """Standalone function for multiprocessing - runs a single parameter evaluation"""
    params, video_path, max_frames, output_dir, iteration, reference_data_dict, time_weight, accuracy_weight, error_tolerance = args_tuple
    
    # Reconstruct reference data
    reference_data = pd.DataFrame(reference_data_dict)
    
    output_csv = f"{output_dir}/test_run_{iteration:04d}.csv"
    
    # Build command line arguments
    cmd = [
        'python', 'dense_headless.py',
        '--video', video_path,
        '--max-frames', str(max_frames),
        '--save-reference', output_csv,
        '--decimation', str(int(params['decimation'])),
        '--pyr-scale', str(params['pyr_scale']),
        '--levels', str(int(params['levels'])),
        '--winsize', str(int(params['winsize'])),
        '--iterations', str(int(params['iterations'])),
        '--poly-n', str(int(params['poly_n'])),
        '--poly-sigma', str(params['poly_sigma']),
        '--grid-step', str(int(params['grid_step'])),
        '--motion-threshold', str(params['motion_threshold']),
        '--min-points', str(int(params['min_points'])),
        '--ransac-threshold', str(params['ransac_threshold']),
        '--smoothing-alpha', str(params['smoothing_alpha'])
    ]
    
    # Measure execution time
    start_time = time.time()
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        processing_time = time.time() - start_time
        
        if result.returncode != 0:
            return OptimizationResult(
                params=params,
                processing_time=999.0,
                accuracy_error=999.0,
                combined_score=999.0,
                success=False,
                output_file=""
            )
        
        # Calculate accuracy error
        accuracy_error = calculate_accuracy_error_standalone(output_csv, reference_data)
        
        # Calculate combined score with tolerance-aware penalty
        normalized_time = processing_time / 60.0
        
        # Apply tolerance-aware penalty for accuracy error
        if accuracy_error <= error_tolerance:
            # Light penalty for errors within tolerance (reduce penalty by 70%)
            tolerance_penalty_factor = 0.3
            penalized_error = accuracy_error * tolerance_penalty_factor
        else:
            # Full penalty for errors exceeding tolerance, plus additional penalty for excess
            excess_error = accuracy_error - error_tolerance
            penalized_error = error_tolerance * 0.3 + excess_error * 1.5  # Extra penalty for excess
        
        combined_score = (time_weight * normalized_time + accuracy_weight * penalized_error)
        
        return OptimizationResult(
            params=params,
            processing_time=processing_time,
            accuracy_error=accuracy_error,
            combined_score=combined_score,
            success=True,
            output_file=output_csv
        )
        
    except subprocess.TimeoutExpired:
        return OptimizationResult(
            params=params,
            processing_time=999.0,
            accuracy_error=999.0,
            combined_score=999.0,
            success=False,
            output_file=""
        )
    except Exception as e:
        return OptimizationResult(
            params=params,
            processing_time=999.0,
            accuracy_error=999.0,
            combined_score=999.0,
            success=False,
            output_file=""
        )

def calculate_accuracy_error_standalone(output_csv: str, reference_data: pd.DataFrame) -> float:
    """Standalone accuracy calculation for multiprocessing"""
    if not os.path.exists(output_csv):
        return 999.0
    
    try:
        generated_data = pd.read_csv(output_csv)
        min_frames = min(len(reference_data), len(generated_data))
        if min_frames == 0:
            return 999.0
        
        ref_subset = reference_data.iloc[:min_frames]
        gen_subset = generated_data.iloc[:min_frames]
        
        pan_x_error = np.sqrt(np.mean((ref_subset['pan_x'] - gen_subset['pan_x'])**2))
        pan_y_error = np.sqrt(np.mean((ref_subset['pan_y'] - gen_subset['pan_y'])**2))
        rotation_error = np.sqrt(np.mean((ref_subset['rotation_deg'] - gen_subset['rotation_deg'])**2))
        
        combined_error = (pan_x_error + pan_y_error + rotation_error * 0.1) / 2.1
        return combined_error
        
    except Exception as e:
        return 999.0

class DenseOptimizer:
    """Optimizer for dense optical flow parameters"""
    
    def __init__(self, 
                 video_path: str,
                 reference_csv: str,
                 max_frames: int = 100,
                 output_dir: str = "./optimization_results",
                 num_workers: int = None,
                 error_tolerance: float = 0.1):
        
        self.video_path = video_path
        self.reference_csv = reference_csv
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.results = []
        self.iteration = 0
        self.best_result = None  # Track best result so far
        self.num_workers = num_workers or min(mp.cpu_count(), 8)  # Limit to 8 cores max by default
        self.error_tolerance = error_tolerance  # Error tolerance as percentage (0.1 = 10%)
        
        print(f"Using {self.num_workers} parallel workers")
        print(f"Error tolerance: {error_tolerance*100:.1f}% (errors within this range are lightly penalized)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load reference data
        self.reference_data = pd.read_csv(reference_csv)
        print(f"Loaded reference data with {len(self.reference_data)} frames")
        
        # Parameter bounds for optimization
        self.param_bounds = {
            'decimation': (2, 10),          # Decimation factor
            'pyr_scale': (0.3, 0.8),        # Pyramid scale
            'levels': (1, 4),               # Pyramid levels
            'winsize': (5, 20),             # Window size
            'iterations': (1, 5),           # Iterations
            'poly_n': (3, 7),               # Polynomial neighborhood
            'poly_sigma': (0.8, 2.0),       # Polynomial sigma
            'grid_step': (4, 20),           # Grid step
            'motion_threshold': (0.1, 2.0), # Motion threshold
            'min_points': (5, 30),          # Minimum points
            'ransac_threshold': (0.5, 5.0), # RANSAC threshold
            'smoothing_alpha': (0.1, 0.9)   # Smoothing factor
        }
        
        # Weight factors for multi-objective optimization
        self.time_weight = 0.6  # Weight for processing time (higher = prioritize speed)
        self.accuracy_weight = 0.4  # Weight for accuracy (higher = prioritize accuracy)
    
    def calculate_tolerance_aware_score(self, processing_time: float, accuracy_error: float) -> float:
        """Calculate combined score with tolerance-aware error penalty"""
        normalized_time = processing_time / 60.0  # Normalize to minutes
        
        # Apply tolerance-aware penalty for accuracy error
        if accuracy_error <= self.error_tolerance:
            # Light penalty for errors within tolerance (reduce penalty by 70%)
            tolerance_penalty_factor = 0.3
            penalized_error = accuracy_error * tolerance_penalty_factor
            print(f"    ✓ Within tolerance ({accuracy_error:.4f} <= {self.error_tolerance:.4f}), "
                  f"applying light penalty: {penalized_error:.4f}")
        else:
            # Full penalty for errors exceeding tolerance, plus additional penalty for excess
            excess_error = accuracy_error - self.error_tolerance
            penalized_error = self.error_tolerance * 0.3 + excess_error * 1.5  # Extra penalty for excess
            print(f"    ⚠ Exceeds tolerance ({accuracy_error:.4f} > {self.error_tolerance:.4f}), "
                  f"applying full penalty: {penalized_error:.4f}")
        
        combined_score = (self.time_weight * normalized_time + self.accuracy_weight * penalized_error)
        return combined_score
    
    def run_dense_headless(self, params: Dict[str, float]) -> OptimizationResult:
        """Run dense_headless.py with given parameters and measure performance"""
        
        self.iteration += 1
        output_csv = f"{self.output_dir}/test_run_{self.iteration:04d}.csv"
        
        # Build command line arguments
        cmd = [
            'python', 'dense_headless.py',
            '--video', self.video_path,
            '--max-frames', str(self.max_frames),
            '--save-reference', output_csv,
            '--decimation', str(int(params['decimation'])),
            '--pyr-scale', str(params['pyr_scale']),
            '--levels', str(int(params['levels'])),
            '--winsize', str(int(params['winsize'])),
            '--iterations', str(int(params['iterations'])),
            '--poly-n', str(int(params['poly_n'])),
            '--poly-sigma', str(params['poly_sigma']),
            '--grid-step', str(int(params['grid_step'])),
            '--motion-threshold', str(params['motion_threshold']),
            '--min-points', str(int(params['min_points'])),
            '--ransac-threshold', str(params['ransac_threshold']),
            '--smoothing-alpha', str(params['smoothing_alpha'])
        ]
        
        print(f"\nIteration {self.iteration}: Testing parameters...")
        print(f"  Decimation: {int(params['decimation'])}, Levels: {int(params['levels'])}, "
              f"Winsize: {int(params['winsize'])}, Grid: {int(params['grid_step'])}")
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            processing_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"  ERROR: Command failed with return code {result.returncode}")
                print(f"  STDERR: {result.stderr}")
                return OptimizationResult(
                    params=params,
                    processing_time=999.0,  # Penalty for failure
                    accuracy_error=999.0,   # Penalty for failure
                    combined_score=999.0,
                    success=False,
                    output_file=""
                )
            
            # Calculate accuracy error
            accuracy_error = self.calculate_accuracy_error(output_csv)
            
            # Calculate combined score (lower is better)
            combined_score = self.calculate_tolerance_aware_score(processing_time, accuracy_error)
            
            print(f"  Time: {processing_time:.2f}s, Error: {accuracy_error:.4f}, Score: {combined_score:.4f}")
            
            result = OptimizationResult(
                params=params,
                processing_time=processing_time,
                accuracy_error=accuracy_error,
                combined_score=combined_score,
                success=True,
                output_file=output_csv
            )
            
            # Update best result if this is better
            if self.best_result is None or (result.success and result.combined_score < self.best_result.combined_score):
                self.best_result = result
                print(f"  ⭐ NEW BEST! Score: {result.combined_score:.4f}")
            
            # Show current best
            if self.best_result:
                print(f"  📊 Current Best - Score: {self.best_result.combined_score:.4f}, "
                      f"Time: {self.best_result.processing_time:.2f}s, Error: {self.best_result.accuracy_error:.4f}")
                print(f"     Best Params: Dec={int(self.best_result.params['decimation'])}, "
                      f"Lev={int(self.best_result.params['levels'])}, "
                      f"Win={int(self.best_result.params['winsize'])}, "
                      f"Grid={int(self.best_result.params['grid_step'])}")
            
            return result
            
        except subprocess.TimeoutExpired:
            print("  ERROR: Command timed out after 5 minutes")
            return OptimizationResult(
                params=params,
                processing_time=999.0,
                accuracy_error=999.0,
                combined_score=999.0,
                success=False,
                output_file=""
            )
        except Exception as e:
            print(f"  ERROR: Exception occurred: {e}")
            return OptimizationResult(
                params=params,
                processing_time=999.0,
                accuracy_error=999.0,
                combined_score=999.0,
                success=False,
                output_file=""
            )
    
    def calculate_accuracy_error(self, output_csv: str) -> float:
        """Calculate error between output and reference motion data"""
        
        if not os.path.exists(output_csv):
            return 999.0
        
        try:
            # Load generated data
            generated_data = pd.read_csv(output_csv)
            
            # Ensure we have the same number of frames (up to the minimum)
            min_frames = min(len(self.reference_data), len(generated_data))
            if min_frames == 0:
                return 999.0
            
            ref_subset = self.reference_data.iloc[:min_frames]
            gen_subset = generated_data.iloc[:min_frames]
            
            # Calculate RMSE for each motion component
            pan_x_error = np.sqrt(np.mean((ref_subset['pan_x'] - gen_subset['pan_x'])**2))
            pan_y_error = np.sqrt(np.mean((ref_subset['pan_y'] - gen_subset['pan_y'])**2))
            rotation_error = np.sqrt(np.mean((ref_subset['rotation_deg'] - gen_subset['rotation_deg'])**2))
            
            # Combined error (weighted average)
            combined_error = (pan_x_error + pan_y_error + rotation_error * 0.1) / 2.1
            
            return combined_error
            
        except Exception as e:
            print(f"  ERROR calculating accuracy: {e}")
            return 999.0
    
    def objective_function(self, x: List[float]) -> float:
        """Objective function for optimization (to be minimized)"""
        
        # Convert parameter vector to dictionary
        param_names = list(self.param_bounds.keys())
        params = {name: x[i] for i, name in enumerate(param_names)}
        
        # Run test and get results
        result = self.run_dense_headless(params)
        self.results.append(result)
        
        return result.combined_score
    
    def optimize_parallel_random(self, max_iterations: int = 50) -> Dict:
        """Run optimization using parallel random search"""
        
        print(f"\nStarting Parallel Random Search optimization...")
        print(f"Max iterations: {max_iterations}")
        print(f"Parallel workers: {self.num_workers}")
        print(f"Time weight: {self.time_weight}, Accuracy weight: {self.accuracy_weight}")
        
        best_score = float('inf')
        best_params = None
        
        # Generate random parameter sets in batches
        batch_size = self.num_workers
        total_batches = (max_iterations + batch_size - 1) // batch_size
        
        # Convert reference data to dict for multiprocessing
        reference_data_dict = self.reference_data.to_dict('records')
        
        iteration = 0
        for batch in range(total_batches):
            print(f"\n--- Batch {batch + 1}/{total_batches} ---")
            
            # Generate random parameter sets for this batch
            param_sets = []
            args_list = []
            
            for i in range(min(batch_size, max_iterations - iteration)):
                iteration += 1
                
                # Generate random parameters within bounds
                params = {}
                for param_name, (min_val, max_val) in self.param_bounds.items():
                    if param_name in ['decimation', 'levels', 'winsize', 'iterations', 'poly_n', 'grid_step', 'min_points']:
                        params[param_name] = np.random.randint(int(min_val), int(max_val) + 1)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)
                
                param_sets.append(params)
                args_list.append((
                    params,
                    self.video_path,
                    self.max_frames,
                    self.output_dir,
                    iteration,
                    reference_data_dict,
                    self.time_weight,
                    self.accuracy_weight,
                    self.error_tolerance
                ))
            
            # Run evaluations in parallel
            print(f"Running {len(args_list)} evaluations in parallel...")
            batch_start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all jobs
                future_to_iteration = {
                    executor.submit(run_single_evaluation, args): i + 1 
                    for i, args in enumerate(args_list)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_iteration):
                    eval_iteration = future_to_iteration[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        
                        # Update best result
                        if result.success and result.combined_score < best_score:
                            best_score = result.combined_score
                            best_params = result.params.copy()
                            self.best_result = result
                            print(f"  ⭐ NEW BEST! Iteration {iteration - len(args_list) + eval_iteration}, Score: {result.combined_score:.4f}")
                        
                        print(f"  ✓ Eval {eval_iteration}: Time {result.processing_time:.2f}s, "
                              f"Error {result.accuracy_error:.4f}, Score {result.combined_score:.4f}")
                        
                    except Exception as e:
                        print(f"  ✗ Eval {eval_iteration} failed: {e}")
            
            batch_time = time.time() - batch_start_time
            print(f"Batch completed in {batch_time:.2f}s")
            
            # Show current best after each batch
            if self.best_result:
                print(f"📊 Current Best Overall:")
                print(f"   Score: {self.best_result.combined_score:.4f} "
                      f"(Time: {self.best_result.processing_time:.2f}s, Error: {self.best_result.accuracy_error:.4f})")
                print(f"   Params: Dec={int(self.best_result.params['decimation'])}, "
                      f"Lev={int(self.best_result.params['levels'])}, "
                      f"Win={int(self.best_result.params['winsize'])}, "
                      f"Grid={int(self.best_result.params['grid_step'])}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'iterations': len(self.results),
            'success': best_params is not None
        }
    
    def optimize_differential_evolution(self, max_iterations: int = 50) -> Dict:
        """Run optimization using Differential Evolution"""
        
        print(f"\nStarting Differential Evolution optimization...")
        print(f"Max iterations: {max_iterations}")
        print(f"Time weight: {self.time_weight}, Accuracy weight: {self.accuracy_weight}")
        
        # Convert bounds to list format for scipy
        bounds = [self.param_bounds[name] for name in self.param_bounds.keys()]
        
        # Run optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=max_iterations,
            popsize=8,  # Smaller population for faster convergence
            atol=0.01,
            tol=0.01,
            seed=42
        )
        
        # Convert result back to parameter dictionary
        param_names = list(self.param_bounds.keys())
        best_params = {name: result.x[i] for i, name in enumerate(param_names)}
        
        return {
            'best_params': best_params,
            'best_score': result.fun,
            'iterations': len(self.results),
            'success': result.success
        }
    
    def optimize_grid_search(self, grid_points: int = 3) -> Dict:
        """Run optimization using grid search (for smaller parameter spaces)"""
        
        print(f"\nStarting Grid Search optimization...")
        print(f"Grid points per parameter: {grid_points}")
        
        # Create parameter grid (only for most important parameters to keep it manageable)
        important_params = ['decimation', 'levels', 'winsize', 'grid_step']
        
        best_score = float('inf')
        best_params = None
        
        # Generate grid points for important parameters
        param_grids = {}
        for param in important_params:
            bounds = self.param_bounds[param]
            param_grids[param] = np.linspace(bounds[0], bounds[1], grid_points)
        
        # Keep other parameters at default values
        default_params = {
            'pyr_scale': 0.5,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'motion_threshold': 0.5,
            'min_points': 10,
            'ransac_threshold': 2.0,
            'smoothing_alpha': 0.5
        }
        
        total_combinations = grid_points ** len(important_params)
        print(f"Total combinations: {total_combinations}")
        
        combination = 0
        for decimation in param_grids['decimation']:
            for levels in param_grids['levels']:
                for winsize in param_grids['winsize']:
                    for grid_step in param_grids['grid_step']:
                        combination += 1
                        
                        params = default_params.copy()
                        params.update({
                            'decimation': decimation,
                            'levels': levels,
                            'winsize': winsize,
                            'grid_step': grid_step
                        })
                        
                        print(f"\nGrid search {combination}/{total_combinations}")
                        result = self.run_dense_headless(params)
                        self.results.append(result)
                        
                        if result.success and result.combined_score < best_score:
                            best_score = result.combined_score
                            best_params = params.copy()
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'iterations': len(self.results),
            'success': best_params is not None
        }
    
    def save_results(self, optimization_result: Dict):
        """Save optimization results to files"""
        
        # Save best parameters
        best_params_file = f"{self.output_dir}/best_parameters.json"
        with open(best_params_file, 'w') as f:
            json.dump(optimization_result, f, indent=2)
        
        # Save all results to CSV
        results_file = f"{self.output_dir}/optimization_results.csv"
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            param_names = list(self.param_bounds.keys())
            header = ['iteration', 'success', 'processing_time', 'accuracy_error', 'combined_score'] + param_names
            writer.writerow(header)
            
            # Data
            for i, result in enumerate(self.results):
                row = [
                    i + 1,
                    result.success,
                    result.processing_time,
                    result.accuracy_error,
                    result.combined_score
                ]
                row.extend([result.params.get(name, 0) for name in param_names])
                writer.writerow(row)
        
        print(f"\nResults saved:")
        print(f"  Best parameters: {best_params_file}")
        print(f"  All results: {results_file}")
    
    def print_summary(self, optimization_result: Dict):
        """Print optimization summary"""
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            print("\nNo successful runs!")
            return
        
        print(f"\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total iterations: {len(self.results)}")
        print(f"Successful runs: {len(successful_results)}")
        print(f"Success rate: {len(successful_results)/len(self.results)*100:.1f}%")
        
        if optimization_result['success']:
            print(f"\nBEST PARAMETERS:")
            for param, value in optimization_result['best_params'].items():
                if param in ['decimation', 'levels', 'winsize', 'iterations', 'poly_n', 'grid_step', 'min_points']:
                    print(f"  --{param.replace('_', '-')}: {int(value)}")
                else:
                    print(f"  --{param.replace('_', '-')}: {value:.3f}")
            
            print(f"\nBest combined score: {optimization_result['best_score']:.4f}")
            
            # Find the actual result for best params
            best_result = min(successful_results, key=lambda x: x.combined_score)
            print(f"Processing time: {best_result.processing_time:.2f}s")
            print(f"Accuracy error: {best_result.accuracy_error:.4f}")
        
        # Show baseline comparison if we have results
        if len(successful_results) >= 2:
            times = [r.processing_time for r in successful_results]
            errors = [r.accuracy_error for r in successful_results]
            scores = [r.combined_score for r in successful_results]
            
            print(f"\nPerformance range:")
            print(f"  Processing time: {min(times):.2f}s - {max(times):.2f}s")
            print(f"  Accuracy error: {min(errors):.4f} - {max(errors):.4f}")
            print(f"  Combined score: {min(scores):.4f} - {max(scores):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Optimize dense optical flow parameters')
    parser.add_argument('--video', required=True, help='Path to test video file')
    parser.add_argument('--reference', required=True, help='Path to reference motion_data.csv file')
    parser.add_argument('--method', choices=['differential', 'grid', 'parallel_random'], default='parallel_random', 
                       help='Optimization method (default: parallel_random)')
    parser.add_argument('--max-frames', type=int, default=100, 
                       help='Maximum frames to process per test (default: 100)')
    parser.add_argument('--max-iterations', type=int, default=30, 
                       help='Maximum optimization iterations (default: 30)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--output-dir', default='./optimization_results', 
                       help='Output directory for results')
    parser.add_argument('--time-weight', type=float, default=0.6,
                       help='Weight for processing time vs accuracy (0-1, default: 0.6)')
    parser.add_argument('--error-tolerance', type=float, default=0.1,
                       help='Error tolerance as percentage (0.1 = 10%, default: 0.1)')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = DenseOptimizer(
        video_path=args.video,
        reference_csv=args.reference,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        num_workers=args.workers,
        error_tolerance=args.error_tolerance
    )
    
    # Set weights
    optimizer.time_weight = args.time_weight
    optimizer.accuracy_weight = 1.0 - args.time_weight
    
    # Run optimization
    if args.method == 'differential':
        result = optimizer.optimize_differential_evolution(args.max_iterations)
    elif args.method == 'parallel_random':
        result = optimizer.optimize_parallel_random(args.max_iterations)
    else:
        result = optimizer.optimize_grid_search()
    
    # Save and display results
    optimizer.save_results(result)
    optimizer.print_summary(result)
    
    print(f"\nOptimization complete! Check {args.output_dir} for detailed results.")


if __name__ == "__main__":
    main() 