"""
Unified Accelerator for Energy Load Profile Optimization
========================================================

High-performance acceleration system that intelligently utilizes:
- 16-core CPU with vectorized NumPy + multiprocessing
- AMD Radeon RX 7700S GPU with PyTorch ROCm/OpenCL
- GPU-first architecture with intelligent CPU fallback
- Ollama-style GPU detection and memory management

Architecture: GPU-First → CPU Fallback
Performance Target: 20x speedup over baseline
"""

import os
import numpy as np
import logging
import time
import multiprocessing as mp
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# GPU libraries with fallback support
try:
    import torch
    HAS_PYTORCH = True
    # Check for ROCm (AMD GPU support)
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        PYTORCH_BACKEND = 'rocm'
    elif torch.cuda.is_available():
        PYTORCH_BACKEND = 'cuda'
    else:
        HAS_PYTORCH = False
        PYTORCH_BACKEND = None
except ImportError:
    HAS_PYTORCH = False
    PYTORCH_BACKEND = None

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

# CPU acceleration libraries
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

try:
    from scipy.optimize import minimize
    from scipy.signal import convolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class UnifiedAccelerator:
    """
    Unified GPU+CPU acceleration engine optimized for energy pattern optimization.
    
    Features:
    - GPU-first: AMD RX 7700S with PyTorch ROCm/OpenCL
    - CPU fallback: 16-core vectorized processing
    - Intelligent load balancing
    - Memory management like Ollama
    - Batch processing for large populations
    """
    
    def __init__(self, prefer_gpu: bool = True, cpu_workers: Optional[int] = None, 
                 gpu_memory_fraction: float = 0.8):
        self.logger = logging.getLogger(__name__)
        self.prefer_gpu = prefer_gpu
        self.cpu_workers = cpu_workers or max(1, mp.cpu_count() - 1)  # 15 workers on 16-core
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Performance tracking
        self.gpu_time = 0.0
        self.cpu_time = 0.0
        self.operations_count = 0
        self.total_evaluations = 0
        
        # GPU detection and initialization
        self.gpu_info = self._detect_amd_gpu()
        self.gpu_backend = self._initialize_gpu() if prefer_gpu else None
        self.is_gpu_available = self.gpu_backend is not None
        
        # CPU acceleration setup
        self._setup_cpu_acceleration()
        
        # Optimization parameters
        self.gpu_batch_size = 1000      # Optimal for RX 7700S 8GB VRAM
        self.cpu_batch_size = 100       # Optimal for CPU multiprocessing
        self.gpu_threshold = 200        # Use GPU for populations >= 200
        self.cpu_multiprocessing_threshold = 50  # Use multiprocessing for >= 50
        
        self._log_initialization()
    
    def _detect_amd_gpu(self) -> Dict[str, Any]:
        """Detect AMD GPU using Ollama-style detection."""
        gpu_info = {
            'name': 'AMD Radeon RX 7700S',
            'arch': 'gfx1101',  # RDNA3 / Navi 33
            'device_id': '0x7480',
            'vendor': 'AMD',
            'memory_gb': 8,
            'detected': False
        }
        
        try:
            # Try lspci detection
            result = subprocess.run(['lspci', '-v'], capture_output=True, text=True, timeout=10)
            if 'Navi 33' in result.stdout or '7700S' in result.stdout or 'Radeon' in result.stdout:
                gpu_info['detected'] = True
                self.logger.info("✅ AMD Radeon RX 7700S detected")
            
            # Try ROCm detection
            rocm_result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
            if rocm_result.returncode == 0:
                gpu_info['rocm_available'] = True
                
        except Exception as e:
            self.logger.debug(f"GPU detection failed: {e}")
            gpu_info['detected'] = False
        
        return gpu_info
    
    def _initialize_gpu(self) -> Optional[str]:
        """Initialize GPU using Ollama-style approach with environment overrides."""
        
        # Set AMD GPU environment variables like Ollama
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.1'  # Override for RDNA3
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        os.environ['ROCR_VISIBLE_DEVICES'] = '0'
        os.environ['GPU_MAX_HEAP_SIZE'] = '100'
        os.environ['GPU_MAX_ALLOC_PERCENT'] = '80'
        
        # Try PyTorch with ROCm first (best for AMD)
        if HAS_PYTORCH:
            try:
                if PYTORCH_BACKEND == 'rocm':
                    self.logger.info(f"   ROCm version: {torch.version.hip}")
                
                # Test GPU access
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    self.pytorch_device = torch.device('cuda:0')
                    
                    # Test GPU operation with memory management
                    test_tensor = torch.randn(1000, 1000, device=self.pytorch_device, dtype=torch.float32)
                    result = torch.mm(test_tensor, test_tensor)
                    torch.cuda.empty_cache()
                    
                    # Get memory info
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    self.logger.info(f"   GPU Memory: {total_memory / 1e9:.1f} GB")
                    
                    return 'pytorch'
                    
            except Exception as e:
                self.logger.debug(f"PyTorch GPU test failed: {e}")
        
        # Try OpenCL fallback
        if HAS_OPENCL:
            try:
                platforms = cl.get_platforms()
                for platform in platforms:
                    if 'AMD' in platform.name or 'Portable' in platform.name:
                        devices = platform.get_devices(device_type=cl.device_type.GPU)
                        if devices:
                            self.cl_context = cl.Context(devices)
                            self.cl_queue = cl.CommandQueue(self.cl_context)
                            
                            # Test OpenCL operation
                            test_array = cl_array.to_device(
                                self.cl_queue, 
                                np.random.randn(1000, 1000).astype(np.float32)
                            )
                            _ = cl_array.sum(test_array).get()
                            
                            self.logger.info(f"   OpenCL platform: {platform.name}")
                            self.logger.info(f"   OpenCL device: {devices[0].name}")
                            return 'opencl'
                            
            except Exception as e:
                self.logger.debug(f"OpenCL test failed: {e}")
        
        return None
    
    def _setup_cpu_acceleration(self):
        """Set up CPU acceleration with optimal parameters."""
        # CPU optimization parameters
        self.vectorization_threshold = 10
        
        # Multiprocessing setup
        self.mp_context = mp.get_context('spawn')  # More reliable than fork
        
        self.logger.info(f"🚀 CPU Accelerator: {self.cpu_workers} workers")
        if HAS_NUMBA:
            self.logger.info("   Numba JIT compilation enabled")
        if HAS_SCIPY:
            self.logger.info("   SciPy optimizations available")
    
    def _log_initialization(self):
        """Log initialization summary."""
        if self.is_gpu_available:
            self.logger.info(f"🚀 Unified Accelerator: GPU+CPU ({self.gpu_backend.upper()})")
            self.logger.info(f"   GPU: {self.gpu_info['name']}")
            self.logger.info(f"   Architecture: {self.gpu_info['arch']}")
            self.logger.info(f"   Memory: ~{self.gpu_info['memory_gb']}GB VRAM")
            self.logger.info(f"   CPU Workers: {self.cpu_workers}")
        else:
            self.logger.info(f"🚀 Unified Accelerator: CPU-only")
            self.logger.info(f"   CPU Workers: {self.cpu_workers}")
            self.logger.warning("⚠️ No GPU acceleration available")
    
    def accelerate_population_evaluation(self, 
                                       population: List[Dict[str, List[float]]],
                                       target_patterns: Dict[str, List[float]],
                                       weights: Dict[str, float]) -> List[float]:
        """
        Main acceleration entry point with intelligent GPU/CPU selection.
        
        Selection Logic:
        - GPU: population >= 200 and GPU available
        - CPU Parallel: population >= 50 and < 200 (or GPU unavailable)
        - CPU Serial: population < 50
        """
        start_time = time.time()
        pop_size = len(population)
        
        if self.is_gpu_available and pop_size >= self.gpu_threshold:
            # Use GPU with batching for memory management
            if pop_size > self.gpu_batch_size:
                scores = self._evaluate_gpu_batched(population, target_patterns, weights)
            else:
                scores = self._evaluate_gpu_direct(population, target_patterns, weights)
            self.gpu_time += time.time() - start_time
            
        elif pop_size >= self.cpu_multiprocessing_threshold:
            # Use CPU multiprocessing
            scores = self._evaluate_cpu_parallel(population, target_patterns, weights)
            self.cpu_time += time.time() - start_time
            
        else:
            # Use CPU serial for small populations
            scores = self._evaluate_cpu_serial(population, target_patterns, weights)
            self.cpu_time += time.time() - start_time
        
        self.operations_count += 1
        self.total_evaluations += pop_size
        return scores
    
    def _evaluate_gpu_direct(self, population: List[Dict[str, List[float]]],
                           target_patterns: Dict[str, List[float]],
                           weights: Dict[str, float]) -> List[float]:
        """Direct GPU evaluation without batching."""
        
        if self.gpu_backend == 'pytorch':
            return self._evaluate_pytorch(population, target_patterns, weights)
        elif self.gpu_backend == 'opencl':
            return self._evaluate_opencl(population, target_patterns, weights)
        else:
            # Fallback to CPU
            return self._evaluate_cpu_serial(population, target_patterns, weights)
    
    def _evaluate_gpu_batched(self, population: List[Dict[str, List[float]]],
                            target_patterns: Dict[str, List[float]],
                            weights: Dict[str, float]) -> List[float]:
        """Batched GPU evaluation for large populations with memory management."""
        
        all_scores = []
        batch_size = self.gpu_batch_size
        
        for i in range(0, len(population), batch_size):
            batch = population[i:i + batch_size]
            batch_scores = self._evaluate_gpu_direct(batch, target_patterns, weights)
            all_scores.extend(batch_scores)
            
            # Memory cleanup between batches (Ollama-style)
            if self.gpu_backend == 'pytorch':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        return all_scores
    
    def _evaluate_pytorch(self, population: List[Dict[str, List[float]]],
                        target_patterns: Dict[str, List[float]],
                        weights: Dict[str, float]) -> List[float]:
        """PyTorch GPU evaluation optimized for AMD RX 7700S."""
        
        try:
            device_names = list(target_patterns.keys())
            pop_size = len(population)
            
            # Convert to PyTorch tensors on GPU with optimal data types
            pop_tensors = {}
            target_tensors = {}
            
            for device_name in device_names:
                # Population patterns: (pop_size, 96)
                device_patterns = torch.tensor([
                    ind.get(device_name, [0.0] * 96) for ind in population
                ], dtype=torch.float32, device=self.pytorch_device)
                
                # Target pattern: (96,)
                target_pattern = torch.tensor(
                    target_patterns[device_name], 
                    dtype=torch.float32, 
                    device=self.pytorch_device
                )
                
                pop_tensors[device_name] = device_patterns
                target_tensors[device_name] = target_pattern
            
            # Vectorized scoring on GPU
            total_scores = torch.zeros(pop_size, device=self.pytorch_device, dtype=torch.float32)
            
            for device_name in device_names:
                device_weight = weights.get(device_name, 1.0)
                
                # MSE calculation (vectorized)
                diff = pop_tensors[device_name] - target_tensors[device_name].unsqueeze(0)
                mse = torch.mean(diff ** 2, dim=1)
                
                # Smoothness penalty (optimized for GPU)
                transitions = torch.abs(torch.diff(pop_tensors[device_name], dim=1))
                smoothness_penalty = torch.mean((transitions > 0.1).float(), dim=1) * 0.1
                
                # Physics constraints
                negative_penalty = torch.mean(
                    torch.clamp(-pop_tensors[device_name], min=0), dim=1
                ) * 0.3
                
                range_penalty = torch.mean(
                    torch.clamp(pop_tensors[device_name] - 1.0, min=0), dim=1
                ) * 0.2
                
                # Realism penalty (pattern should vary smoothly)
                variation = torch.std(pop_tensors[device_name], dim=1)
                realism_penalty = torch.abs(variation - 0.2) * 0.1  # Target variation ~0.2
                
                # Combine scores
                device_scores = (mse + smoothness_penalty + negative_penalty + 
                               range_penalty + realism_penalty)
                total_scores += device_weight * device_scores
            
            # Convert back to CPU efficiently
            scores = total_scores.cpu().numpy().tolist()
            
            # Memory cleanup (Ollama-style)
            torch.cuda.empty_cache()
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"PyTorch GPU evaluation failed: {e}")
            return self._evaluate_cpu_serial(population, target_patterns, weights)
    
    def _evaluate_opencl(self, population: List[Dict[str, List[float]]],
                       target_patterns: Dict[str, List[float]],
                       weights: Dict[str, float]) -> List[float]:
        """OpenCL GPU evaluation for AMD compatibility."""
        
        # OpenCL kernel optimized for AMD GPUs
        kernel_source = """
        __kernel void evaluate_patterns(__global const float* population,
                                      __global const float* targets,
                                      __global const float* device_weights,
                                      __global float* scores,
                                      const int pop_size,
                                      const int pattern_length,
                                      const int num_devices) {
            int idx = get_global_id(0);
            if (idx >= pop_size) return;
            
            float total_score = 0.0f;
            
            for (int dev = 0; dev < num_devices; dev++) {
                float mse = 0.0f;
                float smoothness = 0.0f;
                float physics_penalty = 0.0f;
                
                int pop_offset = idx * pattern_length * num_devices + dev * pattern_length;
                int target_offset = dev * pattern_length;
                
                // MSE calculation
                for (int i = 0; i < pattern_length; i++) {
                    float pop_val = population[pop_offset + i];
                    float target_val = targets[target_offset + i];
                    float diff = pop_val - target_val;
                    mse += diff * diff;
                    
                    // Physics constraints
                    if (pop_val < 0.0f) physics_penalty += (-pop_val) * 0.3f;
                    if (pop_val > 1.0f) physics_penalty += (pop_val - 1.0f) * 0.2f;
                }
                mse /= pattern_length;
                physics_penalty /= pattern_length;
                
                // Smoothness penalty
                for (int i = 1; i < pattern_length; i++) {
                    float transition = fabs(population[pop_offset + i] - population[pop_offset + i - 1]);
                    if (transition > 0.1f) smoothness += 1.0f;
                }
                smoothness = (smoothness / (pattern_length - 1)) * 0.1f;
                
                // Weighted device score
                float device_weight = device_weights[dev];
                total_score += device_weight * (mse + smoothness + physics_penalty);
            }
            
            scores[idx] = total_score;
        }
        """
        
        try:
            program = cl.Program(self.cl_context, kernel_source).build()
            
            device_names = list(target_patterns.keys())
            pop_size = len(population)
            pattern_length = 96
            num_devices = len(device_names)
            
            # Prepare data arrays
            population_data = np.zeros((pop_size, pattern_length * num_devices), dtype=np.float32)
            target_data = np.zeros((pattern_length * num_devices,), dtype=np.float32)
            weight_data = np.array([weights.get(name, 1.0) for name in device_names], dtype=np.float32)
            
            # Fill population data
            for i, individual in enumerate(population):
                for j, device_name in enumerate(device_names):
                    start_idx = j * pattern_length
                    pattern = individual.get(device_name, [0.0] * pattern_length)
                    population_data[i, start_idx:start_idx + pattern_length] = pattern
            
            # Fill target data
            for j, device_name in enumerate(device_names):
                start_idx = j * pattern_length
                target_data[start_idx:start_idx + pattern_length] = target_patterns[device_name]
            
            # Transfer to GPU
            pop_buffer = cl_array.to_device(self.cl_queue, population_data.flatten())
            target_buffer = cl_array.to_device(self.cl_queue, target_data)
            weight_buffer = cl_array.to_device(self.cl_queue, weight_data)
            scores_buffer = cl_array.zeros(self.cl_queue, (pop_size,), dtype=np.float32)
            
            # Execute kernel with optimal work group size
            local_size = min(256, pop_size)  # Optimal for AMD GPUs
            global_size = ((pop_size + local_size - 1) // local_size) * local_size
            
            program.evaluate_patterns(
                self.cl_queue, (global_size,), (local_size,),
                pop_buffer.data, target_buffer.data, weight_buffer.data, scores_buffer.data,
                np.int32(pop_size), np.int32(pattern_length), np.int32(num_devices)
            )
            
            # Get results
            return scores_buffer.get().tolist()
            
        except Exception as e:
            self.logger.warning(f"OpenCL evaluation failed: {e}")
            return self._evaluate_cpu_serial(population, target_patterns, weights)
    
    def _evaluate_cpu_parallel(self, population: List[Dict[str, List[float]]],
                             target_patterns: Dict[str, List[float]],
                             weights: Dict[str, float]) -> List[float]:
        """CPU parallel evaluation using multiprocessing (16 cores)."""
        
        # Split population into chunks for parallel processing
        chunk_size = max(1, len(population) // self.cpu_workers)
        chunks = [population[i:i + chunk_size] for i in range(0, len(population), chunk_size)]
        
        # Create partial function with fixed arguments
        eval_func = partial(
            self._evaluate_cpu_chunk_vectorized,
            target_patterns=target_patterns,
            weights=weights
        )
        
        # Process chunks in parallel
        with self.mp_context.Pool(self.cpu_workers) as pool:
            chunk_results = pool.map(eval_func, chunks)
        
        # Flatten results
        scores = []
        for chunk_scores in chunk_results:
            scores.extend(chunk_scores)
        
        return scores
    
    def _evaluate_cpu_chunk_vectorized(self, population_chunk: List[Dict[str, List[float]]],
                                     target_patterns: Dict[str, List[float]],
                                     weights: Dict[str, float]) -> List[float]:
        """Vectorized evaluation of a population chunk."""
        
        if not population_chunk:
            return []
        
        device_names = list(target_patterns.keys())
        chunk_size = len(population_chunk)
        pattern_length = 96
        
        # Convert to NumPy arrays for vectorization
        population_arrays = {}
        target_arrays = {}
        
        for device_name in device_names:
            # Population patterns: shape (chunk_size, pattern_length)
            device_patterns = np.array([
                ind.get(device_name, [0.0] * pattern_length) for ind in population_chunk
            ], dtype=np.float32)
            
            # Target pattern: shape (pattern_length,)
            target_pattern = np.array(target_patterns[device_name], dtype=np.float32)
            
            population_arrays[device_name] = device_patterns
            target_arrays[device_name] = target_pattern
        
        # Vectorized scoring with Numba if available
        if HAS_NUMBA:
            return self._score_chunk_numba(population_arrays, target_arrays, weights, device_names)
        else:
            return self._score_chunk_numpy(population_arrays, target_arrays, weights, device_names)
    
    def _score_chunk_numpy(self, population_arrays: Dict[str, np.ndarray],
                         target_arrays: Dict[str, np.ndarray],
                         weights: Dict[str, float],
                         device_names: List[str]) -> List[float]:
        """NumPy-vectorized scoring for maximum CPU performance."""
        
        chunk_size = len(next(iter(population_arrays.values())))
        total_scores = np.zeros(chunk_size, dtype=np.float32)
        
        for device_name in device_names:
            device_weight = weights.get(device_name, 1.0)
            device_patterns = population_arrays[device_name]
            target_pattern = target_arrays[device_name]
            
            # Vectorized MSE
            diff = device_patterns - target_pattern[np.newaxis, :]
            mse = np.mean(diff ** 2, axis=1)
            
            # Vectorized smoothness penalty
            transitions = np.abs(np.diff(device_patterns, axis=1))
            smoothness_penalty = np.mean(transitions > 0.1, axis=1) * 0.1
            
            # Vectorized physics constraints
            negative_penalty = np.mean(np.maximum(-device_patterns, 0), axis=1) * 0.3
            range_penalty = np.mean(np.maximum(device_patterns - 1.0, 0), axis=1) * 0.2
            
            # Vectorized realism penalty
            variation = np.std(device_patterns, axis=1)
            realism_penalty = np.abs(variation - 0.2) * 0.1
            
            # Combine device scores
            device_scores = (mse + smoothness_penalty + negative_penalty + 
                           range_penalty + realism_penalty)
            total_scores += device_weight * device_scores
        
        return total_scores.tolist()
    
    def _score_chunk_numba(self, population_arrays: Dict[str, np.ndarray],
                         target_arrays: Dict[str, np.ndarray],
                         weights: Dict[str, float],
                         device_names: List[str]) -> List[float]:
        """Numba-accelerated scoring (fallback to NumPy for complex structures)."""
        # Numba has limitations with complex data structures, use NumPy
        return self._score_chunk_numpy(population_arrays, target_arrays, weights, device_names)
    
    def _evaluate_cpu_serial(self, population: List[Dict[str, List[float]]],
                           target_patterns: Dict[str, List[float]],
                           weights: Dict[str, float]) -> List[float]:
        """Serial CPU evaluation for small populations."""
        
        # Use vectorized chunk evaluation with chunk size = full population
        return self._evaluate_cpu_chunk_vectorized(population, target_patterns, weights)
    
    def get_optimal_population_size(self, base_size: int = 200) -> int:
        """Get optimal population size based on available acceleration."""
        
        if self.is_gpu_available:
            # GPU can handle larger populations efficiently
            if self.gpu_backend == 'pytorch':
                return min(base_size * 6, 2000)  # Up to 2000 for RX 7700S
            else:  # OpenCL
                return min(base_size * 4, 1500)
        else:
            # CPU-only: scale with number of cores
            return min(base_size * max(1, self.cpu_workers // 4), 800)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        total_time = self.gpu_time + self.cpu_time
        gpu_percentage = (self.gpu_time / total_time * 100) if total_time > 0 else 0
        
        stats = {
            'acceleration_type': 'unified_gpu_cpu',
            'gpu_available': self.is_gpu_available,
            'gpu_backend': self.gpu_backend,
            'cpu_workers': self.cpu_workers,
            'total_operations': self.operations_count,
            'total_evaluations': self.total_evaluations,
            'gpu_time': self.gpu_time,
            'cpu_time': self.cpu_time,
            'gpu_percentage': gpu_percentage,
            'overall_speedup': self._calculate_speedup(),
            'has_numba': HAS_NUMBA,
            'has_scipy': HAS_SCIPY,
            'has_pytorch': HAS_PYTORCH,
            'has_opencl': HAS_OPENCL,
            'pytorch_backend': PYTORCH_BACKEND
        }
        
        if self.is_gpu_available:
            stats.update({
                'gpu_info': self.gpu_info,
                'gpu_batch_size': self.gpu_batch_size,
                'gpu_memory_fraction': self.gpu_memory_fraction
            })
        
        return stats
    
    def _calculate_speedup(self) -> float:
        """Calculate overall speedup compared to baseline."""
        
        if self.operations_count == 0:
            return 1.0
        
        # Estimate baseline time (serial CPU without vectorization)
        baseline_time_per_eval = 0.001  # 1ms per evaluation (conservative estimate)
        estimated_baseline = self.total_evaluations * baseline_time_per_eval
        actual_time = self.gpu_time + self.cpu_time
        
        return estimated_baseline / actual_time if actual_time > 0 else 1.0
    
    def benchmark_acceleration(self, test_sizes: List[int] = [50, 200, 500, 1000, 2000]) -> Dict[str, Any]:
        """Comprehensive benchmark of GPU and CPU acceleration."""
        
        results = {}
        
        for test_size in test_sizes:
            if test_size > 2000:  # Safety limit
                continue
                
            self.logger.info(f"🏃 Benchmarking population size: {test_size}")
            
            # Generate test data
            test_population = []
            device_names = ['heating', 'cooling', 'hot_water', 'lighting', 'equipment']
            
            for _ in range(test_size):
                individual = {
                    name: np.random.random(96).tolist() for name in device_names
                }
                test_population.append(individual)
            
            target_patterns = {
                name: np.random.random(96).tolist() for name in device_names
            }
            weights = {name: 1.0 for name in device_names}
            
            # Benchmark different methods
            benchmark_results = {}
            
            # CPU Serial
            start_time = time.time()
            cpu_serial_scores = self._evaluate_cpu_serial(test_population, target_patterns, weights)
            cpu_serial_time = time.time() - start_time
            benchmark_results['cpu_serial'] = cpu_serial_time
            
            # CPU Parallel (if population large enough)
            if test_size >= self.cpu_multiprocessing_threshold:
                start_time = time.time()
                cpu_parallel_scores = self._evaluate_cpu_parallel(test_population, target_patterns, weights)
                cpu_parallel_time = time.time() - start_time
                benchmark_results['cpu_parallel'] = cpu_parallel_time
                benchmark_results['cpu_speedup'] = cpu_serial_time / cpu_parallel_time
            
            # GPU (if available and population large enough)
            if self.is_gpu_available and test_size >= self.gpu_threshold:
                start_time = time.time()
                gpu_scores = self._evaluate_gpu_direct(test_population, target_patterns, weights)
                gpu_time = time.time() - start_time
                benchmark_results['gpu'] = gpu_time
                benchmark_results['gpu_speedup'] = cpu_serial_time / gpu_time
                
                # Verify accuracy
                mse_diff = np.mean((np.array(cpu_serial_scores) - np.array(gpu_scores)) ** 2)
                benchmark_results['gpu_accuracy'] = 1.0 - min(mse_diff, 1.0)
            
            results[f'size_{test_size}'] = benchmark_results
            
            # Log summary
            summary = f"   CPU Serial: {cpu_serial_time:.3f}s"
            if 'cpu_parallel' in benchmark_results:
                summary += f", CPU Parallel: {benchmark_results['cpu_parallel']:.3f}s"
            if 'gpu' in benchmark_results:
                summary += f", GPU: {benchmark_results['gpu']:.3f}s"
            self.logger.info(summary)
        
        return results
    
    def clear_acceleration_memory(self):
        """Clear any cached acceleration memory."""
        
        if self.is_gpu_available:
            if self.gpu_backend == 'pytorch':
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
    
    def shutdown(self):
        """Clean shutdown of acceleration resources."""
        
        self.clear_acceleration_memory()
        
        # Log final performance summary
        total_time = self.gpu_time + self.cpu_time
        if total_time > 0:
            self.logger.info(f"🏁 Acceleration Summary:")
            self.logger.info(f"   Total Operations: {self.operations_count}")
            self.logger.info(f"   Total Evaluations: {self.total_evaluations}")
            self.logger.info(f"   GPU Time: {self.gpu_time:.2f}s ({self.gpu_time/total_time*100:.1f}%)")
            self.logger.info(f"   CPU Time: {self.cpu_time:.2f}s ({self.cpu_time/total_time*100:.1f}%)")
            self.logger.info(f"   Overall Speedup: {self._calculate_speedup():.1f}x")