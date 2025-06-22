"""
Pattern Smoothing Module
========================

This module provides functions to create more realistic daily patterns
by smoothing mathematical patterns and adding natural variations.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple
import logging


class PatternSmoother:
    """Creates realistic, smooth daily patterns for devices."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Device-specific smoothing parameters
        self.device_params = {
            'heater': {
                'smoothing_window': 8,      # 2-hour smoothing window
                'min_transition_time': 4,   # Minimum 1-hour transitions
                'noise_level': 0.02,
                'natural_cycles': True,
                'cycle_length': 16          # 4-hour natural cycles
            },
            'air_conditioner': {
                'smoothing_window': 6,
                'min_transition_time': 3,
                'noise_level': 0.03,
                'natural_cycles': True,
                'cycle_length': 12
            },
            'refrigeration': {
                'smoothing_window': 12,     # Very smooth
                'min_transition_time': 8,
                'noise_level': 0.01,
                'natural_cycles': True,
                'cycle_length': 20          # Longer cycles
            },
            'general_load': {
                'smoothing_window': 4,
                'min_transition_time': 2,
                'noise_level': 0.04,
                'natural_cycles': False,
                'cycle_length': 8
            },
            'lighting': {
                'smoothing_window': 2,      # More responsive
                'min_transition_time': 1,
                'noise_level': 0.02,
                'natural_cycles': False,
                'cycle_length': 4
            },
            'water_heater': {
                'smoothing_window': 6,
                'min_transition_time': 4,
                'noise_level': 0.02,
                'natural_cycles': True,
                'cycle_length': 24          # Daily hot water cycles
            }
        }
    
    def smooth_pattern(self, pattern: List[float], device_type: str) -> List[float]:
        """Create a smooth, realistic pattern from a mathematical one."""
        
        if len(pattern) != 96:
            self.logger.warning(f"Pattern length {len(pattern)} != 96, padding/truncating")
            if len(pattern) < 96:
                pattern.extend([pattern[-1]] * (96 - len(pattern)))
            else:
                pattern = pattern[:96]
        
        params = self.device_params.get(device_type, self.device_params['general_load'])
        
        # Convert to numpy array
        smooth_pattern = np.array(pattern, dtype=float)
        
        # Step 1: Apply moving average smoothing
        smooth_pattern = self._apply_moving_average(smooth_pattern, params['smoothing_window'])
        
        # Step 2: Ensure minimum transition times
        smooth_pattern = self._enforce_transition_times(smooth_pattern, params['min_transition_time'])
        
        # Step 3: Add natural cycles if applicable
        if params['natural_cycles']:
            smooth_pattern = self._add_natural_cycles(smooth_pattern, params['cycle_length'])
        
        # Step 4: Add realistic noise
        smooth_pattern = self._add_realistic_noise(smooth_pattern, params['noise_level'])
        
        # Step 5: Ensure valid range [0, 1]
        smooth_pattern = np.clip(smooth_pattern, 0.0, 1.0)
        
        return smooth_pattern.tolist()
    
    def _apply_moving_average(self, pattern: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average smoothing with edge handling."""
        
        # Use convolution for moving average
        kernel = np.ones(window_size) / window_size
        
        # Pad the pattern to handle edges
        padded_pattern = np.pad(pattern, window_size//2, mode='edge')
        smoothed = np.convolve(padded_pattern, kernel, mode='valid')
        
        # Ensure same length as original
        return smoothed[:len(pattern)]
    
    def _enforce_transition_times(self, pattern: np.ndarray, min_transition: int) -> np.ndarray:
        """Ensure transitions take at least min_transition periods."""
        
        smooth_pattern = pattern.copy()
        
        # Find significant changes (> 0.2 difference)
        changes = np.abs(np.diff(pattern)) > 0.2
        change_indices = np.where(changes)[0]
        
        for idx in change_indices:
            start_val = pattern[idx]
            end_val = pattern[idx + 1]
            
            # Create gradual transition
            transition_length = min(min_transition, 96 - idx - 1)
            if transition_length > 1:
                transition_values = np.linspace(start_val, end_val, transition_length + 1)
                end_idx = min(idx + transition_length + 1, len(smooth_pattern))
                smooth_pattern[idx:end_idx] = transition_values[:end_idx-idx]
        
        return smooth_pattern
    
    def _add_natural_cycles(self, pattern: np.ndarray, cycle_length: int) -> np.ndarray:
        """Add natural cyclical variations."""
        
        # Create subtle sine wave variations
        time_points = np.arange(len(pattern))
        
        # Primary cycle
        primary_cycle = 0.03 * np.sin(2 * np.pi * time_points / cycle_length)
        
        # Secondary harmonic for more natural feel
        secondary_cycle = 0.01 * np.sin(4 * np.pi * time_points / cycle_length + np.pi/3)
        
        # Add cycles to pattern
        enhanced_pattern = pattern + primary_cycle + secondary_cycle
        
        return enhanced_pattern
    
    def _add_realistic_noise(self, pattern: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic random variations."""
        
        # Correlated noise (not pure white noise)
        noise = np.random.normal(0, noise_level, len(pattern))
        
        # Apply exponential smoothing to make noise more realistic
        smooth_noise = np.zeros_like(noise)
        alpha = 0.3  # Smoothing factor
        
        smooth_noise[0] = noise[0]
        for i in range(1, len(noise)):
            smooth_noise[i] = alpha * noise[i] + (1 - alpha) * smooth_noise[i-1]
        
        return pattern + smooth_noise
    
    def create_realistic_pattern_from_type(self, device_type: str, 
                                         usage_intensity: float = 1.0) -> List[float]:
        """Create a completely realistic pattern based on device type and usage."""
        
        if device_type == 'heater':
            return self._create_heater_pattern(usage_intensity)
        elif device_type == 'air_conditioner':
            return self._create_ac_pattern(usage_intensity)
        elif device_type == 'refrigeration':
            return self._create_refrigeration_pattern(usage_intensity)
        elif device_type == 'general_load':
            return self._create_general_load_pattern(usage_intensity)
        elif device_type == 'lighting':
            return self._create_lighting_pattern(usage_intensity)
        elif device_type == 'water_heater':
            return self._create_water_heater_pattern(usage_intensity)
        else:
            return self._create_generic_pattern(usage_intensity)
    
    def _create_heater_pattern(self, intensity: float) -> List[float]:
        """Create realistic heating pattern with thermal inertia."""
        
        # Base pattern: higher at night and morning/evening
        hours = np.arange(96) / 4  # Convert to hours
        
        # Night heating (thermal loss compensation)
        night_heating = 0.4 + 0.3 * np.exp(-((hours - 3) % 24 / 4)**2)
        
        # Morning warm-up
        morning_warmup = 0.3 * np.exp(-((hours - 6.5) % 24 / 1.5)**2)
        
        # Evening heating
        evening_heating = 0.4 * np.exp(-((hours - 19) % 24 / 2)**2)
        
        # Combine patterns
        base_pattern = night_heating + morning_warmup + evening_heating
        base_pattern = np.clip(base_pattern * intensity, 0, 1)
        
        # Apply smoothing
        return self.smooth_pattern(base_pattern.tolist(), 'heater')
    
    def _create_ac_pattern(self, intensity: float) -> List[float]:
        """Create realistic air conditioning pattern."""
        
        hours = np.arange(96) / 4
        
        # Peak during hottest part of day
        afternoon_peak = 0.8 * np.exp(-((hours - 14) % 24 / 3)**2)
        
        # Moderate evening cooling
        evening_cooling = 0.4 * np.exp(-((hours - 20) % 24 / 2)**2)
        
        # Minimal night operation
        night_base = 0.1
        
        base_pattern = afternoon_peak + evening_cooling + night_base
        base_pattern = np.clip(base_pattern * intensity, 0, 1)
        
        return self.smooth_pattern(base_pattern.tolist(), 'air_conditioner')
    
    def _create_refrigeration_pattern(self, intensity: float) -> List[float]:
        """Create realistic refrigeration pattern with compressor cycles."""
        
        # Nearly constant with small variations for door openings and ambient temperature
        base_load = 0.8 * intensity
        
        # Small increases during meal times when door opened more
        hours = np.arange(96) / 4
        meal_increases = (
            0.1 * np.exp(-((hours - 7) % 24 / 1)**2) +   # Breakfast
            0.15 * np.exp(-((hours - 12) % 24 / 1)**2) +  # Lunch  
            0.2 * np.exp(-((hours - 18) % 24 / 1.5)**2)   # Dinner
        )
        
        base_pattern = np.full(96, base_load) + meal_increases
        base_pattern = np.clip(base_pattern, 0, 1)
        
        return self.smooth_pattern(base_pattern.tolist(), 'refrigeration')
    
    def _create_general_load_pattern(self, intensity: float) -> List[float]:
        """Create realistic general household load pattern."""
        
        hours = np.arange(96) / 4
        
        # Morning activity
        morning_activity = 0.4 * (1 / (1 + np.exp(-2 * (hours - 7)))) * (1 / (1 + np.exp(2 * (hours - 10))))
        
        # Daytime moderate load
        daytime_load = 0.3 + 0.1 * np.sin(2 * np.pi * hours / 24)
        
        # Evening peak
        evening_peak = 0.6 * (1 / (1 + np.exp(-3 * (hours - 17)))) * (1 / (1 + np.exp(2 * (hours - 22))))
        
        # Night minimum
        night_min = 0.2
        
        base_pattern = np.maximum.reduce([morning_activity, daytime_load, evening_peak, 
                                        np.full(96, night_min)])
        base_pattern = np.clip(base_pattern * intensity, 0, 1)
        
        return self.smooth_pattern(base_pattern.tolist(), 'general_load')
    
    def _create_lighting_pattern(self, intensity: float) -> List[float]:
        """Create realistic lighting pattern based on sunrise/sunset."""
        
        hours = np.arange(96) / 4
        
        # Morning gradual increase (6-8 AM)
        morning_light = 0.3 * (1 / (1 + np.exp(-4 * (hours - 6.5))))
        morning_light *= (1 / (1 + np.exp(4 * (hours - 8))))
        
        # Minimal daytime (natural light available)
        daytime_base = 0.1
        
        # Evening sharp increase starting around sunset (varies by season)
        evening_start = 17  # Can be adjusted seasonally
        evening_light = 0.8 * (1 / (1 + np.exp(-6 * (hours - evening_start))))
        
        # Night reduction after bedtime (22:30)
        night_reduction = (1 / (1 + np.exp(8 * (hours - 22.5))))
        
        base_pattern = morning_light + daytime_base + evening_light * night_reduction
        base_pattern = np.clip(base_pattern * intensity, 0, 1)
        
        return self.smooth_pattern(base_pattern.tolist(), 'lighting')
    
    def _create_water_heater_pattern(self, intensity: float) -> List[float]:
        """Create realistic water heater pattern with distinct peaks."""
        
        hours = np.arange(96) / 4
        
        # Morning shower peak (6-8 AM)
        morning_showers = 0.8 * np.exp(-((hours - 7) % 24 / 0.8)**2)
        
        # Evening shower/bath peak (18-21)
        evening_showers = 0.9 * np.exp(-((hours - 19.5) % 24 / 1.2)**2)
        
        # Dishwashing peaks after meals
        dishwashing = (
            0.3 * np.exp(-((hours - 8.5) % 24 / 0.5)**2) +   # After breakfast
            0.4 * np.exp(-((hours - 13) % 24 / 0.5)**2) +     # After lunch
            0.5 * np.exp(-((hours - 20.5) % 24 / 0.8)**2)     # After dinner
        )
        
        # Baseline for maintaining temperature
        baseline = 0.15
        
        base_pattern = baseline + morning_showers + evening_showers + dishwashing
        base_pattern = np.clip(base_pattern * intensity, 0, 1)
        
        return self.smooth_pattern(base_pattern.tolist(), 'water_heater')
    
    def _create_generic_pattern(self, intensity: float) -> List[float]:
        """Create a generic realistic pattern."""
        
        hours = np.arange(96) / 4
        
        # Simple realistic pattern with morning and evening peaks
        pattern = (
            0.3 +  # Base load
            0.3 * np.sin(2 * np.pi * hours / 24) +  # Daily cycle
            0.2 * np.sin(4 * np.pi * hours / 24)    # Twice-daily peaks
        )
        
        pattern = np.clip(pattern * intensity, 0, 1)
        return self.smooth_pattern(pattern.tolist(), 'general_load')
    
    def enhance_config_patterns(self, config: Dict) -> Dict:
        """Enhance all device patterns in a configuration with realistic smoothing."""
        
        enhanced_config = config.copy()
        
        if 'devices' not in enhanced_config:
            return enhanced_config
        
        self.logger.info("Enhancing device patterns with realistic smoothing")
        
        for device_name, device_config in enhanced_config['devices'].items():
            if 'daily_pattern' in device_config:
                original_pattern = device_config['daily_pattern']
                
                # Apply smoothing to existing pattern
                smooth_pattern = self.smooth_pattern(original_pattern, device_name)
                
                # Optionally, replace with completely realistic pattern
                # realistic_pattern = self.create_realistic_pattern_from_type(device_name, 1.0)
                
                enhanced_config['devices'][device_name]['daily_pattern'] = smooth_pattern
                
                self.logger.info(f"Enhanced pattern for {device_name}")
        
        return enhanced_config