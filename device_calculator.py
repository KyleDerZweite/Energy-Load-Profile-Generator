"""
Realistic Device Load Calculator - Realism-First Approach
=========================================================

This replaces the mathematical device calculator with a physics-based,
realistic-first approach that automatically adapts to device changes.

Features:
- Thermal inertia and realistic device behavior
- Automatic pattern optimization for realism
- AI-driven adaptation to device changes
- Physics-based load calculations
- Smooth transitions and natural variations
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Tuple, Any
import logging
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class DeviceLoadCalculator:
    """Realistic-first device load calculator with automatic adaptation."""
    
    def __init__(self, device_configs: Dict):
        self.device_configs = device_configs
        self.logger = logging.getLogger(__name__)
        
        # Automatically enhance all patterns for realism
        self._auto_enhance_patterns()
        
        # Device behavior models with physics-based parameters - now dynamic
        self.device_models = {}
        self._initialize_dynamic_device_models()
        
        # State tracking for realistic behavior
        self._device_states = {}
        self._initialize_device_states()
        
        # Auto-adaptation parameters
        self.adaptation_enabled = True
        self.learning_rate = 0.05
        self.adaptation_history = {}
        
        # Device interactions for realistic coupling - now dynamic
        self._device_interactions = {}
        self._initialize_dynamic_interactions()
        
        # Initialize intelligent features
        self._occupancy_patterns = {}
        self._learning_engine = None
        self._pattern_memory = {}
        self._device_signatures = {}
        self._learned_patterns = {}
        
        # Physics engine and constraints (placeholders for now)
        self.physics_engine = None
        self.physics_constraints = {
            'max_power_change_rate': 0.2,  # 20% per 15-min interval
            'min_device_runtime': 2,       # Minimum 30 minutes
            'thermal_coupling_limits': 0.5
        }
        
        self.logger.info("✅ Realistic Device Calculator initialized with auto-enhancement")

    def _auto_enhance_patterns(self):
        """Automatically enhance all device patterns for maximum realism."""
        from pattern_smoother import PatternSmoother
        
        smoother = PatternSmoother()
        
        for device_name, device_config in self.device_configs.items():
            if 'daily_pattern' in device_config:
                # Get existing pattern
                original_pattern = device_config['daily_pattern']
                
                # First smooth existing pattern
                smoothed_pattern = smoother.smooth_pattern(original_pattern, device_name)
                
                # Then replace with physics-based realistic pattern
                realistic_pattern = smoother.create_realistic_pattern_from_type(device_name, 1.0)
                
                # Blend the two for best of both worlds (keeps some user intent)
                blended_pattern = [
                    0.3 * s + 0.7 * r 
                    for s, r in zip(smoothed_pattern, realistic_pattern)
                ]
                
                # Update the config
                device_config['daily_pattern'] = blended_pattern
                device_config['pattern_enhanced'] = True
                
                self.logger.info(f"🔧 Auto-enhanced pattern for {device_name}")

    def _initialize_dynamic_device_models(self):
        """Initialize device models dynamically based on device categories and types."""
        for device_name, device_config in self.device_configs.items():
            category = device_config.get('category', 'appliance')
            power_class = device_config.get('power_class', 'medium')
            usage_pattern = device_config.get('usage_pattern', 'scheduled')
            
            # Determine device model type based on characteristics
            if category == 'infrastructure':
                if 'kühl' in device_name.lower() or 'kälte' in device_name.lower():
                    self.device_models[device_name] = ThermalDevice(device_name, cooling=True)
                elif 'heiz' in device_name.lower() or 'wärme' in device_name.lower():
                    self.device_models[device_name] = ThermalDevice(device_name, heating=True)
                elif 'lüft' in device_name.lower() or 'air' in device_name.lower():
                    self.device_models[device_name] = CyclicDevice(device_name)
                else:
                    self.device_models[device_name] = AdaptiveDevice(device_name)
            elif category == 'equipment':
                if usage_pattern == 'continuous':
                    self.device_models[device_name] = CyclicDevice(device_name)
                else:
                    self.device_models[device_name] = AdaptiveDevice(device_name)
            else:  # appliance
                if 'light' in device_name.lower() or 'beleucht' in device_name.lower():
                    self.device_models[device_name] = ResponsiveDevice(device_name)
                else:
                    self.device_models[device_name] = AdaptiveDevice(device_name)
        
        self.logger.info(f"🤖 Initialized {len(self.device_models)} dynamic device models")

    def _initialize_dynamic_interactions(self):
        """Initialize device interactions dynamically based on device types."""
        for device_name, device_config in self.device_configs.items():
            self._device_interactions[device_name] = {'thermal_coupling': {}}
            
            # Heating devices have negative coupling with cooling devices
            if 'heiz' in device_name.lower() or 'wärme' in device_name.lower():
                for other_name, other_config in self.device_configs.items():
                    if 'kühl' in other_name.lower() or 'kälte' in other_name.lower():
                        self._device_interactions[device_name]['thermal_coupling'][other_name] = -0.3
            
            # Cooling devices have negative coupling with heating devices  
            elif 'kühl' in device_name.lower() or 'kälte' in device_name.lower():
                for other_name, other_config in self.device_configs.items():
                    if 'heiz' in other_name.lower() or 'wärme' in other_name.lower():
                        self._device_interactions[device_name]['thermal_coupling'][other_name] = -0.3
            
            # Lighting devices have small positive coupling with thermal devices
            elif 'light' in device_name.lower() or 'beleucht' in device_name.lower():
                for other_name, other_config in self.device_configs.items():
                    if ('heiz' in other_name.lower() or 'kühl' in other_name.lower() or 
                        'wärme' in other_name.lower() or 'kälte' in other_name.lower()):
                        self._device_interactions[device_name]['thermal_coupling'][other_name] = 0.05

    def _initialize_device_states(self):
        """Initialize realistic state tracking for all devices."""
        for device_name in self.device_configs.keys():
            self._device_states[device_name] = {
                'current_power': 0.0,
                'target_power': 0.0,
                'thermal_state': 0.0,
                'cycle_position': 0,
                'last_change_time': 0,
                'adaptation_factor': 1.0,
                'efficiency': 1.0,
                'wear_factor': 1.0,
                'weather_sensitivity': 0.5,
                'learning_memory': [],
                'realism_factor': 1.0
            }

    def calculate_total_load(self, weather_data: pd.DataFrame, devices: List[str],
                             quantities: Dict[str, int] = None) -> pd.DataFrame:
        """Calculate realistic total load with automatic adaptation."""
        
        if quantities is None:
            quantities = {device: 1 for device in devices}

        self.logger.info(f"🏠 Calculating realistic load for: {devices}")
        
        # Prepare the output dataframe
        result_df = weather_data.copy()
        
        # Initialize device power columns
        for device in devices:
            result_df[f'{device}_power'] = 0.0
        
        # Calculate realistic load for each device
        total_power = np.zeros(len(weather_data))
        
        for device in devices:
            device_quantity = quantities.get(device, 1)
            
            self.logger.info(f"🔧 Processing {device} (quantity: {device_quantity})")
            
            # Get device model
            device_model = self.device_models.get(device, AdaptiveDevice(device))
            
            # Calculate realistic power for this device
            device_power = self._calculate_realistic_device_load(
                device, device_model, weather_data, device_quantity
            )
            
            result_df[f'{device}_power'] = device_power
            total_power += device_power
        
        # Add total power
        result_df['total_power'] = total_power
        
        # Add time-based columns for analysis
        result_df['hour_of_day'] = result_df.index.hour
        result_df['minute_of_day'] = result_df.index.minute
        result_df['day_of_year'] = result_df.index.dayofyear
        
        # Apply intelligent post-processing
        result_df = self._apply_intelligent_post_processing(result_df, devices, weather_data)
        
        self.logger.info(f"✅ Generated intelligent realistic load profile: {len(result_df):,} records")
        return result_df

    def _calculate_realistic_device_load(self, device_name: str, device_model, 
                                       weather_data: pd.DataFrame, quantity: int) -> np.ndarray:
        """Calculate realistic load for a single device."""
        
        config = self.device_configs.get(device_name, {})
        if not config.get('enabled', True):
            return np.zeros(len(weather_data))
        
        # Get device parameters
        peak_power = config.get('peak_power', 1000) * quantity
        daily_pattern = np.array(config.get('daily_pattern', [0.5] * 96))
        
        # Calculate base load using realistic device model
        device_power = device_model.calculate_load(
            weather_data=weather_data,
            peak_power=peak_power,
            daily_pattern=daily_pattern,
            config=config,
            device_state=self._device_states[device_name]
        )
        
        # Apply auto-adaptation if enabled
        if self.adaptation_enabled:
            device_power = self._apply_auto_adaptation(device_name, device_power, weather_data)
        
        return device_power

    def _apply_auto_adaptation(self, device_name: str, device_power: np.ndarray, 
                             weather_data: pd.DataFrame) -> np.ndarray:
        """Apply AI-driven automatic adaptation to device behavior."""
        
        device_state = self._device_states[device_name]
        
        # Adaptation based on weather patterns
        temp_adaptation = self._adapt_to_temperature_patterns(device_power, weather_data['temperature'])
        
        # Adaptation based on time patterns
        time_adaptation = self._adapt_to_time_patterns(device_power, weather_data.index)
        
        # Efficiency degradation over time (realistic aging)
        efficiency_factor = device_state['efficiency']
        
        # Combine adaptations
        adapted_power = device_power * temp_adaptation * time_adaptation * efficiency_factor
        
        # Update device state
        device_state['adaptation_factor'] = np.mean([temp_adaptation, time_adaptation])
        device_state['efficiency'] *= 0.99999  # Slow degradation
        
        return adapted_power

    def _adapt_to_temperature_patterns(self, device_power: np.ndarray, 
                                     temperatures: pd.Series) -> float:
        """Adapt device behavior based on temperature patterns."""
        
        # More realistic temperature response
        temp_range = temperatures.max() - temperatures.min()
        temp_variability = temperatures.std()
        
        # Devices become more efficient in moderate conditions
        if temp_range < 15 and temp_variability < 5:
            return 0.95  # More efficient in stable conditions
        elif temp_range > 25:
            return 1.1   # Less efficient in extreme conditions
        else:
            return 1.0   # Normal efficiency

    def _adapt_to_time_patterns(self, device_power: np.ndarray, time_index: pd.DatetimeIndex) -> float:
        """Adapt device behavior based on time patterns."""
        
        # Weekend vs weekday patterns
        is_weekend = time_index.to_series().dt.weekday >= 5
        weekend_ratio = is_weekend.sum() / len(is_weekend)
        
        # Seasonal adaptation
        month = time_index.month[0] if len(time_index) > 0 else 6
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * (month - 3) / 12)
        
        # Time-of-day adaptation
        avg_hour = time_index.hour.values.mean() if len(time_index) > 0 else 12
        hour_factor = 1.0 + 0.05 * np.sin(2 * np.pi * avg_hour / 24)
        
        return seasonal_factor * hour_factor * (1 + 0.05 * weekend_ratio)

    def _apply_realistic_post_processing(self, load_data: pd.DataFrame, devices: List[str]) -> pd.DataFrame:
        """Apply final realistic post-processing to the entire load profile."""
        
        # Add correlated variations between devices
        correlation_matrix = self._get_device_correlations(devices)
        
        # Apply natural load diversity
        load_data = self._apply_load_diversity(load_data, devices)
        
        # Add realistic noise and variations
        load_data = self._add_realistic_variations(load_data, devices)
        
        # Ensure no negative values
        for device in devices:
            device_col = f'{device}_power'
            if device_col in load_data.columns:
                load_data[device_col] = np.maximum(load_data[device_col], 0)
        
        # Recalculate total
        device_columns = [f'{device}_power' for device in devices 
                         if f'{device}_power' in load_data.columns]
        load_data['total_power'] = load_data[device_columns].sum(axis=1)
        
        return load_data

    def _get_device_correlations(self, devices: List[str]) -> Dict:
        """Get realistic correlations between device operations - now dynamic."""
        
        correlations = {}
        
        for device1 in devices:
            for device2 in devices:
                if device1 != device2:
                    # Heating/cooling negative correlation
                    if (('heiz' in device1.lower() or 'wärme' in device1.lower()) and 
                        ('kühl' in device2.lower() or 'kälte' in device2.lower())):
                        correlations[(device1, device2)] = -0.8
                    elif (('kühl' in device1.lower() or 'kälte' in device1.lower()) and 
                          ('heiz' in device2.lower() or 'wärme' in device2.lower())):
                        correlations[(device1, device2)] = -0.8
                    # Lighting and general equipment positive correlation
                    elif (('light' in device1.lower() or 'beleucht' in device1.lower()) and 
                          device2 in self.device_configs):
                        device2_category = self.device_configs[device2].get('category', '')
                        if device2_category == 'equipment':
                            correlations[(device1, device2)] = 0.6
                    # Infrastructure devices slight positive correlation
                    elif (self.device_configs.get(device1, {}).get('category') == 'infrastructure' and
                          self.device_configs.get(device2, {}).get('category') == 'infrastructure'):
                        correlations[(device1, device2)] = 0.3
        
        return correlations

    def _apply_load_diversity(self, load_data: pd.DataFrame, devices: List[str]) -> pd.DataFrame:
        """Apply realistic load diversity factors."""
        
        # Not all devices peak at the same time (diversity factor)
        for i, device in enumerate(devices):
            device_col = f'{device}_power'
            if device_col in load_data.columns:
                # Apply phase shift for diversity
                phase_shift = i * 0.2  # 20% phase difference between devices
                diversity_factor = 0.9 + 0.1 * np.sin(
                    2 * np.pi * np.arange(len(load_data)) / 96 + phase_shift
                )
                load_data[device_col] *= diversity_factor
        
        return load_data

    def _add_realistic_variations(self, load_data: pd.DataFrame, devices: List[str]) -> pd.DataFrame:
        """Add realistic random variations and correlations."""
        
        # Generate correlated noise for realistic variations
        base_noise = np.random.normal(0, 0.02, len(load_data))
        
        for device in devices:
            device_col = f'{device}_power'
            if device_col in load_data.columns:
                # Device-specific noise characteristics
                device_noise_level = self._get_device_noise_level(device)
                
                # Correlated noise (devices in same building have correlated variations)
                device_noise = 0.7 * base_noise + 0.3 * np.random.normal(0, device_noise_level, len(load_data))
                
                # Apply noise proportional to power level
                load_data[device_col] += device_noise * load_data[device_col]
        
        return load_data

    def _get_device_noise_level(self, device: str) -> float:
        """Get realistic noise level for each device type."""
        
        noise_levels = {
            'heater': 0.03,
            'air_conditioner': 0.04,
            'refrigeration': 0.02,
            'general_load': 0.05,
            'lighting': 0.02,
            'water_heater': 0.03
        }
        
        return noise_levels.get(device, 0.03)

    def get_device_statistics(self, load_data: pd.DataFrame, device_name: str) -> Dict:
        """Get enhanced statistics for a specific device."""
        device_column = f'{device_name}_power'

        if device_column not in load_data.columns:
            return {}

        device_data = load_data[device_column]
        peak_power = self.device_configs[device_name]['peak_power']

        # Standard statistics
        stats = {
            'average_power_w': device_data.mean(),
            'max_power_w': device_data.max(),
            'min_power_w': device_data.min(),
            'total_energy_kwh': device_data.sum() * 0.25 / 1000,
            'capacity_factor': device_data.mean() / peak_power,
            'peak_capacity_factor': device_data.max() / peak_power,
            'peak_to_average_ratio': device_data.max() / device_data.mean() if device_data.mean() > 0 else 0,
            'configured_peak_power_w': peak_power
        }
        
        # Realistic behavior statistics
        transitions = np.abs(np.diff(device_data))
        stats.update({
            'max_transition_w': transitions.max(),
            'avg_transition_w': transitions.mean(),
            'smooth_transitions_pct': (transitions < 0.1 * peak_power).mean() * 100,
            'realism_score': self._calculate_realism_score(device_data, device_name),
            'efficiency_factor': self._device_states[device_name]['efficiency'],
            'adaptation_factor': self._device_states[device_name]['adaptation_factor']
        })

        return stats

    def _calculate_realism_score(self, device_data: pd.Series, device_name: str) -> float:
        """Calculate a realism score for the device behavior (0-100)."""
        
        score = 100.0
        
        # Penalize large jumps
        transitions = np.abs(np.diff(device_data))
        peak_power = self.device_configs[device_name]['peak_power']
        large_jumps = (transitions > 0.2 * peak_power).sum()
        score -= large_jumps * 10
        
        # Reward smooth operation
        smooth_ratio = (transitions < 0.05 * peak_power).mean()
        score += smooth_ratio * 20
        
        # Penalize unrealistic patterns
        if device_data.std() < 0.01 * peak_power:  # Too constant
            score -= 30
        
        return max(0, min(100, score))

    def _apply_intelligent_post_processing(self, load_data: pd.DataFrame, 
                                         devices: List[str], 
                                         weather_data: pd.DataFrame) -> pd.DataFrame:
        """Apply intelligent post-processing with device interactions and learning."""
        
        # Apply learned device interactions
        load_data = self._apply_device_interactions(load_data, devices)
        
        # Apply occupancy-aware adjustments
        if self._occupancy_patterns:
            load_data = self._apply_occupancy_adjustments(load_data, devices)
        
        # Apply physics constraints (if physics engine is available)
        if self.physics_engine:
            load_data = self.physics_engine.enforce_constraints(load_data, devices, self.physics_constraints)
        else:
            # Simple physics constraints without full engine
            load_data = self._apply_basic_physics_constraints(load_data)
        
        # Apply intelligent load diversity
        load_data = self._apply_intelligent_diversity(load_data, devices)
        
        # Add realistic correlations and uncertainties
        load_data = self._add_realistic_correlations(load_data, devices, weather_data)
        
        # Final realism validation
        realism_score = self._validate_realism(load_data, devices)
        load_data.attrs['realism_score'] = realism_score
        
        return load_data
    
    def _apply_basic_physics_constraints(self, load_data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic physics constraints without full physics engine."""
        # Simple smoothing to prevent unrealistic jumps
        for device_col in [col for col in load_data.columns if col.endswith('_power')]:
            # Apply gentle smoothing to prevent sudden changes
            load_data[device_col] = load_data[device_col].rolling(window=3, center=True, min_periods=1).mean()
        
        # Recalculate total power
        device_columns = [col for col in load_data.columns if col.endswith('_power') and col != 'total_power']
        load_data['total_power'] = load_data[device_columns].sum(axis=1)
        
        return load_data
    
    def _apply_device_interactions(self, load_data: pd.DataFrame, devices: List[str]) -> pd.DataFrame:
        """Apply learned device interaction effects."""
        
        # Thermal interactions (heat gains/losses between devices)
        for device1 in devices:
            for device2 in devices:
                if device1 != device2:
                    interaction = self._device_interactions.get(device1, {}).get('thermal_coupling', {}).get(device2, 0)
                    if abs(interaction) > 0.01:
                        device1_col = f'{device1}_power'
                        device2_col = f'{device2}_power'
                        if device1_col in load_data.columns and device2_col in load_data.columns:
                            # Apply thermal coupling
                            thermal_effect = load_data[device1_col] * interaction * 0.1
                            load_data[device2_col] += thermal_effect
        
        return load_data
    
    def _apply_occupancy_adjustments(self, load_data: pd.DataFrame, devices: List[str]) -> pd.DataFrame:
        """Apply occupancy-aware load adjustments."""
        
        if 'occupancy_probability' in self._occupancy_patterns:
            occupancy_pattern = self._occupancy_patterns['occupancy_probability']
            
            for device in devices:
                device_col = f'{device}_power'
                if device_col in load_data.columns:
                    # Get device sensitivity to occupancy
                    occupancy_sensitivity = self._get_occupancy_sensitivity(device)
                    
                    # Apply occupancy pattern
                    for i, timestamp in enumerate(load_data.index):
                        hour = timestamp.hour
                        weekday = timestamp.weekday()
                        
                        # Get occupancy probability for this time
                        occupancy_prob = occupancy_pattern.get((weekday, hour), 0.5)
                        
                        # Apply occupancy effect
                        occupancy_factor = 1.0 + (occupancy_prob - 0.5) * occupancy_sensitivity
                        load_data.iloc[i][device_col] *= occupancy_factor
        
        return load_data
    
    def _get_occupancy_sensitivity(self, device: str) -> float:
        """Get device sensitivity to occupancy patterns."""
        sensitivity_map = {
            'lighting': 0.8,
            'general_load': 0.6,
            'water_heater': 0.4,
            'air_conditioner': 0.3,
            'heater': 0.3,
            'refrigeration': 0.1
        }
        return sensitivity_map.get(device, 0.3)
    
    def _apply_intelligent_diversity(self, load_data: pd.DataFrame, devices: List[str]) -> pd.DataFrame:
        """Apply intelligent load diversity based on learned patterns."""
        
        # Calculate dynamic diversity factors based on actual correlations
        for i, device in enumerate(devices):
            device_col = f'{device}_power'
            if device_col in load_data.columns:
                # Phase shift based on device type and learned patterns
                device_type_phase = self._get_device_type_phase(device)
                learned_phase = self._learned_patterns.get(device, {}).get('phase_shift', 0)
                
                total_phase = device_type_phase + learned_phase
                
                # Apply diversity with learned timing
                diversity_factor = 0.85 + 0.15 * np.sin(
                    2 * np.pi * np.arange(len(load_data)) / 96 + total_phase
                )
                load_data[device_col] *= diversity_factor
        
        return load_data
    
    def _get_device_type_phase(self, device: str) -> float:
        """Get typical phase shift for device type."""
        phase_map = {
            'heater': 0.0,
            'air_conditioner': 0.5,
            'water_heater': 0.25,
            'lighting': 0.75,
            'general_load': 0.1,
            'refrigeration': 0.3
        }
        return phase_map.get(device, 0.0) * 2 * np.pi
    
    def _add_realistic_correlations(self, load_data: pd.DataFrame, 
                                  devices: List[str], 
                                  weather_data: pd.DataFrame) -> pd.DataFrame:
        """Add realistic correlations and uncertainties."""
        
        # Weather-correlated variations
        temp_variation = weather_data['temperature'].diff().fillna(0)
        
        # Add correlated noise based on weather changes
        base_noise = np.random.normal(0, 0.01, len(load_data))
        weather_noise = temp_variation * 0.002
        
        for device in devices:
            device_col = f'{device}_power'
            if device_col in load_data.columns:
                # Device-specific correlation with weather
                weather_sensitivity = self._device_states[device]['weather_sensitivity']
                
                # Combine noise sources
                device_noise = (base_noise + weather_noise * weather_sensitivity) * load_data[device_col]
                load_data[device_col] += device_noise
                
                # Ensure non-negative values
                load_data[device_col] = np.maximum(load_data[device_col], 0)
        
        return load_data
    
    def _validate_realism(self, load_data: pd.DataFrame, devices: List[str]) -> float:
        """Validate overall realism of the generated load profile."""
        
        realism_scores = []
        
        for device in devices:
            device_col = f'{device}_power'
            if device_col in load_data.columns:
                device_score = self._calculate_device_realism_score(load_data[device_col], device)
                realism_scores.append(device_score)
        
        # Physics validation (if physics engine is available)
        if self.physics_engine:
            physics_score = self.physics_engine.validate_physics_compliance(load_data, devices)
        else:
            physics_score = 85.0  # Default good score without full physics engine
        
        # Overall realism score
        device_avg = np.mean(realism_scores) if realism_scores else 50
        overall_score = 0.7 * device_avg + 0.3 * physics_score
        
        return overall_score
    
    def _calculate_device_realism_score(self, device_data: pd.Series, device_name: str) -> float:
        """Calculate enhanced realism score for device behavior."""
        
        score = 100.0
        config = self.device_configs.get(device_name, {})
        peak_power = config.get('peak_power', device_data.max())
        
        # Transition smoothness (enhanced)
        transitions = np.abs(np.diff(device_data))
        large_transitions = transitions > 0.15 * peak_power
        score -= large_transitions.sum() * 5
        
        # Power level distribution realism
        power_levels = device_data / peak_power
        level_hist, _ = np.histogram(power_levels, bins=10, range=(0, 1))
        level_distribution_score = 100 - np.std(level_hist) * 2
        score = 0.6 * score + 0.4 * level_distribution_score
        
        # Cycling behavior validation - now dynamic
        device_config = self.device_configs.get(device_name, {})
        usage_pattern = device_config.get('usage_pattern', 'scheduled')
        if (usage_pattern == 'continuous' or 
            'kühl' in device_name.lower() or 'kälte' in device_name.lower() or
            'heiz' in device_name.lower() or 'wärme' in device_name.lower()):
            cycling_score = self._validate_cycling_behavior(device_data, device_name)
            score = 0.8 * score + 0.2 * cycling_score
        
        # Physics compliance (if physics engine is available)
        if self.physics_engine:
            physics_compliance = self.physics_engine.check_device_physics(device_data, device_name)
            score = 0.7 * score + 0.3 * physics_compliance
        # Otherwise use basic physics validation
        else:
            physics_compliance = self._basic_physics_validation(device_data, device_name)
            score = 0.8 * score + 0.2 * physics_compliance
        
        return max(0, min(100, score))
    
    def _basic_physics_validation(self, device_data: pd.Series, device_name: str) -> float:
        """Basic physics validation without full physics engine."""
        # Check for realistic power transitions
        power_changes = device_data.diff().abs()
        max_change = device_data.max() * 0.3  # Max 30% change per interval
        smooth_transitions = (power_changes <= max_change).mean()
        
        # Check for realistic power levels (not always at max)
        avg_utilization = device_data.mean() / device_data.max() if device_data.max() > 0 else 0.5
        utilization_score = 1.0 - abs(avg_utilization - 0.6)  # Target 60% utilization
        
        return max(0, min(100, (smooth_transitions + max(utilization_score, 0)) * 50))

    def _validate_cycling_behavior(self, device_data: pd.Series, device_name: str) -> float:
        """Validate realistic cycling behavior for cyclical devices."""
        
        score = 100.0
        
        # Detect cycles
        power_binary = (device_data > device_data.mean()).astype(int)
        cycle_changes = np.diff(power_binary)
        
        if len(cycle_changes) > 0:
            # Check for reasonable cycle lengths
            on_periods = []
            off_periods = []
            current_state = power_binary[0]
            current_length = 1
            
            for change in cycle_changes:
                if change != 0:  # State change
                    if current_state == 1:
                        on_periods.append(current_length)
                    else:
                        off_periods.append(current_length)
                    current_state = 1 - current_state
                    current_length = 1
                else:
                    current_length += 1
            
            # Validate cycle lengths
            if on_periods:
                avg_on_time = np.mean(on_periods)
                if avg_on_time < 2 or avg_on_time > 20:  # 30 min to 5 hours
                    score -= 20
            
            if off_periods:
                avg_off_time = np.mean(off_periods)
                if avg_off_time < 1 or avg_off_time > 24:  # 15 min to 6 hours
                    score -= 20
        
        return max(0, score)
    
    def _calculate_learning_confidence(self) -> Dict[str, float]:
        """Calculate confidence scores for learned parameters."""
        
        confidence_scores = {}
        
        for device_name in self.device_configs.keys():
            # Base confidence on amount of data and consistency
            learning_data = self.learning_data.get(device_name, {})
            
            data_amount_score = min(1.0, len(learning_data.get('observations', [])) / 1000)
            consistency_score = 1.0 - learning_data.get('variation_coefficient', 0.5)
            physics_compliance_score = learning_data.get('physics_compliance', 0.5)
            
            overall_confidence = (
                0.4 * data_amount_score +
                0.3 * consistency_score +
                0.3 * physics_compliance_score
            )
            
            confidence_scores[device_name] = overall_confidence
        
        return confidence_scores


class IntelligentDeviceLearner:
    """Intelligent device discovery and parameter learning from load profiles."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.device_signatures = self._load_device_signatures()
    
    def _load_device_signatures(self) -> Dict[str, Dict]:
        """Load known device signatures for pattern matching."""
        return {
            'heater': {
                'power_range': (1000, 5000),
                'cycling_frequency': (0.1, 0.5),  # cycles per hour
                'temperature_correlation': (-0.7, -0.9),
                'seasonal_pattern': 'winter_dominant'
            },
            'air_conditioner': {
                'power_range': (1500, 6000),
                'cycling_frequency': (0.2, 0.8),
                'temperature_correlation': (0.6, 0.9),
                'seasonal_pattern': 'summer_dominant'
            },
            'refrigeration': {
                'power_range': (200, 1200),
                'cycling_frequency': (2, 6),
                'temperature_correlation': (0.1, 0.4),
                'seasonal_pattern': 'year_round'
            },
            'water_heater': {
                'power_range': (2000, 4000),
                'cycling_frequency': (0.5, 2),
                'temperature_correlation': (-0.2, -0.5),
                'seasonal_pattern': 'winter_bias'
            },
            'lighting': {
                'power_range': (100, 2000),
                'cycling_frequency': (0.05, 0.2),
                'temperature_correlation': (-0.1, 0.1),
                'seasonal_pattern': 'daylight_inverse'
            },
            'general_load': {
                'power_range': (500, 3000),
                'cycling_frequency': (0.1, 1),
                'temperature_correlation': (-0.2, 0.3),
                'seasonal_pattern': 'occupancy_dependent'
            }
        }
    
    def discover_devices_from_profile(self, load_profile: pd.DataFrame) -> Dict[str, Dict]:
        """Discover devices and their parameters from load profile analysis."""
        
        discovered_devices = {}
        
        # Use spectral analysis to identify different device signatures
        total_load = load_profile['total_power'] if 'total_power' in load_profile.columns else load_profile.iloc[:, 0]
        
        # Frequency domain analysis
        freq_components = self._analyze_frequency_components(total_load)
        
        # Power level clustering
        power_clusters = self._cluster_power_levels(total_load)
        
        # Temperature correlation analysis
        temp_correlations = self._analyze_temperature_correlations(load_profile)
        
        # Match discovered patterns to known device signatures
        for device_type, signature in self.device_signatures.items():
            match_score = self._calculate_signature_match({
                'frequency_components': freq_components,
                'power_clusters': power_clusters,
                'temp_correlations': temp_correlations
            }, signature)
            
            if match_score > 0.6:  # Confidence threshold
                device_params = self._extract_device_parameters(load_profile, device_type, signature)
                device_params['confidence'] = match_score
                discovered_devices[device_type] = device_params
        
        return discovered_devices
    
    def _analyze_frequency_components(self, load_data: pd.Series) -> Dict[str, float]:
        """Analyze frequency components to identify cycling patterns."""
        
        # FFT analysis
        fft = np.fft.fft(load_data.values)
        frequencies = np.fft.fftfreq(len(load_data), d=0.25)  # 15-min intervals
        power_spectrum = np.abs(fft) ** 2
        
        # Identify dominant frequencies
        dominant_freqs = frequencies[np.argsort(power_spectrum)[-10:]]
        
        return {
            'dominant_frequencies': dominant_freqs.tolist(),
            'spectral_peak_ratio': power_spectrum.max() / power_spectrum.mean(),
            'frequency_spread': np.std(dominant_freqs)
        }
    
    def _cluster_power_levels(self, load_data: pd.Series) -> Dict[str, Any]:
        """Cluster power levels to identify distinct operating modes."""
        
        # Reshape for clustering
        power_reshaped = load_data.values.reshape(-1, 1)
        
        # Try different numbers of clusters
        best_clusters = 2
        best_score = -np.inf
        
        for n_clusters in range(2, 8):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(power_reshaped)
                score = -kmeans.inertia_  # Negative because lower inertia is better
                
                if score > best_score:
                    best_score = score
                    best_clusters = n_clusters
            except:
                continue
        
        # Fit final clustering
        kmeans = KMeans(n_clusters=best_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(power_reshaped)
        cluster_centers = kmeans.cluster_centers_.flatten()
        
        return {
            'num_clusters': best_clusters,
            'cluster_centers': cluster_centers.tolist(),
            'cluster_separation': np.std(cluster_centers),
            'power_range': (load_data.min(), load_data.max())
        }
    
    def _analyze_temperature_correlations(self, load_profile: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlations with temperature and other weather variables."""
        
        correlations = {}
        
        if 'temperature' in load_profile.columns:
            total_load = load_profile['total_power'] if 'total_power' in load_profile.columns else load_profile.iloc[:, 0]
            correlations['temperature'] = stats.pearsonr(total_load, load_profile['temperature'])[0]
        
        if 'humidity' in load_profile.columns:
            total_load = load_profile['total_power'] if 'total_power' in load_profile.columns else load_profile.iloc[:, 0]
            correlations['humidity'] = stats.pearsonr(total_load, load_profile['humidity'])[0]
        
        return correlations
    
    def _calculate_signature_match(self, discovered_patterns: Dict, device_signature: Dict) -> float:
        """Calculate how well discovered patterns match a device signature."""
        
        match_score = 0.0
        total_weight = 0.0
        
        # Power range match
        if 'power_clusters' in discovered_patterns:
            power_range = discovered_patterns['power_clusters']['power_range']
            sig_range = device_signature['power_range']
            
            overlap = max(0, min(power_range[1], sig_range[1]) - max(power_range[0], sig_range[0]))
            total_range = max(power_range[1], sig_range[1]) - min(power_range[0], sig_range[0])
            power_match = overlap / total_range if total_range > 0 else 0
            
            match_score += power_match * 0.3
            total_weight += 0.3
        
        # Temperature correlation match
        if 'temp_correlations' in discovered_patterns and 'temperature' in discovered_patterns['temp_correlations']:
            observed_corr = discovered_patterns['temp_correlations']['temperature']
            expected_range = device_signature['temperature_correlation']
            
            if expected_range[0] <= observed_corr <= expected_range[1]:
                corr_match = 1.0
            else:
                # Calculate distance from expected range
                dist_to_range = min(abs(observed_corr - expected_range[0]), 
                                  abs(observed_corr - expected_range[1]))
                corr_match = max(0, 1.0 - dist_to_range)
            
            match_score += corr_match * 0.4
            total_weight += 0.4
        
        # Frequency match
        if 'frequency_components' in discovered_patterns:
            # This would be more complex in practice
            freq_match = 0.5  # Placeholder
            match_score += freq_match * 0.3
            total_weight += 0.3
        
        return match_score / total_weight if total_weight > 0 else 0.0
    
    def _extract_device_parameters(self, load_profile: pd.DataFrame, 
                                 device_type: str, signature: Dict) -> Dict[str, Any]:
        """Extract specific device parameters from load profile."""
        
        total_load = load_profile['total_power'] if 'total_power' in load_profile.columns else load_profile.iloc[:, 0]
        
        parameters = {
            'peak_power': float(total_load.quantile(0.95)),  # 95th percentile as peak
            'average_power': float(total_load.mean()),
            'capacity_factor': float(total_load.mean() / total_load.max()),
        }
        
        # Device-specific parameter extraction - now dynamic
        if ('temperature' in load_profile.columns and 
            ('heiz' in device_type.lower() or 'wärme' in device_type.lower() or
             'kühl' in device_type.lower() or 'kälte' in device_type.lower())):
            # Calculate temperature coefficient
            temp_coeff = self._calculate_temperature_coefficient(total_load, load_profile['temperature'])
            parameters['temp_coefficient'] = temp_coeff
        
        # Extract daily pattern
        daily_pattern = self._extract_daily_pattern(total_load)
        parameters['daily_pattern'] = daily_pattern
        
        return parameters
    
    def _calculate_temperature_coefficient(self, load_data: pd.Series, temperature: pd.Series) -> float:
        """Calculate temperature coefficient from load and temperature correlation."""
        
        # Linear regression to find temperature sensitivity
        from sklearn.linear_model import LinearRegression
        
        X = temperature.values.reshape(-1, 1)
        y = load_data.values
        
        reg = LinearRegression().fit(X, y)
        return float(reg.coef_[0])
    
    def _extract_daily_pattern(self, load_data: pd.Series) -> List[float]:
        """Extract average daily pattern from load data."""
        
        # Assuming 15-minute intervals, create 96-point daily pattern
        patterns = []
        
        # Group by time of day
        if hasattr(load_data.index, 'time'):
            daily_groups = load_data.groupby(load_data.index.time)
            pattern_values = []
            
            for i in range(96):  # 15-minute intervals
                hour = i // 4
                minute = (i % 4) * 15
                time_key = pd.Timestamp(f"2000-01-01 {hour:02d}:{minute:02d}:00").time()
                
                if time_key in daily_groups.groups:
                    avg_value = daily_groups.get_group(time_key).mean()
                else:
                    # Interpolate if missing
                    avg_value = load_data.mean()
                
                pattern_values.append(float(avg_value))
            
            # Normalize to 0-1 range
            max_val = max(pattern_values)
            if max_val > 0:
                pattern_values = [v / max_val for v in pattern_values]
        else:
            # Fallback: generic pattern
            pattern_values = [0.5] * 96
        
        return pattern_values


class ThermalDevice:
    """Physics-based thermal device model (heaters, AC, water heaters)."""
    
    def __init__(self, device_type: str, heating: bool = False, cooling: bool = False):
        self.device_type = device_type
        self.heating = heating
        self.cooling = cooling
        self.thermal_params = self._get_thermal_parameters()
        
    def _get_thermal_parameters(self) -> Dict:
        """Get device-specific thermal parameters."""
        
        params = {
            'heater': {
                'thermal_mass': 0.8,
                'response_time': 12,  # 15-min intervals to reach 63% of target
                'hysteresis': 2.0,
                'efficiency_curve': 'inverse_temp'
            },
            'air_conditioner': {
                'thermal_mass': 0.6,
                'response_time': 8,
                'hysteresis': 1.5,
                'efficiency_curve': 'temp_dependent'
            },
            'water_heater': {
                'thermal_mass': 0.9,
                'response_time': 16,
                'hysteresis': 3.0,
                'efficiency_curve': 'constant'
            }
        }
        
        return params.get(self.device_type, params['heater'])
    
    def calculate_load(self, weather_data: pd.DataFrame, peak_power: float,
                      daily_pattern: np.ndarray, config: Dict, device_state: Dict) -> np.ndarray:
        """Calculate realistic thermal device load."""
        
        temperatures = weather_data['temperature'].values
        time_indices = np.arange(len(weather_data))
        
        # Initialize
        power_output = np.zeros_like(temperatures, dtype=float)
        thermal_state = device_state.get('thermal_state', 0.0)
        
        comfort_temp = config.get('comfort_temp', 20)
        temp_coefficient = config.get('temp_coefficient', 0)
        
        for i, temp in enumerate(temperatures):
            # Calculate demand based on physics
            temp_difference = abs(temp - comfort_temp)
            
            # Thermal load calculation
            if self.device_type == 'heater':
                thermal_demand = max(0, comfort_temp - temp) / 10  # Normalized
            elif self.device_type == 'air_conditioner':
                thermal_demand = max(0, temp - comfort_temp) / 10
            else:  # water_heater
                # Water heater demand based on usage pattern and ambient temperature
                usage_demand = daily_pattern[(i * 4) % 96]  # Map to 15-min intervals
                thermal_demand = usage_demand * (1 + (comfort_temp - temp) * 0.02)
            
            # Apply thermal inertia
            target_thermal_state = thermal_demand
            response_rate = 1.0 / self.thermal_params['response_time']
            thermal_state += (target_thermal_state - thermal_state) * response_rate
            
            # Convert thermal state to power with daily pattern influence
            interval_index = (i * 4) % 96
            pattern_factor = daily_pattern[interval_index]
            
            # Calculate power output
            base_power = thermal_state * peak_power * pattern_factor
            
            # Apply efficiency curve
            efficiency = self._calculate_efficiency(temp, comfort_temp)
            
            power_output[i] = base_power * efficiency
        
        # Update device state
        device_state['thermal_state'] = thermal_state
        
        # Apply final smoothing for realism
        return self._apply_thermal_smoothing(power_output)
    
    def _calculate_efficiency(self, current_temp: float, comfort_temp: float) -> float:
        """Calculate device efficiency based on operating conditions."""
        
        temp_diff = abs(current_temp - comfort_temp)
        
        if self.thermal_params['efficiency_curve'] == 'inverse_temp':
            # Heaters are less efficient in extreme cold
            return max(0.7, 1.0 - temp_diff * 0.02)
        elif self.thermal_params['efficiency_curve'] == 'temp_dependent':
            # AC efficiency varies with temperature difference
            return max(0.6, 1.0 - temp_diff * 0.015)
        else:
            return 1.0  # Constant efficiency
    
    def _apply_thermal_smoothing(self, power_data: np.ndarray) -> np.ndarray:
        """Apply thermal smoothing for realistic response."""
        
        # Use exponential smoothing for thermal inertia
        smoothed = np.zeros_like(power_data)
        alpha = 0.3  # Smoothing factor
        
        smoothed[0] = power_data[0]
        for i in range(1, len(power_data)):
            smoothed[i] = alpha * power_data[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed


class CyclicDevice:
    """Model for devices with cyclical operation (refrigeration, heat pumps)."""
    
    def __init__(self, device_type: str):
        self.device_type = device_type
        self.cycle_params = self._get_cycle_parameters()
        
    def _get_cycle_parameters(self) -> Dict:
        """Get device-specific cycle parameters."""
        
        return {
            'on_time_range': (4, 8),      # 1-2 hours
            'off_time_range': (8, 16),    # 2-4 hours
            'startup_time': 2,            # 30 minutes
            'shutdown_time': 1,           # 15 minutes
            'efficiency_variation': 0.1
        }
    
    def calculate_load(self, weather_data: pd.DataFrame, peak_power: float,
                      daily_pattern: np.ndarray, config: Dict, device_state: Dict) -> np.ndarray:
        """Calculate realistic cyclical device load."""
        
        n_points = len(weather_data)
        power_output = np.zeros(n_points, dtype=float)
        
        # Initialize cycle state
        cycle_position = device_state.get('cycle_position', 0)
        is_on = device_state.get('is_on', False)
        cycle_timer = device_state.get('cycle_timer', 0)
        
        # Generate realistic cycles
        on_time = np.random.randint(*self.cycle_params['on_time_range'])
        off_time = np.random.randint(*self.cycle_params['off_time_range'])
        
        for i in range(n_points):
            # Determine cycle state
            if cycle_timer <= 0:
                if is_on:
                    # Switch to off
                    is_on = False
                    cycle_timer = off_time + np.random.randint(-2, 3)
                else:
                    # Switch to on
                    is_on = True
                    cycle_timer = on_time + np.random.randint(-2, 3)
            
            cycle_timer -= 1
            
            # Calculate power based on cycle state and daily pattern
            interval_index = (i * 4) % 96
            pattern_factor = daily_pattern[interval_index]
            
            if is_on:
                # During startup/shutdown
                startup_time = self.cycle_params['startup_time']
                shutdown_time = self.cycle_params['shutdown_time']
                
                if cycle_timer > (on_time - startup_time):
                    # Startup phase
                    startup_factor = (on_time - cycle_timer) / startup_time
                    power_factor = min(1.0, startup_factor)
                elif cycle_timer < shutdown_time:
                    # Shutdown phase
                    shutdown_factor = cycle_timer / shutdown_time
                    power_factor = max(0.1, shutdown_factor)
                else:
                    # Full operation
                    power_factor = 1.0
                
                base_power = peak_power * pattern_factor * power_factor
                
                # Add compressor efficiency variations
                efficiency = 1.0 + np.random.normal(0, self.cycle_params['efficiency_variation'])
                power_output[i] = base_power * efficiency
                
            else:
                # Off cycle - minimal standby power
                power_output[i] = peak_power * 0.02  # 2% standby power
        
        # Update device state
        device_state.update({
            'cycle_position': cycle_position,
            'is_on': is_on,
            'cycle_timer': cycle_timer
        })
        
        return power_output


class AdaptiveDevice:
    """Model for adaptive devices (general loads, smart devices)."""
    
    def __init__(self, device_name: str):
        self.device_name = device_name
    
    def calculate_load(self, weather_data: pd.DataFrame, peak_power: float,
                      daily_pattern: np.ndarray, config: Dict, device_state: Dict) -> np.ndarray:
        """Calculate adaptive device load with learning behavior."""
        
        n_points = len(weather_data)
        power_output = np.zeros(n_points, dtype=float)
        
        # Adaptive parameters
        adaptation_rate = 0.02
        learning_memory = device_state.get('learning_memory', [])
        
        for i in range(n_points):
            # Base load from daily pattern
            interval_index = (i * 4) % 96
            base_factor = daily_pattern[interval_index]
            
            # Adaptive adjustments based on conditions
            hour = (interval_index // 4) % 24
            
            # Weekend vs weekday adaptation
            is_weekend = weather_data.index[i].weekday() >= 5
            weekend_factor = 1.2 if is_weekend else 1.0
            
            # Weather-based adaptation
            temp = weather_data['temperature'].iloc[i]
            weather_factor = 1.0 + (temp - 20) * 0.01  # Slight temperature dependence
            
            # Calculate adaptive power
            adaptive_factor = base_factor * weekend_factor * weather_factor
            
            # Add realistic variations
            variation = np.random.normal(1.0, 0.05)
            
            power_output[i] = peak_power * adaptive_factor * variation
            
            # Store for learning (simplified)
            if len(learning_memory) < 100:
                learning_memory.append(adaptive_factor)
            else:
                learning_memory[i % 100] = adaptive_factor
        
        device_state['learning_memory'] = learning_memory
        
        # Apply smoothing for realistic behavior
        return self._apply_adaptive_smoothing(power_output)
    
    def _apply_adaptive_smoothing(self, power_data: np.ndarray) -> np.ndarray:
        """Apply smoothing appropriate for adaptive devices."""
        
        # Light smoothing to maintain responsiveness
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(power_data, sigma=1.0)


class ResponsiveDevice:
    """Model for quickly responsive devices (lighting, electronics)."""
    
    def __init__(self, device_name: str):
        self.device_name = device_name
    
    def calculate_load(self, weather_data: pd.DataFrame, peak_power: float,
                      daily_pattern: np.ndarray, config: Dict, device_state: Dict) -> np.ndarray:
        """Calculate responsive device load."""
        
        n_points = len(weather_data)
        power_output = np.zeros(n_points, dtype=float)
        
        for i in range(n_points):
            interval_index = (i * 4) % 96
            hour = (interval_index // 4) % 24
            
            # Base pattern
            base_factor = daily_pattern[interval_index]
            
            # Seasonal daylight adjustment for lighting
            month = weather_data.index[i].month
            daylight_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 6) / 12)
            
            # Occupancy-based variations
            if 6 <= hour <= 8 or 17 <= hour <= 23:
                occupancy_factor = 1.0
            elif 9 <= hour <= 16:
                occupancy_factor = 0.3  # Lower during day
            else:
                occupancy_factor = 0.1  # Minimal at night
            
            # Quick response with small random variations
            response_factor = base_factor * daylight_factor * occupancy_factor
            variation = np.random.normal(1.0, 0.03)
            
            power_output[i] = peak_power * response_factor * variation
        
        # Minimal smoothing to maintain responsiveness
        alpha = 0.1
        for i in range(1, len(power_output)):
            power_output[i] = alpha * power_output[i] + (1 - alpha) * power_output[i-1]
        
        return power_output