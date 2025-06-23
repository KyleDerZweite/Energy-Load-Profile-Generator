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
try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
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
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
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


class MultiScalePatternLearner:
    """Multi-scale pattern learning for device optimization with genetic algorithms."""
    
    def __init__(self):
        self.patterns = {}
        self.logger = logging.getLogger(__name__)
        self.target_data = None
        self.device_configs = {}
        
    def set_target_data(self, target_load_profile: pd.DataFrame):
        """Set the target load profile to match during optimization."""
        if 'Value' in target_load_profile.columns:
            self.target_data = target_load_profile['Value'].values
        elif 'total_power' in target_load_profile.columns:
            # Convert from W to kW if needed
            self.target_data = target_load_profile['total_power'].values
            if self.target_data.mean() > 1000:  # Likely in Watts
                self.target_data = self.target_data / 1000
        else:
            # Use first numeric column
            numeric_cols = target_load_profile.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.target_data = target_load_profile[numeric_cols[0]].values
        
        self.logger.info(f"Target data set: {len(self.target_data)} points, range {self.target_data.min():.1f}-{self.target_data.max():.1f} kW")
    
    def learn_patterns(self, device_configs: Dict, target_profile: pd.DataFrame = None, 
                      weather_data: pd.DataFrame = None) -> Dict:
        """Learn optimal device patterns using genetic algorithm optimization."""
        
        if target_profile is not None:
            self.set_target_data(target_profile)
        
        if self.target_data is None:
            self.logger.error("No target data set for pattern learning")
            return {'learned': False, 'error': 'No target data'}
        
        self.device_configs = device_configs
        
        # Extract target daily pattern (average over all days)
        target_daily_pattern = self._extract_target_daily_pattern()
        target_stats = self._calculate_target_statistics()
        
        # Initialize genetic algorithm for device pattern optimization
        optimized_patterns = self._genetic_algorithm_optimization(
            target_daily_pattern, target_stats, weather_data
        )
        
        # Validate optimized patterns with physics engine
        physics_validation = self._validate_physics_compliance(optimized_patterns)
        
        # Calculate learning confidence
        confidence_scores = self._calculate_learning_confidence(optimized_patterns, physics_validation)
        
        results = {
            'learned': True,
            'patterns': optimized_patterns,
            'target_daily_pattern': target_daily_pattern.tolist(),
            'target_statistics': target_stats,
            'physics_validation': physics_validation,
            'confidence_scores': confidence_scores,
            'optimization_summary': {
                'devices_optimized': len(optimized_patterns),
                'target_match_score': self._calculate_target_match_score(optimized_patterns),
                'physics_compliance_score': physics_validation.get('overall_score', 0),
                'pattern_realism_score': self._calculate_pattern_realism_score(optimized_patterns)
            }
        }
        
        self.logger.info(f"Pattern learning completed: {len(optimized_patterns)} devices optimized")
        return results
    
    def _extract_target_daily_pattern(self) -> np.ndarray:
        """Extract average daily pattern from target data."""
        
        # Assume 15-minute intervals (96 per day)
        intervals_per_day = 96
        num_days = len(self.target_data) // intervals_per_day
        
        if num_days == 0:
            # Less than a day of data, repeat to fill 96 intervals
            repeated_data = np.tile(self.target_data, intervals_per_day // len(self.target_data) + 1)
            return repeated_data[:intervals_per_day]
        
        # Reshape into days and calculate average daily pattern
        reshaped_data = self.target_data[:num_days * intervals_per_day].reshape(num_days, intervals_per_day)
        daily_pattern = np.mean(reshaped_data, axis=0)
        
        # Ensure pattern loops properly (first and last values should be similar)
        self._ensure_pattern_loops(daily_pattern)
        
        return daily_pattern
    
    def _ensure_pattern_loops(self, pattern: np.ndarray) -> None:
        """Ensure daily pattern loops properly by smoothing start/end transition."""
        
        # Smooth the transition between end and start
        window_size = 4  # 1 hour on each end
        
        start_values = pattern[:window_size]
        end_values = pattern[-window_size:]
        
        # Calculate target transition value (average of start and end)
        transition_target = (start_values.mean() + end_values.mean()) / 2
        
        # Apply smooth transition
        for i in range(window_size):
            weight = i / (window_size - 1)  # 0 to 1
            pattern[i] = (1 - weight) * transition_target + weight * pattern[i]
            pattern[-(i+1)] = (1 - weight) * transition_target + weight * pattern[-(i+1)]
    
    def _calculate_target_statistics(self) -> Dict:
        """Calculate key statistics from target data."""
        
        return {
            'total_energy_kwh': float(self.target_data.sum() * 0.25),  # 15-min intervals
            'average_power_kw': float(self.target_data.mean()),
            'peak_power_kw': float(self.target_data.max()),
            'min_power_kw': float(self.target_data.min()),
            'load_factor': float(self.target_data.mean() / self.target_data.max()),
            'variability': float(self.target_data.std() / self.target_data.mean()),
            'daily_peak_to_avg_ratio': float(self._extract_target_daily_pattern().max() / self._extract_target_daily_pattern().mean())
        }
    
    def _genetic_algorithm_optimization(self, target_pattern: np.ndarray, 
                                      target_stats: Dict, weather_data: pd.DataFrame = None) -> Dict:
        """Genetic algorithm optimization of device patterns."""
        
        from scipy.optimize import differential_evolution
        
        optimized_patterns = {}
        enabled_devices = [name for name, config in self.device_configs.items() if config.get('enabled', True)]
        
        self.logger.info(f"Starting genetic optimization for {len(enabled_devices)} devices")
        
        for device_name in enabled_devices:
            self.logger.info(f"Optimizing patterns for {device_name}")
            
            # Define optimization bounds for daily pattern (96 values between 0 and 1)
            bounds = [(0.0, 1.0) for _ in range(96)]
            
            # Define objective function for this device
            def objective(pattern_genes):
                return self._evaluate_device_pattern(device_name, pattern_genes, target_pattern, target_stats)
            
            # Run differential evolution (genetic algorithm variant)
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=50,  # Reasonable number of iterations
                popsize=15,  # Population size
                atol=1e-6,
                tol=1e-6,
                workers=1  # Single threaded for stability
            )
            
            # Store optimized pattern
            optimized_pattern = result.x.tolist()
            
            # Ensure pattern is realistic and loops properly
            optimized_pattern = self._post_process_pattern(optimized_pattern, device_name)
            
            optimized_patterns[device_name] = {
                'daily_pattern': optimized_pattern,
                'optimization_score': float(result.fun),
                'pattern_energy': float(np.array(optimized_pattern).sum() * 0.25),
                'pattern_peak': float(max(optimized_pattern)),
                'pattern_avg': float(np.mean(optimized_pattern)),
                'convergence_success': bool(result.success)
            }
            
            self.logger.info(f"{device_name} optimization: score={result.fun:.3f}, success={result.success}")
        
        return optimized_patterns
    
    def _evaluate_device_pattern(self, device_name: str, pattern_genes: np.ndarray,
                               target_pattern: np.ndarray, target_stats: Dict) -> float:
        """Evaluate a device pattern using multi-objective scoring."""
        
        # Convert genes to proper pattern
        daily_pattern = np.array(pattern_genes)
        
        # Get device configuration
        device_config = self.device_configs.get(device_name, {})
        peak_power_w = device_config.get('peak_power', 1000)
        quantity = device_config.get('quantity', 1)
        
        # Scale pattern to power values (kW)
        device_power_kw = daily_pattern * peak_power_w * quantity / 1000
        
        # Physics-based constraints scoring
        physics_score = self._evaluate_physics_constraints(daily_pattern, device_name)
        
        # Pattern realism scoring
        realism_score = self._evaluate_pattern_realism(daily_pattern, device_name)
        
        # Target matching scoring (contribution to total load)
        target_match_score = self._evaluate_target_contribution(device_power_kw, target_pattern, device_name)
        
        # Pattern smoothness scoring
        smoothness_score = self._evaluate_pattern_smoothness(daily_pattern)
        
        # Combined objective (minimize, so negate good scores)
        total_score = (
            -0.35 * target_match_score +    # Most important: match target
            -0.25 * physics_score +         # Physics compliance
            -0.20 * realism_score +         # Pattern realism
            -0.20 * smoothness_score        # Smooth transitions
        )
        
        return total_score
    
    def _evaluate_physics_constraints(self, pattern: np.ndarray, device_name: str) -> float:
        """Evaluate physics-based constraints for the pattern."""
        
        score = 100.0
        
        # Check for realistic power transitions
        transitions = np.abs(np.diff(pattern))
        max_transition = transitions.max()
        
        # Penalize large transitions (devices can't change power instantly)
        if max_transition > 0.3:  # More than 30% change per 15-min interval
            score -= (max_transition - 0.3) * 200
        
        # Check for device-specific physics
        if 'heiz' in device_name.lower() or 'heat' in device_name.lower():
            # Heating devices should have thermal inertia
            avg_transition = transitions.mean()
            if avg_transition > 0.1:  # Too jerky for thermal device
                score -= (avg_transition - 0.1) * 100
        
        elif 'kühl' in device_name.lower() or 'cool' in device_name.lower() or 'air_cond' in device_name.lower():
            # Cooling devices should cycle reasonably
            on_periods = self._detect_on_periods(pattern)
            if len(on_periods) > 0:
                avg_on_time = np.mean(on_periods)
                if avg_on_time < 2:  # Less than 30 minutes
                    score -= 50
                elif avg_on_time > 20:  # More than 5 hours
                    score -= 30
        
        elif 'light' in device_name.lower() or 'beleucht' in device_name.lower():
            # Lighting should follow occupancy patterns
            night_avg = pattern[0:24].mean()  # Midnight to 6 AM
            day_avg = pattern[36:72].mean()   # 9 AM to 6 PM
            
            if night_avg > day_avg * 0.5:  # Night too bright
                score -= 40
        
        # Check for early morning unrealistic peaks (3-4 AM)
        early_morning = pattern[12:16].mean()  # 3-4 AM
        morning_peak = pattern[24:32].mean()   # 6-8 AM
        
        if early_morning > morning_peak * 0.8:  # Early morning too high
            score -= 60
        
        return max(0, min(100, score))
    
    def _detect_on_periods(self, pattern: np.ndarray) -> List[int]:
        """Detect on periods in a cycling pattern."""
        
        threshold = pattern.mean()
        is_on = pattern > threshold
        
        on_periods = []
        current_period = 0
        
        for state in is_on:
            if state:  # On
                current_period += 1
            else:  # Off
                if current_period > 0:
                    on_periods.append(current_period)
                    current_period = 0
        
        if current_period > 0:
            on_periods.append(current_period)
        
        return on_periods
    
    def _evaluate_pattern_realism(self, pattern: np.ndarray, device_name: str) -> float:
        """Evaluate the realism of the pattern."""
        
        score = 100.0
        
        # Pattern should not be too flat
        pattern_std = pattern.std()
        if pattern_std < 0.05:  # Too constant
            score -= 40
        
        # Pattern should not be too spiky
        spikiness = np.sum(np.abs(np.diff(pattern, 2)))  # Second derivative sum
        if spikiness > 10:
            score -= 30
        
        # Pattern should loop properly
        start_end_diff = abs(pattern[0] - pattern[-1])
        if start_end_diff > pattern_std * 0.5:
            score -= 20
        
        # Device-specific realism checks
        if 'server' in device_name.lower() or 'network' in device_name.lower():
            # Servers should be relatively constant
            if pattern_std > 0.3:
                score -= 30
        
        return max(0, min(100, score))
    
    def _evaluate_target_contribution(self, device_power_kw: np.ndarray, target_pattern: np.ndarray, device_name: str) -> float:
        """Evaluate how well device pattern contributes to target matching."""
        
        # For now, use correlation as a proxy for good contribution
        # In a full implementation, this would simulate the total load with this device
        correlation = np.corrcoef(device_power_kw, target_pattern[:len(device_power_kw)])[0, 1]
        
        if np.isnan(correlation):
            correlation = 0
        
        # Convert correlation to 0-100 score
        score = max(0, correlation * 100)
        
        return score
    
    def _evaluate_pattern_smoothness(self, pattern: np.ndarray) -> float:
        """Evaluate pattern smoothness."""
        
        # Calculate smoothness based on transitions
        transitions = np.abs(np.diff(pattern))
        smoothness = 100 - (transitions.mean() * 500)  # Scale to 0-100
        
        return max(0, min(100, smoothness))
    
    def _post_process_pattern(self, pattern: List[float], device_name: str) -> List[float]:
        """Post-process optimized pattern for realism and constraints."""
        
        pattern_array = np.array(pattern)
        
        # Ensure pattern loops properly
        self._ensure_pattern_loops(pattern_array)
        
        # Apply device-specific constraints
        if 'light' in device_name.lower():
            # Lighting should have clear day/night pattern
            for i in range(96):
                hour = (i // 4) % 24
                if 1 <= hour <= 5:  # 1-5 AM should be very low
                    pattern_array[i] = min(pattern_array[i], 0.1)
                elif 22 <= hour <= 23:  # 10-11 PM should be decreasing
                    pattern_array[i] = min(pattern_array[i], 0.3)
        
        # Apply smoothing to prevent unrealistic transitions
        from scipy.ndimage import gaussian_filter1d
        smoothed_pattern = gaussian_filter1d(pattern_array, sigma=1.0)
        
        # Blend original and smoothed (keep some optimization but add realism)
        final_pattern = 0.7 * smoothed_pattern + 0.3 * pattern_array
        
        # Ensure values are in valid range
        final_pattern = np.clip(final_pattern, 0.0, 1.0)
        
        return final_pattern.tolist()
    
    def _validate_physics_compliance(self, patterns: Dict) -> Dict:
        """Validate all patterns against physics constraints."""
        
        validation_results = {}
        total_score = 0
        
        for device_name, pattern_data in patterns.items():
            pattern = np.array(pattern_data['daily_pattern'])
            physics_score = self._evaluate_physics_constraints(pattern, device_name)
            
            validation_results[device_name] = {
                'physics_score': physics_score,
                'compliant': physics_score > 70,  # Threshold for compliance
                'issues': self._identify_physics_issues(pattern, device_name)
            }
            
            total_score += physics_score
        
        validation_results['overall_score'] = total_score / len(patterns) if patterns else 0
        validation_results['overall_compliant'] = validation_results['overall_score'] > 75
        
        return validation_results
    
    def _identify_physics_issues(self, pattern: np.ndarray, device_name: str) -> List[str]:
        """Identify specific physics issues with a pattern."""
        
        issues = []
        
        # Check transitions
        transitions = np.abs(np.diff(pattern))
        if transitions.max() > 0.3:
            issues.append("Large power transitions (>30% per 15min)")
        
        # Check early morning peaks
        early_morning = pattern[12:16].mean()
        morning = pattern[24:32].mean()
        if early_morning > morning * 0.8:
            issues.append("Unrealistic 3-4 AM peak")
        
        # Check looping
        if abs(pattern[0] - pattern[-1]) > pattern.std() * 0.5:
            issues.append("Pattern doesn't loop properly")
        
        return issues
    
    def _calculate_learning_confidence(self, patterns: Dict, physics_validation: Dict) -> Dict:
        """Calculate confidence scores for learned patterns."""
        
        confidence_scores = {}
        
        for device_name, pattern_data in patterns.items():
            # Base confidence on optimization success and physics compliance
            opt_success = pattern_data.get('convergence_success', False)
            physics_score = physics_validation.get(device_name, {}).get('physics_score', 0)
            
            # Calculate confidence
            confidence = 0
            if opt_success:
                confidence += 40
            confidence += physics_score * 0.6  # Physics score weighted
            
            confidence_scores[device_name] = max(0, min(100, confidence))
        
        return confidence_scores
    
    def _calculate_target_match_score(self, patterns: Dict) -> float:
        """Calculate how well optimized patterns would match target."""
        
        # This is a simplified calculation
        # In full implementation, would simulate total load and compare
        
        total_score = 0
        for device_name, pattern_data in patterns.items():
            # Use optimization score as proxy
            opt_score = pattern_data.get('optimization_score', 0)
            # Convert to positive score (optimization minimizes)
            score = max(0, 100 + opt_score * 10)  # Rough conversion
            total_score += score
        
        return total_score / len(patterns) if patterns else 0
    
    def _calculate_pattern_realism_score(self, patterns: Dict) -> float:
        """Calculate overall pattern realism score."""
        
        total_score = 0
        
        for device_name, pattern_data in patterns.items():
            pattern = np.array(pattern_data['daily_pattern'])
            realism_score = self._evaluate_pattern_realism(pattern, device_name)
            total_score += realism_score
        
        return total_score / len(patterns) if patterns else 0


class PhysicsRealismEngine:
    """Physics-based realism engine for comprehensive pattern validation."""
    
    def __init__(self):
        self.constraints = {
            'max_power_change_rate': 0.25,  # Max 25% per 15-min interval
            'min_cycling_time': 2,          # Minimum 30 minutes per cycle
            'thermal_time_constant': 4,     # Thermal devices: 1 hour time constant
            'early_morning_limit': 0.3      # 3-4 AM should be <30% of daily peak
        }
        self.logger = logging.getLogger(__name__)
    
    def validate_patterns(self, patterns: Dict, device_configs: Dict = None) -> Dict:
        """Comprehensive physics validation of device patterns."""
        
        validation_results = {
            'devices': {},
            'overall_score': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        total_score = 0
        device_count = 0
        
        for device_name, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and 'daily_pattern' in pattern_data:
                pattern = np.array(pattern_data['daily_pattern'])
            else:
                pattern = np.array(pattern_data)
            
            device_validation = self._validate_device_pattern(device_name, pattern, device_configs)
            validation_results['devices'][device_name] = device_validation
            
            total_score += device_validation['score']
            device_count += 1
            
            # Collect critical issues
            if device_validation['score'] < 50:
                validation_results['critical_issues'].extend(device_validation['issues'])
            elif device_validation['score'] < 75:
                validation_results['warnings'].extend(device_validation['issues'])
        
        validation_results['overall_score'] = total_score / device_count if device_count > 0 else 0
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        self.logger.info(f"Physics validation completed: {validation_results['overall_score']:.1f}/100")
        return validation_results
    
    def _validate_device_pattern(self, device_name: str, pattern: np.ndarray, device_configs: Dict = None) -> Dict:
        """Validate a single device pattern against physics constraints."""
        
        validation = {
            'score': 100.0,
            'issues': [],
            'warnings': [],
            'physics_compliance': {}
        }
        
        # Power transition validation
        transitions = np.abs(np.diff(pattern))
        max_transition = transitions.max()
        avg_transition = transitions.mean()
        
        if max_transition > self.constraints['max_power_change_rate']:
            validation['score'] -= 30
            validation['issues'].append(f"Excessive power changes: {max_transition:.2f} > {self.constraints['max_power_change_rate']}")
        
        validation['physics_compliance']['power_transitions'] = {
            'max_change_rate': float(max_transition),
            'avg_change_rate': float(avg_transition),
            'compliant': max_transition <= self.constraints['max_power_change_rate']
        }
        
        # Early morning peak validation
        early_morning_avg = pattern[12:16].mean()  # 3-4 AM
        daily_peak = pattern.max()
        early_morning_ratio = early_morning_avg / daily_peak if daily_peak > 0 else 0
        
        if early_morning_ratio > self.constraints['early_morning_limit']:
            validation['score'] -= 40
            validation['issues'].append(f"Unrealistic 3-4 AM peak: {early_morning_ratio:.2f} > {self.constraints['early_morning_limit']}")
        
        validation['physics_compliance']['early_morning'] = {
            'peak_ratio': float(early_morning_ratio),
            'compliant': early_morning_ratio <= self.constraints['early_morning_limit']
        }
        
        # Pattern looping validation
        start_end_diff = abs(pattern[0] - pattern[-1])
        pattern_std = pattern.std()
        loops_properly = start_end_diff < pattern_std * 0.3
        
        if not loops_properly:
            validation['score'] -= 25
            validation['issues'].append(f"Pattern doesn't loop: start/end difference {start_end_diff:.3f}")
        
        validation['physics_compliance']['pattern_looping'] = {
            'start_end_difference': float(start_end_diff),
            'loops_properly': loops_properly
        }
        
        # Device-specific physics validation
        device_specific_validation = self._validate_device_specific_physics(device_name, pattern)
        validation['score'] += device_specific_validation['score_adjustment']
        validation['issues'].extend(device_specific_validation['issues'])
        validation['warnings'].extend(device_specific_validation['warnings'])
        validation['physics_compliance'].update(device_specific_validation['compliance'])
        
        # Ensure score is in valid range
        validation['score'] = max(0, min(100, validation['score']))
        
        return validation
    
    def _validate_device_specific_physics(self, device_name: str, pattern: np.ndarray) -> Dict:
        """Validate device-specific physics constraints."""
        
        result = {
            'score_adjustment': 0,
            'issues': [],
            'warnings': [],
            'compliance': {}
        }
        
        device_type = self._classify_device_type(device_name)
        
        if device_type == 'thermal':
            # Thermal devices should have smooth transitions due to thermal inertia
            smoothness = self._calculate_smoothness(pattern)
            if smoothness < 0.7:  # Less than 70% smooth
                result['score_adjustment'] -= 15
                result['warnings'].append(f"Thermal device not smooth enough: {smoothness:.2f}")
            
            result['compliance']['thermal_smoothness'] = {
                'smoothness_factor': float(smoothness),
                'compliant': smoothness >= 0.7
            }
        
        elif device_type == 'cyclic':
            # Cyclic devices (refrigeration, heat pumps) should have reasonable cycles
            cycle_analysis = self._analyze_cycles(pattern)
            
            if cycle_analysis['avg_on_time'] < self.constraints['min_cycling_time']:
                result['score_adjustment'] -= 20
                result['issues'].append(f"Cycles too short: {cycle_analysis['avg_on_time']:.1f} < {self.constraints['min_cycling_time']}")
            
            result['compliance']['cycling_behavior'] = cycle_analysis
        
        elif device_type == 'lighting':
            # Lighting should follow daylight/occupancy patterns
            lighting_validation = self._validate_lighting_pattern(pattern)
            result['score_adjustment'] += lighting_validation['score_adjustment']
            result['issues'].extend(lighting_validation['issues'])
            result['compliance']['lighting_behavior'] = lighting_validation['compliance']
        
        return result
    
    def _classify_device_type(self, device_name: str) -> str:
        """Classify device type for physics validation."""
        
        name_lower = device_name.lower()
        
        if any(term in name_lower for term in ['heiz', 'heat', 'wärme', 'kühl', 'cool', 'air_cond', 'hvac']):
            return 'thermal'
        elif any(term in name_lower for term in ['refrig', 'kälte', 'pump', 'compressor']):
            return 'cyclic'
        elif any(term in name_lower for term in ['light', 'beleucht', 'lamp']):
            return 'lighting'
        elif any(term in name_lower for term in ['server', 'computer', 'network']):
            return 'electronic'
        else:
            return 'general'
    
    def _calculate_smoothness(self, pattern: np.ndarray) -> float:
        """Calculate pattern smoothness (0-1, higher is smoother)."""
        
        if len(pattern) < 2:
            return 1.0
        
        # Calculate normalized variation
        transitions = np.abs(np.diff(pattern))
        max_possible_variation = pattern.max() - pattern.min()
        
        if max_possible_variation == 0:
            return 1.0  # Constant pattern is perfectly smooth
        
        normalized_variation = transitions.sum() / (len(transitions) * max_possible_variation)
        smoothness = max(0, 1 - normalized_variation)
        
        return smoothness
    
    def _analyze_cycles(self, pattern: np.ndarray) -> Dict:
        """Analyze cycling behavior of the pattern."""
        
        threshold = pattern.mean()
        is_on = pattern > threshold
        
        # Find on and off periods
        on_periods = []
        off_periods = []
        current_on_period = 0
        current_off_period = 0
        
        for state in is_on:
            if state:  # On
                if current_off_period > 0:
                    off_periods.append(current_off_period)
                    current_off_period = 0
                current_on_period += 1
            else:  # Off
                if current_on_period > 0:
                    on_periods.append(current_on_period)
                    current_on_period = 0
                current_off_period += 1
        
        # Handle final period
        if current_on_period > 0:
            on_periods.append(current_on_period)
        if current_off_period > 0:
            off_periods.append(current_off_period)
        
        return {
            'num_cycles': len(on_periods),
            'avg_on_time': np.mean(on_periods) if on_periods else 0,
            'avg_off_time': np.mean(off_periods) if off_periods else 0,
            'on_periods': on_periods,
            'off_periods': off_periods,
            'compliant': (np.mean(on_periods) if on_periods else 0) >= self.constraints['min_cycling_time']
        }
    
    def _validate_lighting_pattern(self, pattern: np.ndarray) -> Dict:
        """Validate lighting-specific patterns."""
        
        result = {
            'score_adjustment': 0,
            'issues': [],
            'compliance': {}
        }
        
        # Check day/night pattern
        night_avg = pattern[0:24].mean()    # Midnight to 6 AM
        day_avg = pattern[36:72].mean()     # 9 AM to 6 PM
        evening_avg = pattern[72:84].mean() # 6 PM to 9 PM
        
        # Night should be lowest
        if night_avg > day_avg * 0.3:
            result['score_adjustment'] -= 15
            result['issues'].append(f"Night lighting too high: {night_avg:.2f} vs day {day_avg:.2f}")
        
        # Evening should be moderate
        if evening_avg > day_avg * 0.8:
            result['score_adjustment'] += 5  # Good evening usage
        
        result['compliance'] = {
            'night_avg': float(night_avg),
            'day_avg': float(day_avg),
            'evening_avg': float(evening_avg),
            'night_day_ratio': float(night_avg / day_avg if day_avg > 0 else 0),
            'proper_day_night_pattern': night_avg <= day_avg * 0.3
        }
        
        return result
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        if validation_results['overall_score'] < 60:
            recommendations.append("Overall physics compliance is poor. Consider complete pattern redesign.")
        elif validation_results['overall_score'] < 80:
            recommendations.append("Physics compliance needs improvement. Focus on major issues first.")
        
        # Analyze common issues
        all_issues = validation_results['critical_issues'] + validation_results['warnings']
        
        if any('power changes' in issue for issue in all_issues):
            recommendations.append("Reduce sudden power changes by adding thermal inertia or smoothing.")
        
        if any('3-4 AM peak' in issue for issue in all_issues):
            recommendations.append("Fix unrealistic early morning peaks - most devices should be off 3-4 AM.")
        
        if any("doesn't loop" in issue for issue in all_issues):
            recommendations.append("Ensure daily patterns start and end at similar values for proper 24h cycling.")
        
        return recommendations


class OccupancyPatternLearner:
    """Advanced occupancy pattern learning for building usage optimization."""
    
    def __init__(self):
        self.occupancy_data = {}
        self.logger = logging.getLogger(__name__)
        self.learned_patterns = {}
    
    def learn_occupancy(self, load_data: pd.DataFrame, building_type: str = 'office') -> Dict:
        """Learn occupancy patterns from load data analysis."""
        
        if 'datetime' not in load_data.columns and load_data.index.name != 'datetime':
            # Try to create datetime from index or first column
            if hasattr(load_data.index, 'hour'):
                datetimes = load_data.index
            else:
                # Assume first column might be datetime
                datetimes = pd.to_datetime(load_data.iloc[:, 0])
        else:
            datetimes = load_data['datetime'] if 'datetime' in load_data.columns else load_data.index
        
        # Extract load data (assume total_power or Value column)
        if 'total_power' in load_data.columns:
            load_values = load_data['total_power']
        elif 'Value' in load_data.columns:
            load_values = load_data['Value']
        else:
            # Use first numeric column
            numeric_cols = load_data.select_dtypes(include=[np.number]).columns
            load_values = load_data[numeric_cols[0]]
        
        # Learn patterns
        occupancy_patterns = self._extract_occupancy_patterns(datetimes, load_values, building_type)
        usage_patterns = self._extract_usage_patterns(datetimes, load_values)
        seasonal_patterns = self._extract_seasonal_patterns(datetimes, load_values)
        
        self.learned_patterns = {
            'occupancy': occupancy_patterns,
            'usage': usage_patterns,
            'seasonal': seasonal_patterns,
            'building_type': building_type,
            'confidence': self._calculate_pattern_confidence(datetimes, load_values)
        }
        
        self.logger.info(f"Occupancy learning completed for {building_type} building")
        return {'occupancy_learned': True, 'patterns': self.learned_patterns}
    
    def _extract_occupancy_patterns(self, datetimes: pd.Series, load_values: pd.Series, building_type: str) -> Dict:
        """Extract occupancy probability patterns by hour and day type."""
        
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'datetime': pd.to_datetime(datetimes),
            'load': load_values
        })
        
        df['hour'] = df['datetime'].dt.hour
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = df['weekday'] >= 5
        
        # Normalize load to 0-1 for occupancy probability estimation
        df['load_normalized'] = (df['load'] - df['load'].min()) / (df['load'].max() - df['load'].min())
        
        # Calculate occupancy probability by hour for weekdays and weekends
        weekday_occupancy = df[~df['is_weekend']].groupby('hour')['load_normalized'].mean()
        weekend_occupancy = df[df['is_weekend']].groupby('hour')['load_normalized'].mean()
        
        # Apply building-type specific adjustments
        if building_type == 'office':
            # Office buildings: high occupancy 8-18, low otherwise
            for hour in range(24):
                if 8 <= hour <= 18:
                    weekday_occupancy[hour] = max(weekday_occupancy[hour], 0.7)
                elif hour <= 6 or hour >= 20:
                    weekday_occupancy[hour] = min(weekday_occupancy[hour], 0.2)
                    weekend_occupancy[hour] = min(weekend_occupancy[hour], 0.1)
        
        elif building_type == 'residential':
            # Residential: peaks morning/evening, lower during day
            for hour in range(24):
                if hour in [7, 8, 18, 19, 20]:
                    weekday_occupancy[hour] = max(weekday_occupancy[hour], 0.8)
                elif 9 <= hour <= 16:
                    weekday_occupancy[hour] = min(weekday_occupancy[hour], 0.4)
        
        return {
            'weekday_hourly': weekday_occupancy.to_dict(),
            'weekend_hourly': weekend_occupancy.to_dict(),
            'building_type': building_type
        }
    
    def _extract_usage_patterns(self, datetimes: pd.Series, load_values: pd.Series) -> Dict:
        """Extract equipment usage patterns."""
        
        df = pd.DataFrame({
            'datetime': pd.to_datetime(datetimes),
            'load': load_values
        })
        
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        
        # Identify usage peaks and patterns
        hourly_avg = df.groupby('hour')['load'].mean()
        monthly_avg = df.groupby('month')['load'].mean()
        
        # Find peak hours
        peak_hours = hourly_avg.nlargest(3).index.tolist()
        low_hours = hourly_avg.nsmallest(3).index.tolist()
        
        return {
            'hourly_average': hourly_avg.to_dict(),
            'monthly_average': monthly_avg.to_dict(),
            'peak_hours': peak_hours,
            'low_hours': low_hours,
            'daily_peak_ratio': float(hourly_avg.max() / hourly_avg.mean()),
            'seasonal_variation': float(monthly_avg.std() / monthly_avg.mean())
        }
    
    def _extract_seasonal_patterns(self, datetimes: pd.Series, load_values: pd.Series) -> Dict:
        """Extract seasonal usage patterns."""
        
        df = pd.DataFrame({
            'datetime': pd.to_datetime(datetimes),
            'load': load_values
        })
        
        df['month'] = df['datetime'].dt.month
        df['season'] = df['month'].apply(self._month_to_season)
        
        seasonal_avg = df.groupby('season')['load'].mean()
        monthly_avg = df.groupby('month')['load'].mean()
        
        return {
            'seasonal_average': seasonal_avg.to_dict(),
            'monthly_average': monthly_avg.to_dict(),
            'heating_season_factor': float(seasonal_avg.get('winter', 0) / seasonal_avg.mean()),
            'cooling_season_factor': float(seasonal_avg.get('summer', 0) / seasonal_avg.mean())
        }
    
    def _month_to_season(self, month: int) -> str:
        """Convert month number to season."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def _calculate_pattern_confidence(self, datetimes: pd.Series, load_values: pd.Series) -> Dict:
        """Calculate confidence in learned patterns."""
        
        # Calculate confidence based on data completeness and consistency
        data_completeness = len(load_values.dropna()) / len(load_values)
        
        # Calculate consistency (inverse of coefficient of variation by hour)
        df = pd.DataFrame({
            'datetime': pd.to_datetime(datetimes),
            'load': load_values
        })
        df['hour'] = df['datetime'].dt.hour
        
        hourly_cv = df.groupby('hour')['load'].apply(lambda x: x.std() / x.mean() if x.mean() > 0 else 1)
        consistency = 1 - hourly_cv.mean()
        
        # Time coverage (how many days/weeks of data)
        time_span = (df['datetime'].max() - df['datetime'].min()).days
        time_coverage = min(1.0, time_span / 365)  # Ideal: 1 year of data
        
        overall_confidence = (data_completeness * 0.4 + consistency * 0.4 + time_coverage * 0.2) * 100
        
        return {
            'overall_confidence': float(max(0, min(100, overall_confidence))),
            'data_completeness': float(data_completeness * 100),
            'pattern_consistency': float(consistency * 100),
            'time_coverage': float(time_coverage * 100)
        }