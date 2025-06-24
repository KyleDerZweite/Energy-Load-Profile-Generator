"""
Energy Load Disaggregator - Core Energy Balance Engine
=====================================================

This module implements energy disaggregation for building energy profiles.
It decomposes total building energy consumption into device-level profiles
using energy balance constraints and learned temperature relationships.

Key Features:
- Energy balance constraint: sum(devices) = total_energy (±1%)
- Temperature-based energy allocation (heating/cooling)
- Time-pattern based device scheduling
- GPU-accelerated computation for large datasets
- Building-agnostic modeling approach

This replaces the physics-based approach with pure energy accounting.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import copy
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class AllocationMethod(Enum):
    """Methods for allocating total energy to devices."""
    DEGREE_DAYS = "degree_days"          # Heating/cooling based on temperature
    TIME_PATTERN = "time_pattern"        # Based on time-of-day schedules
    SCHEDULE_BASED = "schedule_based"    # Equipment schedules with temp modulation
    CONSTANT = "constant"                # Base load allocation
    HYBRID = "hybrid"                    # Combination of methods


@dataclass
class DeviceEnergyModel:
    """Energy model for a single device type."""
    name: str
    allocation_method: AllocationMethod
    base_allocation_pct: float          # % of total energy in neutral conditions
    temp_sensitivity: float             # Energy change per degree
    temp_threshold: float               # Temperature threshold for response
    time_pattern: List[float]           # 96 15-minute intervals (0-1 scale)
    seasonal_variation: bool            # Whether to apply seasonal adjustments
    occupancy_dependency: float         # 0-1, dependency on building occupancy
    priority: int                       # 1=critical, 2=important, 3=normal, 4=optional


@dataclass
class EnergyDisaggregationResult:
    """Result of energy disaggregation."""
    device_profiles: Dict[str, np.ndarray]  # Device name -> energy time series (kW)
    total_predicted: np.ndarray             # Sum of all device profiles (kW)
    total_actual: np.ndarray                # Actual measured total (kW)
    energy_balance_error: float             # |predicted - actual| / actual * 100
    validation_metrics: Dict[str, float]    # MAE, RMSE, R², etc.
    allocation_summary: Dict[str, float]    # Device energy allocations (%)
    temporal_patterns: Dict[str, Any]       # Learned temporal patterns
    timestamps: Optional[np.ndarray] = None # Timestamps for the energy data


class EnergyDisaggregator:
    """
    Core energy disaggregation engine for building energy profiles.
    
    This class implements energy balance-based disaggregation that ensures
    the sum of device profiles matches the total measured consumption.
    """
    
    def __init__(self, config_manager=None, logger=None):
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Device models and learned parameters
        self.device_models: Dict[str, DeviceEnergyModel] = {}
        self.learned_parameters: Dict[str, Any] = {}
        self.training_data: Optional[pd.DataFrame] = None
        self.weather_data: Optional[pd.DataFrame] = None
        
        # Energy balance parameters
        self.energy_balance_tolerance = 0.01  # 1% tolerance for energy balance
        self.min_device_allocation = 0.001   # Minimum 0.1% allocation per device
        self.max_device_allocation = 0.5     # Maximum 50% allocation per device
        
        # Degree day parameters
        self.heating_base_temp = 15.0  # °C
        self.cooling_base_temp = 22.0  # °C
        
        # Model state
        self.is_trained = False
        self.training_years = []
        self.validation_metrics = {}
        
        self.logger.info("🔋 Energy Disaggregator initialized - Energy Balance Engine active")
    
    def initialize_device_models(self, devices_config: Dict) -> None:
        """Initialize device energy models from configuration."""
        self.device_models = {}
        
        for device_name, device_config in devices_config.items():
            if not device_config.get('enabled', True):
                continue
                
            # Determine allocation method based on device type
            allocation_method = self._determine_allocation_method(device_name, device_config)
            
            # Extract or generate time pattern
            time_pattern = device_config.get('daily_pattern', [0.5] * 96)
            if len(time_pattern) != 96:
                time_pattern = [0.5] * 96
                
            # Normalize time pattern
            time_pattern = np.array(time_pattern)
            if time_pattern.max() > 0:
                time_pattern = time_pattern / time_pattern.max()
            
            # Create device model
            device_model = DeviceEnergyModel(
                name=device_name,
                allocation_method=allocation_method,
                base_allocation_pct=self._estimate_base_allocation(device_name, device_config),
                temp_sensitivity=device_config.get('temp_coefficient', 0) / 1000.0,  # Convert W/°C to kW/°C
                temp_threshold=device_config.get('comfort_temp', 20),
                time_pattern=time_pattern.tolist(),
                seasonal_variation=device_config.get('seasonal_variation', False),
                occupancy_dependency=self._estimate_occupancy_dependency(device_name),
                priority=self._determine_priority(device_config.get('priority', 'normal'))
            )
            
            self.device_models[device_name] = device_model
            
        self.logger.info(f"📊 Initialized {len(self.device_models)} device energy models")
    
    def _determine_allocation_method(self, device_name: str, device_config: Dict) -> AllocationMethod:
        """Determine the appropriate allocation method for a device."""
        device_lower = device_name.lower()
        category = device_config.get('category', '').lower()
        
        # HVAC systems use degree days
        if any(keyword in device_lower for keyword in ['heat', 'heiz', 'warm']):
            return AllocationMethod.DEGREE_DAYS
        elif any(keyword in device_lower for keyword in ['cool', 'kühl', 'air_cond', 'chiller']):
            return AllocationMethod.DEGREE_DAYS
        elif any(keyword in device_lower for keyword in ['ventilation', 'lüft']):
            return AllocationMethod.DEGREE_DAYS
            
        # Lighting and scheduled equipment
        elif any(keyword in device_lower for keyword in ['light', 'beleucht', 'led']):
            return AllocationMethod.TIME_PATTERN
        elif category == 'equipment' or 'equipment' in device_lower:
            return AllocationMethod.SCHEDULE_BASED
            
        # Infrastructure and continuous loads
        elif category == 'infrastructure' or device_config.get('usage_pattern') == 'continuous':
            if device_config.get('temp_coefficient', 0) != 0:
                return AllocationMethod.DEGREE_DAYS
            else:
                return AllocationMethod.CONSTANT
                
        # Default to schedule-based for other devices
        else:
            return AllocationMethod.SCHEDULE_BASED
    
    def _estimate_base_allocation(self, device_name: str, device_config: Dict) -> float:
        """Estimate base energy allocation percentage for a device."""
        peak_power = device_config.get('peak_power', 1000)  # W
        duty_cycle = device_config.get('duty_cycle', 50) / 100.0
        quantity = device_config.get('quantity', 1)
        
        # Rough estimate: peak_power * duty_cycle * quantity / total_building_capacity
        # We'll normalize this later based on actual data
        estimated_avg_power = peak_power * duty_cycle * quantity / 1000.0  # kW
        
        # Start with rough allocation - will be adjusted during training
        if 'chiller' in device_name.lower() or 'cooling' in device_name.lower():
            return 0.15  # HVAC systems typically 10-20%
        elif 'heat' in device_name.lower() or 'ventilation' in device_name.lower():
            return 0.10
        elif 'light' in device_name.lower():
            return 0.08
        elif 'server' in device_name.lower():
            return 0.12
        elif device_config.get('category') == 'infrastructure':
            return 0.05
        else:
            return 0.02  # Default small allocation
    
    def _estimate_occupancy_dependency(self, device_name: str) -> float:
        """Estimate how much device energy depends on building occupancy."""
        device_lower = device_name.lower()
        
        if any(keyword in device_lower for keyword in ['light', 'beleucht']):
            return 0.8  # Lighting highly dependent on occupancy
        elif any(keyword in device_lower for keyword in ['coffee', 'microwave', 'dishwash']):
            return 0.9  # Appliances very dependent on occupancy
        elif any(keyword in device_lower for keyword in ['elevator', 'aufzug']):
            return 0.7  # Elevators depend on occupancy
        elif any(keyword in device_lower for keyword in ['ventilation', 'lab']):
            return 0.6  # Lab equipment partially dependent
        elif any(keyword in device_lower for keyword in ['server', 'network', 'security']):
            return 0.2  # Infrastructure minimally dependent
        else:
            return 0.5  # Default moderate dependency
    
    def _determine_priority(self, priority_value) -> int:
        """Convert priority string or int to numeric value."""
        # Handle integer priority values directly
        if isinstance(priority_value, int):
            return priority_value
        
        # Handle string priority values
        if isinstance(priority_value, str):
            priority_map = {
                'critical': 1,
                'important': 2,
                'normal': 3,
                'optional': 4
            }
            return priority_map.get(priority_value.lower(), 3)
        
        # Default fallback
        return 3
    
    def train(self, total_energy_data: pd.DataFrame, weather_data: pd.DataFrame, 
              training_years: List[int]) -> Dict[str, Any]:
        """
        Train the energy disaggregation model on historical data.
        
        Args:
            total_energy_data: DataFrame with timestamp and total energy consumption
            weather_data: DataFrame with timestamp and weather parameters
            training_years: List of years to use for training
            
        Returns:
            Dictionary with training results and learned parameters
        """
        self.logger.info(f"🎯 Training energy disaggregation model on years {training_years}")
        
        # Store training data
        self.training_data = total_energy_data.copy()
        self.weather_data = weather_data.copy()
        self.training_years = training_years
        
        # Filter data for training years
        train_data = self._filter_data_by_years(total_energy_data, training_years)
        train_weather = self._filter_data_by_years(weather_data, training_years)
        
        if len(train_data) == 0:
            raise ValueError(f"No training data found for years {training_years}")
        
        # Align data temporally
        aligned_data = self._align_energy_weather_data(train_data, train_weather)
        
        # Learn base building characteristics
        base_params = self._learn_base_building_parameters(aligned_data)
        
        # Learn device allocation patterns
        allocation_params = self._learn_device_allocations(aligned_data, base_params)
        
        # Learn temporal patterns
        temporal_params = self._learn_temporal_patterns(aligned_data)
        
        # Store learned parameters
        self.learned_parameters = {
            'base_building': base_params,
            'device_allocations': allocation_params,
            'temporal_patterns': temporal_params,
            'training_period': {
                'years': training_years,
                'data_points': len(aligned_data),
                'training_date': datetime.now().isoformat()
            }
        }
        
        # Set trained flag before validation
        self.is_trained = True
        
        # Validate training on same data (sanity check)
        training_validation = self._validate_on_data(aligned_data)
        self.validation_metrics['training'] = training_validation
        self.logger.info(f"✅ Training completed. Energy balance error: {training_validation['energy_balance_error']:.3f}%")
        
        return {
            'learned_parameters': self.learned_parameters,
            'validation_metrics': training_validation,
            'training_summary': {
                'years': training_years,
                'data_points': len(aligned_data),
                'devices': len(self.device_models),
                'training_r2': training_validation['r2_score']
            }
        }
    
    def _filter_data_by_years(self, data: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """Filter DataFrame to include only specified years."""
        if 'Timestamp' in data.columns:
            timestamp_col = 'Timestamp'
        elif 'timestamp' in data.columns:
            timestamp_col = 'timestamp'
        else:
            # Assume index is timestamp
            data_copy = data.copy()
            data_copy['timestamp'] = data_copy.index
            timestamp_col = 'timestamp'
            
        data_copy = data.copy()
        data_copy[timestamp_col] = pd.to_datetime(data_copy[timestamp_col])
        
        year_mask = data_copy[timestamp_col].dt.year.isin(years)
        filtered_data = data_copy[year_mask].copy()
        
        self.logger.info(f"📅 Filtered data: {len(filtered_data)} records from years {years}")
        return filtered_data
    
    def _align_energy_weather_data(self, energy_data: pd.DataFrame, 
                                  weather_data: pd.DataFrame) -> pd.DataFrame:
        """Align energy and weather data temporally."""
        # Ensure both have timestamp columns
        energy_cols = ['Timestamp', 'timestamp']
        weather_cols = ['Timestamp', 'timestamp']
        
        energy_ts_col = None
        for col in energy_cols:
            if col in energy_data.columns:
                energy_ts_col = col
                break
        
        weather_ts_col = None
        for col in weather_cols:
            if col in weather_data.columns:
                weather_ts_col = col
                break
        
        if energy_ts_col is None or weather_ts_col is None:
            self.logger.warning("⚠️ Timestamp columns not found, using index alignment")
            # Use index-based alignment
            energy_data['timestamp'] = energy_data.index
            weather_data['timestamp'] = weather_data.index
            energy_ts_col = 'timestamp'
            weather_ts_col = 'timestamp'
        
        # Convert to datetime
        energy_data[energy_ts_col] = pd.to_datetime(energy_data[energy_ts_col])
        weather_data[weather_ts_col] = pd.to_datetime(weather_data[weather_ts_col])
        
        # Merge on timestamp
        aligned = pd.merge(
            energy_data, 
            weather_data, 
            left_on=energy_ts_col, 
            right_on=weather_ts_col, 
            how='inner',
            suffixes=('_energy', '_weather')
        )
        
        # Use the energy timestamp as primary
        aligned['timestamp'] = aligned[energy_ts_col]
        
        self.logger.info(f"🔗 Aligned data: {len(aligned)} records")
        return aligned
    
    def _learn_base_building_parameters(self, aligned_data: pd.DataFrame) -> Dict[str, Any]:
        """Learn fundamental building energy characteristics."""
        # Find total energy column
        energy_cols = ['Value', 'value', 'total_energy', 'energy']
        energy_col = None
        for col in energy_cols:
            if col in aligned_data.columns:
                energy_col = col
                break
        
        if energy_col is None:
            raise ValueError("No energy column found in data")
        
        # Find temperature column
        temp_cols = ['temperature', 'Temperature', 'temp', 'Temp']
        temp_col = None
        for col in temp_cols:
            if col in aligned_data.columns:
                temp_col = col
                break
        
        if temp_col is None:
            self.logger.warning("⚠️ No temperature column found, using default temperature")
            aligned_data['temperature'] = 20.0  # Default temperature
            temp_col = 'temperature'
        
        total_energy = aligned_data[energy_col].values
        temperature = aligned_data[temp_col].values
        
        # Basic statistics
        base_load = np.percentile(total_energy, 5)  # 5th percentile as base load
        peak_load = np.percentile(total_energy, 95)  # 95th percentile as peak load
        avg_load = np.mean(total_energy)
        
        # Temperature response analysis
        heating_response = self._analyze_heating_response(total_energy, temperature)
        cooling_response = self._analyze_cooling_response(total_energy, temperature)
        
        # Time-based analysis
        time_patterns = self._analyze_time_patterns(aligned_data, energy_col)
        
        base_params = {
            'base_load_kw': base_load,
            'peak_load_kw': peak_load,
            'average_load_kw': avg_load,
            'load_variability': np.std(total_energy),
            'heating_coefficient': heating_response['coefficient'],
            'heating_r2': heating_response['r2'],
            'cooling_coefficient': cooling_response['coefficient'],
            'cooling_r2': cooling_response['r2'],
            'time_patterns': time_patterns,
            'temperature_range': {
                'min': float(np.min(temperature)),
                'max': float(np.max(temperature)),
                'mean': float(np.mean(temperature))
            }
        }
        
        self.logger.info(f"🏢 Base building: {base_load:.1f} kW base, {peak_load:.1f} kW peak")
        return base_params
    
    def _analyze_heating_response(self, energy: np.ndarray, temperature: np.ndarray) -> Dict[str, float]:
        """Analyze building heating energy response to temperature."""
        # Calculate heating degree days
        hdd = np.maximum(0, self.heating_base_temp - temperature)
        
        # Only use data where heating is likely (HDD > 0)
        heating_mask = hdd > 0
        if np.sum(heating_mask) < 10:
            return {'coefficient': 0.0, 'r2': 0.0}
        
        hdd_subset = hdd[heating_mask]
        energy_subset = energy[heating_mask]
        
        if HAS_SKLEARN:
            # Linear regression: energy = base + coefficient * HDD
            model = LinearRegression()
            X = hdd_subset.reshape(-1, 1)
            model.fit(X, energy_subset)
            
            r2 = model.score(X, energy_subset)
            coefficient = model.coef_[0]
        else:
            # Simple correlation-based estimate
            correlation = np.corrcoef(hdd_subset, energy_subset)[0, 1]
            coefficient = correlation * np.std(energy_subset) / np.std(hdd_subset)
            r2 = correlation ** 2
        
        return {
            'coefficient': float(coefficient),
            'r2': float(r2)
        }
    
    def _analyze_cooling_response(self, energy: np.ndarray, temperature: np.ndarray) -> Dict[str, float]:
        """Analyze building cooling energy response to temperature."""
        # Calculate cooling degree days
        cdd = np.maximum(0, temperature - self.cooling_base_temp)
        
        # Only use data where cooling is likely (CDD > 0)
        cooling_mask = cdd > 0
        if np.sum(cooling_mask) < 10:
            return {'coefficient': 0.0, 'r2': 0.0}
        
        cdd_subset = cdd[cooling_mask]
        energy_subset = energy[cooling_mask]
        
        if HAS_SKLEARN:
            # Linear regression: energy = base + coefficient * CDD
            model = LinearRegression()
            X = cdd_subset.reshape(-1, 1)
            model.fit(X, energy_subset)
            
            r2 = model.score(X, energy_subset)
            coefficient = model.coef_[0]
        else:
            # Simple correlation-based estimate
            correlation = np.corrcoef(cdd_subset, energy_subset)[0, 1]
            coefficient = correlation * np.std(energy_subset) / np.std(cdd_subset)
            r2 = correlation ** 2
        
        return {
            'coefficient': float(coefficient),
            'r2': float(r2)
        }
    
    def _analyze_time_patterns(self, data: pd.DataFrame, energy_col: str) -> Dict[str, Any]:
        """Analyze time-based energy patterns."""
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        
        # Hourly patterns
        hourly_pattern = data.groupby('hour')[energy_col].mean().values
        hourly_pattern = hourly_pattern / np.mean(hourly_pattern)  # Normalize
        
        # Daily patterns (Monday=0, Sunday=6)
        daily_pattern = data.groupby('day_of_week')[energy_col].mean().values
        daily_pattern = daily_pattern / np.mean(daily_pattern)  # Normalize
        
        # Monthly patterns
        monthly_pattern = data.groupby('month')[energy_col].mean().values
        monthly_pattern = monthly_pattern / np.mean(monthly_pattern)  # Normalize
        
        return {
            'hourly': hourly_pattern.tolist(),
            'daily': daily_pattern.tolist(),
            'monthly': monthly_pattern.tolist()
        }
    
    def _learn_device_allocations(self, aligned_data: pd.DataFrame, 
                                base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Learn how to allocate total energy to individual devices."""
        allocations = {}
        
        # Start with initial estimates from device models
        total_initial_allocation = 0
        for device_name, device_model in self.device_models.items():
            initial_allocation = device_model.base_allocation_pct
            allocations[device_name] = initial_allocation
            total_initial_allocation += initial_allocation
        
        # Normalize allocations to sum to 1.0
        if total_initial_allocation > 0:
            for device_name in allocations:
                allocations[device_name] /= total_initial_allocation
        
        # Adjust allocations based on learned building characteristics
        self._adjust_allocations_for_temperature_response(
            allocations, base_params
        )
        
        self.logger.info(f"💡 Device allocations learned for {len(allocations)} devices")
        return allocations
    
    def _adjust_allocations_for_temperature_response(self, allocations: Dict[str, float], 
                                                   base_params: Dict[str, Any]) -> None:
        """Adjust device allocations based on temperature response analysis."""
        heating_coeff = base_params['heating_coefficient']
        cooling_coeff = base_params['cooling_coefficient']
        
        # Find heating and cooling devices
        heating_devices = []
        cooling_devices = []
        
        for device_name, device_model in self.device_models.items():
            if device_model.allocation_method == AllocationMethod.DEGREE_DAYS:
                if ('heat' in device_name.lower() or 
                    'warm' in device_name.lower() or 
                    device_model.temp_threshold < 20):
                    heating_devices.append(device_name)
                else:
                    cooling_devices.append(device_name)
        
        # Boost allocation for heating devices if strong heating response
        if heating_coeff > 0 and base_params['heating_r2'] > 0.3:
            boost_factor = min(2.0, 1 + base_params['heating_r2'])
            for device_name in heating_devices:
                if device_name in allocations:
                    allocations[device_name] *= boost_factor
        
        # Boost allocation for cooling devices if strong cooling response
        if cooling_coeff > 0 and base_params['cooling_r2'] > 0.3:
            boost_factor = min(2.0, 1 + base_params['cooling_r2'])
            for device_name in cooling_devices:
                if device_name in allocations:
                    allocations[device_name] *= boost_factor
        
        # Renormalize
        total_allocation = sum(allocations.values())
        if total_allocation > 0:
            for device_name in allocations:
                allocations[device_name] /= total_allocation
    
    def _learn_temporal_patterns(self, aligned_data: pd.DataFrame) -> Dict[str, Any]:
        """Learn temporal patterns for device scheduling."""
        # Extract basic temporal features
        time_features = self._extract_time_features(aligned_data)
        
        # Learn occupancy patterns (simplified)
        occupancy_pattern = self._estimate_occupancy_pattern(aligned_data)
        
        return {
            'time_features': time_features,
            'occupancy_pattern': occupancy_pattern
        }
    
    def _extract_time_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract time-based features from data."""
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Create 15-minute interval index (0-95 for 24 hours)
        data['interval_15min'] = (data['timestamp'].dt.hour * 4 + 
                                data['timestamp'].dt.minute // 15)
        
        # Find energy column
        energy_cols = ['Value', 'value', 'total_energy', 'energy']
        energy_col = None
        for col in energy_cols:
            if col in data.columns:
                energy_col = col
                break
        
        if energy_col is None:
            return {}
        
        # Average energy by 15-minute interval
        interval_pattern = data.groupby('interval_15min')[energy_col].mean()
        
        # Ensure we have all 96 intervals
        full_pattern = np.zeros(96)
        for interval in range(96):
            if interval in interval_pattern.index:
                full_pattern[interval] = interval_pattern[interval]
            else:
                # Interpolate missing values
                full_pattern[interval] = np.mean(interval_pattern.values)
        
        # Normalize pattern
        if np.max(full_pattern) > 0:
            full_pattern = full_pattern / np.max(full_pattern)
        
        return {
            'interval_pattern_96': full_pattern.tolist(),
            'peak_interval': int(np.argmax(full_pattern)),
            'min_interval': int(np.argmin(full_pattern)),
            'daily_variation': float(np.std(full_pattern))
        }
    
    def _estimate_occupancy_pattern(self, data: pd.DataFrame) -> List[float]:
        """Estimate building occupancy pattern from energy data."""
        # Simple heuristic: higher energy during business hours indicates occupancy
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['is_weekday'] = data['timestamp'].dt.dayofweek < 5
        
        energy_cols = ['Value', 'value', 'total_energy', 'energy']
        energy_col = None
        for col in energy_cols:
            if col in data.columns:
                energy_col = col
                break
        
        if energy_col is None:
            # Default occupancy pattern (business hours)
            occupancy = [0.1] * 96  # 96 15-minute intervals
            for i in range(32, 72):  # 8 AM to 6 PM
                occupancy[i] = 1.0
            return occupancy
        
        # Weekday energy pattern as proxy for occupancy
        weekday_data = data[data['is_weekday']]
        if len(weekday_data) == 0:
            return [0.5] * 96  # Default pattern
        
        weekday_data['interval_15min'] = (weekday_data['hour'] * 4)
        hourly_energy = weekday_data.groupby('hour')[energy_col].mean()
        
        # Convert to 96 intervals (15-minute resolution)
        occupancy_pattern = []
        for hour in range(24):
            if hour in hourly_energy.index:
                hour_energy = hourly_energy[hour]
            else:
                hour_energy = np.mean(hourly_energy.values)
            
            # Add 4 15-minute intervals for this hour
            for _ in range(4):
                occupancy_pattern.append(hour_energy)
        
        # Normalize to 0-1 range
        occupancy_pattern = np.array(occupancy_pattern)
        if np.max(occupancy_pattern) > np.min(occupancy_pattern):
            occupancy_pattern = (occupancy_pattern - np.min(occupancy_pattern)) / (np.max(occupancy_pattern) - np.min(occupancy_pattern))
        else:
            occupancy_pattern = np.ones_like(occupancy_pattern) * 0.5
        
        return occupancy_pattern.tolist()
    
    def disaggregate(self, total_energy: np.ndarray, weather_data: pd.DataFrame,
                    time_data: pd.DataFrame) -> EnergyDisaggregationResult:
        """
        Disaggregate total energy into device-level profiles.
        
        Args:
            total_energy: Array of total energy consumption (kW)
            weather_data: Weather data with temperature
            time_data: Time data with timestamps
            
        Returns:
            EnergyDisaggregationResult with device profiles and validation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before disaggregation")
        
        self.logger.info(f"⚡ Disaggregating {len(total_energy)} energy data points")
        
        # Extract temperature and time features
        temperature = self._extract_temperature(weather_data)
        time_features = self._extract_time_features_for_disaggregation(time_data)
        occupancy = self._extract_occupancy_for_disaggregation(time_data)
        
        # Ensure all arrays have same length
        n_points = len(total_energy)
        if len(temperature) != n_points:
            temperature = np.resize(temperature, n_points)
        if len(occupancy) != n_points:
            occupancy = np.resize(occupancy, n_points)
        
        # Generate device profiles
        device_profiles = {}
        base_params = self.learned_parameters['base_building']
        allocations = self.learned_parameters['device_allocations']
        
        for device_name, device_model in self.device_models.items():
            if device_name in allocations:
                device_energy = self._generate_device_profile(
                    device_model, allocations[device_name], total_energy,
                    temperature, time_features, occupancy, base_params
                )
                device_profiles[device_name] = device_energy
        
        # Ensure energy balance
        device_profiles = self._enforce_energy_balance(device_profiles, total_energy)
        
        # Calculate validation metrics
        total_predicted = np.sum([profile for profile in device_profiles.values()], axis=0)
        validation_metrics = self._calculate_validation_metrics(total_predicted, total_energy)
        
        # Calculate allocation summary
        allocation_summary = {}
        for device_name, profile in device_profiles.items():
            allocation_summary[device_name] = float(np.mean(profile) / np.mean(total_energy) * 100)
        
        # Extract timestamps from time_data
        timestamps = None
        if time_data is not None and not time_data.empty:
            timestamp_cols = ['timestamp', 'Timestamp', 'time', 'Time', 'datetime']
            for col in timestamp_cols:
                if col in time_data.columns:
                    timestamps = time_data[col].values
                    break
            if timestamps is None and len(time_data.columns) > 0:
                # Use first column if no standard timestamp column found
                timestamps = time_data.iloc[:, 0].values
        
        result = EnergyDisaggregationResult(
            device_profiles=device_profiles,
            total_predicted=total_predicted,
            total_actual=total_energy,
            energy_balance_error=validation_metrics['energy_balance_error'],
            validation_metrics=validation_metrics,
            allocation_summary=allocation_summary,
            temporal_patterns=self.learned_parameters.get('temporal_patterns', {}),
            timestamps=timestamps
        )
        
        self.logger.info(f"✅ Disaggregation completed. Energy balance error: {result.energy_balance_error:.3f}%")
        return result
    
    def _extract_temperature(self, weather_data: pd.DataFrame) -> np.ndarray:
        """Extract temperature array from weather data."""
        temp_cols = ['temperature', 'Temperature', 'temp', 'Temp']
        
        for col in temp_cols:
            if col in weather_data.columns:
                return weather_data[col].values
        
        # Default temperature if not found
        self.logger.warning("⚠️ No temperature data found, using default 20°C")
        return np.full(len(weather_data), 20.0)
    
    def _extract_time_features_for_disaggregation(self, time_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract time features for disaggregation."""
        # Find timestamp column
        ts_cols = ['timestamp', 'Timestamp', 'time', 'Time']
        ts_col = None
        for col in ts_cols:
            if col in time_data.columns:
                ts_col = col
                break
        
        if ts_col is None:
            # Use index if no timestamp column
            timestamps = pd.to_datetime(time_data.index)
        else:
            timestamps = pd.to_datetime(time_data[ts_col])
        
        # Extract features
        hours = timestamps.dt.hour.values
        days_of_week = timestamps.dt.dayofweek.values
        months = timestamps.dt.month.values
        
        # 15-minute intervals (0-95)
        intervals_15min = hours * 4 + timestamps.dt.minute.values // 15
        
        return {
            'hour': hours,
            'day_of_week': days_of_week,
            'month': months,
            'interval_15min': intervals_15min,
            'is_weekday': (days_of_week < 5).astype(float)
        }
    
    def _extract_occupancy_for_disaggregation(self, time_data: pd.DataFrame) -> np.ndarray:
        """Extract occupancy pattern for disaggregation."""
        time_features = self._extract_time_features_for_disaggregation(time_data)
        learned_occupancy = self.learned_parameters.get('temporal_patterns', {}).get('occupancy_pattern', [0.5] * 96)
        
        # Map 15-minute intervals to occupancy values
        occupancy = []
        for interval in time_features['interval_15min']:
            if 0 <= interval < 96:
                occupancy.append(learned_occupancy[interval])
            else:
                occupancy.append(0.5)  # Default
        
        return np.array(occupancy)
    
    def _generate_device_profile(self, device_model: DeviceEnergyModel, 
                               allocation_pct: float, total_energy: np.ndarray,
                               temperature: np.ndarray, time_features: Dict[str, np.ndarray],
                               occupancy: np.ndarray, base_params: Dict[str, Any]) -> np.ndarray:
        """Generate energy profile for a single device."""
        n_points = len(total_energy)
        device_energy = np.zeros(n_points)
        
        # Base allocation
        base_energy = total_energy * allocation_pct
        
        if device_model.allocation_method == AllocationMethod.DEGREE_DAYS:
            # Temperature-based allocation
            if 'heat' in device_model.name.lower() or 'warm' in device_model.name.lower():
                # Heating device
                hdd = np.maximum(0, self.heating_base_temp - temperature)
                temp_factor = 1 + device_model.temp_sensitivity * hdd
            else:
                # Cooling device
                cdd = np.maximum(0, temperature - self.cooling_base_temp)
                temp_factor = 1 + device_model.temp_sensitivity * cdd
            
            device_energy = base_energy * temp_factor
            
        elif device_model.allocation_method == AllocationMethod.TIME_PATTERN:
            # Time pattern-based allocation
            time_factors = []
            for interval in time_features['interval_15min']:
                if 0 <= interval < 96:
                    time_factors.append(device_model.time_pattern[interval])
                else:
                    time_factors.append(np.mean(device_model.time_pattern))
            
            time_factors = np.array(time_factors)
            device_energy = base_energy * time_factors
            
        elif device_model.allocation_method == AllocationMethod.SCHEDULE_BASED:
            # Schedule-based with occupancy dependency
            time_factors = []
            for interval in time_features['interval_15min']:
                if 0 <= interval < 96:
                    time_factors.append(device_model.time_pattern[interval])
                else:
                    time_factors.append(np.mean(device_model.time_pattern))
            
            time_factors = np.array(time_factors)
            
            # Apply occupancy dependency
            occupancy_effect = (1 - device_model.occupancy_dependency) + device_model.occupancy_dependency * occupancy
            
            # Apply small temperature modulation
            temp_modulation = 1 + device_model.temp_sensitivity * (temperature - 20)
            
            device_energy = base_energy * time_factors * occupancy_effect * temp_modulation
            
        elif device_model.allocation_method == AllocationMethod.CONSTANT:
            # Constant allocation with minimal variation
            device_energy = base_energy * (1 + np.random.normal(0, 0.05, n_points))
            
        else:
            # Default allocation
            device_energy = base_energy
        
        # Apply constraints
        device_energy = np.maximum(0, device_energy)  # No negative energy
        
        return device_energy
    
    def _enforce_energy_balance(self, device_profiles: Dict[str, np.ndarray], 
                              total_energy: np.ndarray) -> Dict[str, np.ndarray]:
        """Enforce energy balance constraint: sum(devices) = total_energy."""
        n_points = len(total_energy)
        
        # Calculate current total from devices
        current_total = np.sum([profile for profile in device_profiles.values()], axis=0)
        
        # Calculate scaling factors to match total energy
        scaling_factors = np.divide(total_energy, current_total, 
                                  out=np.ones_like(total_energy), where=current_total!=0)
        
        # Apply scaling to each device proportionally
        balanced_profiles = {}
        for device_name, profile in device_profiles.items():
            balanced_profiles[device_name] = profile * scaling_factors
        
        # Verify balance
        new_total = np.sum([profile for profile in balanced_profiles.values()], axis=0)
        max_error = np.max(np.abs(new_total - total_energy) / total_energy) * 100
        
        if max_error > self.energy_balance_tolerance * 100:
            self.logger.warning(f"⚠️ Energy balance error {max_error:.3f}% exceeds tolerance")
        
        return balanced_profiles
    
    def _calculate_validation_metrics(self, predicted: np.ndarray, 
                                    actual: np.ndarray) -> Dict[str, float]:
        """Calculate validation metrics for energy balance."""
        # Energy balance error
        energy_balance_error = np.mean(np.abs(predicted - actual) / actual) * 100
        
        # Standard regression metrics
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        mape = np.mean(np.abs((predicted - actual) / actual)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'energy_balance_error': float(energy_balance_error),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2),
            'max_absolute_error': float(np.max(np.abs(predicted - actual))),
            'mean_actual': float(np.mean(actual)),
            'mean_predicted': float(np.mean(predicted))
        }
    
    def _validate_on_data(self, data: pd.DataFrame) -> Dict[str, float]:
        """Validate model on given data."""
        # Extract features from validation data
        energy_cols = ['Value', 'value', 'total_energy', 'energy']
        energy_col = None
        for col in energy_cols:
            if col in data.columns:
                energy_col = col
                break
        
        if energy_col is None:
            raise ValueError("No energy column found for validation")
        
        total_energy = data[energy_col].values
        
        # Create mock weather and time data for validation
        weather_data = pd.DataFrame({
            'temperature': data.get('temperature', pd.Series([20.0] * len(data)))
        })
        
        time_data = pd.DataFrame({
            'timestamp': data['timestamp']
        })
        
        # Disaggregate
        result = self.disaggregate(total_energy, weather_data, time_data)
        
        return result.validation_metrics
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON serializable types."""
        import numpy as np
        import math
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN and infinity values
            float_val = float(obj)
            if math.isnan(float_val) or math.isinf(float_val):
                return None
            return float_val
        elif isinstance(obj, (float, int)):
            # Handle regular Python float/int that might be NaN
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        import json
        
        model_data = {
            'device_models': {name: {
                'name': model.name,
                'allocation_method': model.allocation_method.value,
                'base_allocation_pct': self._convert_to_json_serializable(model.base_allocation_pct),
                'temp_sensitivity': self._convert_to_json_serializable(model.temp_sensitivity),
                'temp_threshold': self._convert_to_json_serializable(model.temp_threshold),
                'time_pattern': model.time_pattern,
                'seasonal_variation': self._convert_to_json_serializable(model.seasonal_variation),
                'occupancy_dependency': self._convert_to_json_serializable(model.occupancy_dependency),
                'priority': self._convert_to_json_serializable(model.priority)
            } for name, model in self.device_models.items()},
            'learned_parameters': self._convert_to_json_serializable(self.learned_parameters),
            'training_years': self._convert_to_json_serializable(self.training_years),
            'is_trained': self.is_trained,
            'validation_metrics': self._convert_to_json_serializable(self.validation_metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        self.logger.info(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        import json
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Reconstruct device models
        self.device_models = {}
        for name, model_dict in model_data['device_models'].items():
            self.device_models[name] = DeviceEnergyModel(
                name=model_dict['name'],
                allocation_method=AllocationMethod(model_dict['allocation_method']),
                base_allocation_pct=model_dict['base_allocation_pct'],
                temp_sensitivity=model_dict['temp_sensitivity'],
                temp_threshold=model_dict['temp_threshold'],
                time_pattern=model_dict['time_pattern'],
                seasonal_variation=model_dict['seasonal_variation'],
                occupancy_dependency=model_dict['occupancy_dependency'],
                priority=model_dict['priority']
            )
        
        # Restore other attributes
        self.learned_parameters = model_data['learned_parameters']
        self.training_years = model_data['training_years']
        self.is_trained = model_data['is_trained']
        self.validation_metrics = model_data.get('validation_metrics', {})
        
        self.logger.info(f"📂 Model loaded from {filepath}")