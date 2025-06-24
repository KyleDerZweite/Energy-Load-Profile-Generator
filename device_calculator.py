"""
Energy Balance Device Calculator
===============================

This module provides device load calculation based on energy balance principles
rather than physics simulation. It integrates with the energy disaggregation system
to provide consistent device-level energy profiles.

Features:
- Energy balance-based device modeling
- Integration with energy disaggregator
- Weather-dependent device responses
- Time-pattern based load calculation
- Device allocation management
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Tuple, Any, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Import our energy balance components
from energy_disaggregator import EnergyDisaggregator, DeviceEnergyModel
from weather_energy_analyzer import WeatherEnergyAnalyzer


class EnergyBalanceDeviceCalculator:
    """
    Device load calculator based on energy balance principles.
    
    This replaces physics-based calculations with energy accounting,
    ensuring device profiles sum to total building energy consumption.
    """
    
    def __init__(self, device_configs: Dict, config_manager=None, accelerator=None):
        self.device_configs = device_configs
        self.config_manager = config_manager
        self.accelerator = accelerator
        self.logger = logging.getLogger(__name__)
        
        # Initialize energy disaggregator for device calculations
        self.disaggregator = EnergyDisaggregator(config_manager, accelerator, self.logger)
        self.disaggregator.initialize_device_models(device_configs)
        
        # Device allocation cache
        self._device_allocations = {}
        self._last_total_energy = None
        
        self.logger.info(f"🔋 Energy Balance Device Calculator initialized with {len(device_configs)} devices")
    
    def calculate_device_loads(self, total_energy: Union[pd.DataFrame, np.ndarray],
                             weather_data: pd.DataFrame,
                             time_data: Optional[pd.DataFrame] = None,
                             devices: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Calculate device loads using energy balance approach.
        
        Args:
            total_energy: Total building energy consumption
            weather_data: Weather data for the same period
            time_data: Time data (optional)
            devices: List of specific devices to calculate (optional)
            
        Returns:
            Dictionary mapping device names to energy load arrays
        """
        self.logger.info("⚡ Calculating device loads using energy balance")
        
        # Convert total_energy to array if needed
        if isinstance(total_energy, pd.DataFrame):
            # Find energy column
            energy_cols = ['Value', 'value', 'energy', 'Energy', 'total_energy']
            energy_col = None
            for col in energy_cols:
                if col in total_energy.columns:
                    energy_col = col
                    break
            
            if energy_col is None:
                raise ValueError("No energy column found in total_energy DataFrame")
            
            energy_array = total_energy[energy_col].values
            
            # Use timestamp from energy data if time_data not provided
            if time_data is None:
                if 'Timestamp' in total_energy.columns:
                    time_data = total_energy[['Timestamp']]
                else:
                    time_data = total_energy
        else:
            energy_array = np.array(total_energy)
            
            # Create time_data if not provided
            if time_data is None:
                time_data = pd.DataFrame({
                    'timestamp': pd.date_range(start='2024-01-01', periods=len(energy_array), freq='15min')
                })
        
        # Check if we need to retrain or can use cached allocations
        if not self._can_use_cached_allocations(energy_array):
            self.logger.info("🔄 Retraining device models for new energy data")
            self._train_for_energy_data(energy_array, weather_data, time_data)
        
        # Perform energy disaggregation
        if not self.disaggregator.is_trained:
            # Quick training mode for device calculation
            self.disaggregator._learn_device_allocations(energy_array, weather_data, time_data)
            self.disaggregator.is_trained = True
        
        # Get device profiles
        result = self.disaggregator.disaggregate(energy_array, weather_data, time_data)
        
        # Filter to requested devices if specified
        if devices:
            device_profiles = {device: result.device_profiles[device] 
                             for device in devices if device in result.device_profiles}
        else:
            device_profiles = result.device_profiles
        
        self.logger.info(f"✅ Calculated loads for {len(device_profiles)} devices")
        return device_profiles
    
    def calculate_total_load(self, weather_data: pd.DataFrame, devices: List[str],
                           peak_power: float, time_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total building load from individual device contributions.
        
        Args:
            weather_data: Weather data
            devices: List of devices to include
            peak_power: Building peak power capacity
            time_data: Time data for the calculation period
            
        Returns:
            DataFrame with total load time series
        """
        self.logger.info(f"📊 Calculating total load from {len(devices)} devices")
        
        # Generate base total energy using weather relationships
        total_energy = self._generate_base_total_energy(weather_data, time_data, peak_power)
        
        # Calculate device loads
        device_loads = self.calculate_device_loads(total_energy, weather_data, time_data, devices)
        
        # Sum device loads to get total
        total_load = np.zeros(len(total_energy))
        for device_name in devices:
            if device_name in device_loads:
                total_load += device_loads[device_name]
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'timestamp': pd.to_datetime(time_data['timestamp'] if 'timestamp' in time_data.columns else time_data.index),
            'total_load': total_load
        })
        
        # Add individual device columns
        for device_name in devices:
            if device_name in device_loads:
                result_df[device_name] = device_loads[device_name]
        
        self.logger.info(f"✅ Total load calculated: {np.mean(total_load):.1f} kW average")
        return result_df
    
    def get_device_allocation_summary(self) -> Dict[str, float]:
        """Get current device allocation percentages."""
        if hasattr(self.disaggregator, 'learned_parameters'):
            allocations = self.disaggregator.learned_parameters.get('device_allocations', {})
            return {device: alloc * 100 for device, alloc in allocations.items()}
        return {}
    
    def update_device_config(self, device_name: str, new_config: Dict) -> None:
        """Update configuration for a specific device."""
        self.device_configs[device_name] = new_config
        self.disaggregator.device_models[device_name] = DeviceEnergyModel(device_name, new_config)
        self.logger.info(f"🔧 Updated configuration for device: {device_name}")
    
    def add_device(self, device_name: str, device_config: Dict) -> None:
        """Add a new device to the calculator."""
        self.device_configs[device_name] = device_config
        self.disaggregator.device_models[device_name] = DeviceEnergyModel(device_name, device_config)
        self.logger.info(f"➕ Added new device: {device_name}")
    
    def remove_device(self, device_name: str) -> None:
        """Remove a device from the calculator."""
        if device_name in self.device_configs:
            del self.device_configs[device_name]
        if device_name in self.disaggregator.device_models:
            del self.disaggregator.device_models[device_name]
        self.logger.info(f"➖ Removed device: {device_name}")
    
    def _can_use_cached_allocations(self, energy_array: np.ndarray) -> bool:
        """Check if we can reuse cached device allocations."""
        if self._last_total_energy is None:
            return False
        
        # Check if energy profile is similar
        if len(energy_array) != len(self._last_total_energy):
            return False
        
        # Check correlation
        correlation = np.corrcoef(energy_array, self._last_total_energy)[0, 1]
        return correlation > 0.95  # 95% correlation threshold
    
    def _train_for_energy_data(self, energy_array: np.ndarray, weather_data: pd.DataFrame,
                              time_data: pd.DataFrame) -> None:
        """Train disaggregator for specific energy data."""
        # Quick learning mode
        self.disaggregator._learn_device_allocations(energy_array, weather_data, time_data)
        self.disaggregator.is_trained = True
        self._last_total_energy = energy_array.copy()
    
    def _generate_base_total_energy(self, weather_data: pd.DataFrame, 
                                   time_data: pd.DataFrame, peak_power: float) -> np.ndarray:
        """Generate base total energy using simple weather relationships."""
        # Extract temperature
        temp_cols = ['temperature', 'Temperature', 'temp', 'Temp']
        temperature = None
        for col in temp_cols:
            if col in weather_data.columns:
                temperature = weather_data[col].values
                break
        
        if temperature is None:
            temperature = np.full(len(weather_data), 20.0)  # Default 20°C
        
        # Simple degree-day based model
        heating_base = 15.0  # °C
        cooling_base = 22.0  # °C
        
        hdd = np.maximum(0, heating_base - temperature)
        cdd = np.maximum(0, temperature - cooling_base)
        
        # Base load + heating/cooling components
        base_load = peak_power * 0.3  # 30% base load
        heating_load = hdd * 2.0  # 2 kW per degree day
        cooling_load = cdd * 1.5  # 1.5 kW per degree day
        
        total_energy = base_load + heating_load + cooling_load
        
        # Add time-of-day variation
        if 'timestamp' in time_data.columns:
            timestamps = pd.to_datetime(time_data['timestamp'])
            hours = timestamps.dt.hour.values
            
            # Business hours pattern
            time_factors = np.ones_like(hours, dtype=float)
            time_factors[(hours >= 8) & (hours <= 18)] *= 1.2  # 20% higher during business hours
            time_factors[(hours >= 0) & (hours <= 6)] *= 0.8   # 20% lower at night
            
            total_energy = total_energy * time_factors
        
        # Ensure positive and reasonable values
        total_energy = np.maximum(total_energy, base_load * 0.5)
        total_energy = np.minimum(total_energy, peak_power)
        
        return total_energy
    
    def get_device_summary(self) -> Dict[str, Any]:
        """Get summary of all configured devices."""
        summary = {
            'total_devices': len(self.device_configs),
            'device_types': {},
            'allocation_methods': {},
            'devices': {}
        }
        
        for device_name, config in self.device_configs.items():
            device_type = config.get('type', 'unknown')
            allocation_method = config.get('allocation_method', 'unknown')
            
            summary['device_types'][device_type] = summary['device_types'].get(device_type, 0) + 1
            summary['allocation_methods'][allocation_method] = summary['allocation_methods'].get(allocation_method, 0) + 1
            
            summary['devices'][device_name] = {
                'type': device_type,
                'allocation_method': allocation_method,
                'base_allocation_pct': config.get('base_allocation_pct', 0),
                'temp_sensitivity': config.get('temp_sensitivity', 0),
                'occupancy_dependency': config.get('occupancy_dependency', 0)
            }
        
        return summary


# Backward compatibility aliases
DeviceLoadCalculator = EnergyBalanceDeviceCalculator

# Legacy function for backward compatibility
def calculate_device_loads(device_configs: Dict, total_energy: np.ndarray,
                         weather_data: pd.DataFrame, time_data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Legacy function for backward compatibility."""
    calculator = EnergyBalanceDeviceCalculator(device_configs)
    return calculator.calculate_device_loads(total_energy, weather_data, time_data)