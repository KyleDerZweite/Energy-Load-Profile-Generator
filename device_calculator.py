import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union
import logging

class DeviceLoadCalculator:
    """Calculates energy load for various electrical devices based on weather and usage patterns."""

    def __init__(self, device_configs: Dict):
        self.device_configs = device_configs
        self.logger = logging.getLogger(__name__)
        # Pre-compute daily patterns for faster lookup
        self._precompute_daily_patterns()

    def _precompute_daily_patterns(self):
        """Pre-compute and cache daily patterns for all devices."""
        self._pattern_cache = {}

        for device_name, config in self.device_configs.items():
            daily_pattern = config.get('daily_pattern', [1.0] * 96)
            if len(daily_pattern) != 96:
                if len(daily_pattern) == 24:
                    # Convert 24-hour pattern to 96 x 15-minute pattern
                    extended_pattern = []
                    for hour_value in daily_pattern:
                        extended_pattern.extend([hour_value] * 4)  # 4 x 15-min intervals per hour
                    daily_pattern = extended_pattern
                else:
                    daily_pattern = [1.0] * 96

            # Cache as numpy array for faster indexing
            self._pattern_cache[device_name] = np.array(daily_pattern)

    def calculate_device_load(self, device_name: str, temperature: float,
                              hour_of_day: int, minute_of_day: int, day_of_year: int,
                              humidity: float = 50) -> float:
        """Calculate power consumption for a specific device."""

        if device_name not in self.device_configs:
            self.logger.warning(f"Device '{device_name}' not configured")
            return 0.0

        config = self.device_configs[device_name]

        if not config.get('enabled', True):
            return 0.0

        # Peak power consumption (maximum possible power)
        peak_power = config['peak_power']

        # Temperature influence
        temp_diff = temperature - config['comfort_temp']
        temp_coefficient = config.get('temp_coefficient', 0)
        temp_factor = 1 + (temp_diff * temp_coefficient / peak_power)
        temp_factor = max(0.0, min(2.0, temp_factor))  # Clamp between 0% and 200%

        # Daily pattern influence (96 x 15-minute intervals)
        # Use pre-computed pattern for faster lookup
        if device_name in self._pattern_cache:
            daily_pattern_array = self._pattern_cache[device_name]
        else:
            # Fallback to original logic if not cached
            daily_pattern = config.get('daily_pattern', [1.0] * 96)
            if len(daily_pattern) != 96:
                self.logger.warning(f"Invalid daily pattern for {device_name}, expected 96 values (15-min intervals), got {len(daily_pattern)}")
                if len(daily_pattern) == 24:
                    extended_pattern = []
                    for hour_value in daily_pattern:
                        extended_pattern.extend([hour_value] * 4)
                    daily_pattern = extended_pattern
                else:
                    daily_pattern = [1.0] * 96
            daily_pattern_array = np.array(daily_pattern)

        # Calculate 15-minute interval index (0-95)
        interval_index = (hour_of_day * 4) + (minute_of_day // 15)
        interval_index = min(95, max(0, interval_index))  # Ensure valid index

        # Ensure daily pattern values are between 0.0 and 1.0
        daily_factor = max(0.0, min(1.0, float(daily_pattern_array[interval_index])))

        # Humidity influence (for some devices)
        humidity_factor = 1.0
        if device_name in ['air_conditioner', 'refrigeration']:
            # Higher humidity increases load
            humidity_factor = 1 + (humidity - 50) * 0.002  # 0.2% per humidity percent
            humidity_factor = max(0.8, min(1.3, humidity_factor))

        # Add realistic random variation (±5%)
        random_factor = 1 + np.random.normal(0, 0.05)
        random_factor = max(0.5, min(1.5, random_factor))  # Clamp to reasonable range

        # Calculate total power consumption
        # Base calculation uses daily_factor as the primary multiplier (0.0 to 1.0)
        base_consumption = peak_power * daily_factor

        # Apply additional factors
        total_power = (base_consumption * temp_factor * humidity_factor * random_factor)

        # Ensure we never exceed peak power under normal conditions (allow some headroom for extreme conditions)
        max_allowed_power = peak_power * 1.2  # 20% headroom for extreme conditions
        total_power = min(total_power, max_allowed_power)

        return max(0, total_power)

    def _calculate_device_load_vectorized(self, device_name: str,
                                          temperature: np.ndarray,
                                          hour_of_day: np.ndarray,
                                          minute_of_day: np.ndarray,
                                          day_of_year: np.ndarray,
                                          humidity: np.ndarray) -> np.ndarray:
        """Vectorized version of calculate_device_load for internal use."""

        if device_name not in self.device_configs:
            return np.zeros_like(temperature, dtype=float)

        config = self.device_configs[device_name]

        if not config.get('enabled', True):
            return np.zeros_like(temperature, dtype=float)

        # Peak power consumption
        peak_power = config['peak_power']

        # Temperature influence (vectorized)
        temp_diff = temperature - config['comfort_temp']
        temp_coefficient = config.get('temp_coefficient', 0)
        temp_factor = 1 + (temp_diff * temp_coefficient / peak_power)
        temp_factor = np.clip(temp_factor, 0.0, 2.0)

        # Daily pattern influence (vectorized)
        # Use pre-computed pattern
        daily_pattern_array = self._pattern_cache.get(device_name, np.ones(96))

        # Calculate interval indices (vectorized)
        interval_indices = (hour_of_day * 4) + (minute_of_day // 15)
        interval_indices = np.clip(interval_indices, 0, 95)

        # Get daily factors using advanced indexing
        daily_factor = daily_pattern_array[interval_indices]
        daily_factor = np.clip(daily_factor, 0.0, 1.0)

        # Humidity influence (vectorized)
        humidity_factor = np.ones_like(humidity, dtype=float)
        if device_name in ['air_conditioner', 'refrigeration']:
            humidity_factor = 1 + (humidity - 50) * 0.002
            humidity_factor = np.clip(humidity_factor, 0.8, 1.3)

        # Random variation (vectorized)
        random_factor = 1 + np.random.normal(0, 0.05, size=temperature.shape)
        random_factor = np.clip(random_factor, 0.5, 1.5)

        # Calculate total power consumption (vectorized)
        base_consumption = peak_power * daily_factor
        total_power = base_consumption * temp_factor * humidity_factor * random_factor

        # Apply maximum power limit
        max_allowed_power = peak_power * 1.2
        total_power = np.clip(total_power, 0, max_allowed_power)

        return total_power

    def calculate_total_load(self, weather_data: pd.DataFrame, devices: List[str],
                             quantities: Dict[str, int] = None) -> pd.DataFrame:
        """Calculate total load for multiple devices over time."""

        if quantities is None:
            quantities = {device: 1 for device in devices}

        self.logger.info(f"Calculating load for devices: {devices}")

        # Use vectorized approach for better performance
        df = weather_data.copy()

        # Extract time features as numpy arrays for vectorized operations
        timestamps = df.index
        temperatures = df['temperature'].values
        humidity_values = df.get('humidity', pd.Series(50, index=df.index)).values
        conditions = df.get('condition', pd.Series('Unknown', index=df.index)).values

        # Pre-calculate time features
        hours = df.index.hour.values
        minutes = df.index.minute.values
        days_of_year = df.index.dayofyear.values

        n_records = len(df)

        # Pre-allocate result arrays
        total_power = np.zeros(n_records)
        device_powers = {}

        # Calculate power for each device using vectorized operations
        for device in devices:
            device_quantity = quantities.get(device, 1)

            # Log progress for large datasets
            if n_records > 10000:
                self.logger.info(f"Processing device: {device}")

            # Use vectorized calculation
            device_power = self._calculate_device_load_vectorized(
                device, temperatures, hours, minutes, days_of_year, humidity_values
            ) * device_quantity

            device_powers[f'{device}_power'] = device_power
            total_power += device_power

        # Build result DataFrame with same structure as original
        load_data = []
        for idx in range(n_records):
            # Progress logging to match original behavior
            if idx % 5000 == 0 and idx > 0:
                progress = (idx / n_records) * 100
                self.logger.info(f"Processing... {progress:.1f}% complete")

            load_entry = {
                'datetime': timestamps[idx],
                'temperature': temperatures[idx],
                'humidity': humidity_values[idx],
                'condition': conditions[idx],
                'total_power': total_power[idx],
                'hour_of_day': hours[idx],
                'minute_of_day': minutes[idx],
                'day_of_year': days_of_year[idx]
            }

            # Add individual device powers
            for device in devices:
                load_entry[f'{device}_power'] = device_powers[f'{device}_power'][idx]

            load_data.append(load_entry)

        df = pd.DataFrame(load_data)
        df.set_index('datetime', inplace=True)

        self.logger.info(f"Generated load profile with {len(df):,} records")
        return df

    def get_device_statistics(self, load_data: pd.DataFrame, device_name: str) -> Dict:
        """Get statistics for a specific device."""
        device_column = f'{device_name}_power'

        if device_column not in load_data.columns:
            return {}

        device_data = load_data[device_column]
        peak_power = self.device_configs[device_name]['peak_power']

        return {
            'average_power_w': device_data.mean(),
            'max_power_w': device_data.max(),
            'min_power_w': device_data.min(),
            'total_energy_kwh': device_data.sum() * 0.25 / 1000,  # 15-min intervals
            'capacity_factor': device_data.mean() / peak_power,
            'peak_capacity_factor': device_data.max() / peak_power,
            'peak_to_average_ratio': device_data.max() / device_data.mean() if device_data.mean() > 0 else 0,
            'configured_peak_power_w': peak_power
        }

