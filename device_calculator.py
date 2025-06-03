import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import logging

class DeviceLoadCalculator:
    """Calculates energy load for various electrical devices based on weather and usage patterns."""

    def __init__(self, device_configs: Dict):
        self.device_configs = device_configs
        self.logger = logging.getLogger(__name__)

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
        daily_pattern = config.get('daily_pattern', [1.0] * 96)
        if len(daily_pattern) != 96:
            self.logger.warning(f"Invalid daily pattern for {device_name}, expected 96 values (15-min intervals), got {len(daily_pattern)}")
            # Create a simple pattern based on hourly if available, otherwise default
            if len(daily_pattern) == 24:
                # Convert 24-hour pattern to 96 x 15-minute pattern
                extended_pattern = []
                for hour_value in daily_pattern:
                    extended_pattern.extend([hour_value] * 4)  # 4 x 15-min intervals per hour
                daily_pattern = extended_pattern
            else:
                daily_pattern = [1.0] * 96

        # Calculate 15-minute interval index (0-95)
        interval_index = (hour_of_day * 4) + (minute_of_day // 15)
        interval_index = min(95, max(0, interval_index))  # Ensure valid index

        # Ensure daily pattern values are between 0.0 and 1.0
        daily_factor = max(0.0, min(1.0, daily_pattern[interval_index]))

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

    def calculate_total_load(self, weather_data: pd.DataFrame, devices: List[str],
                             quantities: Dict[str, int] = None) -> pd.DataFrame:
        """Calculate total load for multiple devices over time."""

        if quantities is None:
            quantities = {device: 1 for device in devices}

        self.logger.info(f"Calculating load for devices: {devices}")

        load_data = []
        total_records = len(weather_data)

        for idx, (timestamp, row) in enumerate(weather_data.iterrows()):
            if idx % 5000 == 0 and idx > 0:
                progress = (idx / total_records) * 100
                self.logger.info(f"Processing... {progress:.1f}% complete")

            temperature = row['temperature']
            humidity = row.get('humidity', 50)
            hour_of_day = timestamp.hour
            minute_of_day = timestamp.minute
            day_of_year = timestamp.timetuple().tm_yday

            total_power = 0
            device_powers = {}

            for device in devices:
                device_quantity = quantities.get(device, 1)
                device_power = self.calculate_device_load(
                    device, temperature, hour_of_day, minute_of_day, day_of_year, humidity
                ) * device_quantity

                device_powers[f'{device}_power'] = device_power
                total_power += device_power

            load_entry = {
                'datetime': timestamp,
                'temperature': temperature,
                'humidity': humidity,
                'condition': row.get('condition', 'Unknown'),
                'total_power': total_power,
                'hour_of_day': hour_of_day,
                'minute_of_day': minute_of_day,
                'day_of_year': day_of_year
            }
            load_entry.update(device_powers)
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