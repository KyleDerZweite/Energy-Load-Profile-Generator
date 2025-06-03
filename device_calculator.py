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
                            hour_of_day: int, day_of_year: int, 
                            humidity: float = 50) -> float:
        """Calculate power consumption for a specific device."""
        
        if device_name not in self.device_configs:
            self.logger.warning(f"Device '{device_name}' not configured")
            return 0.0
        
        config = self.device_configs[device_name]
        
        if not config.get('enabled', True):
            return 0.0
        
        # Base power consumption
        base_power = config['base_power']
        
        # Temperature influence
        temp_diff = temperature - config['comfort_temp']
        temp_coefficient = config.get('temp_coefficient', 0)
        temp_factor = 1 + (temp_diff * temp_coefficient / base_power)
        temp_factor = max(0.1, temp_factor)  # Minimum 10% of base power
        
        # Daily pattern influence (24-hour pattern)
        daily_pattern = config.get('daily_pattern', [1.0] * 24)
        if len(daily_pattern) != 24:
            self.logger.warning(f"Invalid daily pattern for {device_name}, using default")
            daily_pattern = [1.0] * 24
        
        daily_factor = daily_pattern[hour_of_day]
        
        # Seasonal influence
        seasonal_factor = config.get('seasonal_factor', 1.0)
        if isinstance(seasonal_factor, (list, tuple)):
            # Interpolate seasonal factor based on day of year
            season_index = (day_of_year - 1) * len(seasonal_factor) / 365
            seasonal_factor = np.interp(season_index, range(len(seasonal_factor)), seasonal_factor)
        else:
            # Simple seasonal adjustment based on month
            month = datetime(2024, 1, 1) + pd.Timedelta(days=day_of_year-1)
            month_num = month.month
            if device_name in ['heater', 'lighting']:
                # More usage in winter months
                winter_boost = 1.5 if month_num in [12, 1, 2] else 1.0
                seasonal_factor *= winter_boost
            elif device_name == 'air_conditioner':
                # More usage in summer months
                summer_boost = 1.3 if month_num in [6, 7, 8] else 0.3
                seasonal_factor *= summer_boost
        
        # Humidity influence (for some devices)
        humidity_factor = 1.0
        if device_name in ['air_conditioner', 'refrigeration']:
            # Higher humidity increases load
            humidity_factor = 1 + (humidity - 50) * 0.002  # 0.2% per humidity percent
            humidity_factor = max(0.8, min(1.3, humidity_factor))
        
        # Add realistic random variation (±5%)
        random_factor = 1 + np.random.normal(0, 0.05)
        random_factor = max(0.5, min(1.5, random_factor))  # Clamp to reasonable range
        
        # Calculate total power
        total_power = (base_power * temp_factor * daily_factor * 
                      seasonal_factor * humidity_factor * random_factor)
        
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
            day_of_year = timestamp.timetuple().tm_yday
            
            total_power = 0
            device_powers = {}
            
            for device in devices:
                device_quantity = quantities.get(device, 1)
                device_power = self.calculate_device_load(
                    device, temperature, hour_of_day, day_of_year, humidity
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
        
        return {
            'average_power_w': device_data.mean(),
            'max_power_w': device_data.max(),
            'min_power_w': device_data.min(),
            'total_energy_kwh': device_data.sum() * 0.25 / 1000,  # 15-min intervals
            'capacity_factor': device_data.mean() / self.device_configs[device_name]['base_power'],
            'peak_to_average_ratio': device_data.max() / device_data.mean() if device_data.mean() > 0 else 0
        }