from device_calculator import DeviceLoadCalculator
import numpy as np
import pandas as pd
from datetime import datetime

def generate_test_data(num_records=250000):
    """Generate test data for profiling."""
    print(f"Generating {num_records:,} test records...")

    # Create datetime index
    dates = pd.date_range(start='2024-01-01', periods=num_records, freq='15min')

    # Generate realistic data
    np.random.seed(42)  # Reproducible results

    # Temperature with seasonal variation
    day_of_year = dates.dayofyear
    seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_temp = 8 * np.sin(2 * np.pi * dates.hour / 24)
    temp_noise = np.random.normal(0, 3, num_records)
    temperatures = seasonal_temp + daily_temp + temp_noise

    # Humidity
    base_humidity = 60 + 20 * np.sin(2 * np.pi * (day_of_year - 120) / 365)
    humidity_noise = np.random.normal(0, 10, num_records)
    humidity = np.clip(base_humidity + humidity_noise, 30, 90)

    # Conditions
    conditions = np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Sunny'],
                                  size=num_records, p=[0.4, 0.3, 0.1, 0.2])

    return pd.DataFrame({
        'temperature': temperatures,
        'humidity': humidity,
        'condition': conditions
    }, index=dates)

def create_device_configs():
    """Create sample device configurations."""
    return {
        'air_conditioner': {
            'peak_power': 3500,
            'comfort_temp': 24,
            'temp_coefficient': 150,
            'enabled': True,
            'daily_pattern': [0.3 + 0.4 * (i % 4 == 0) for i in range(96)]
        },
        'heating_system': {
            'peak_power': 4000,
            'comfort_temp': 18,
            'temp_coefficient': -120,
            'enabled': True,
            'daily_pattern': [0.2 + 0.6 * (i // 4 in [6, 7, 8, 18, 19, 20]) for i in range(96)]
        },
        'refrigerator': {
            'peak_power': 150,
            'comfort_temp': 20,
            'temp_coefficient': 5,
            'enabled': True,
            'daily_pattern': [0.7] * 96
        },
        'water_heater': {
            'peak_power': 3000,
            'comfort_temp': 15,
            'temp_coefficient': 50,
            'enabled': True,
            'daily_pattern': [0.1 + 0.7 * (i // 4 in [6, 7, 18, 19, 20]) for i in range(96)]
        }
    }

def main():
    """Main function for PyCharm profiling."""
    print("Starting Energy Load Profile Generator Profiling...")
    print(f"Time: {datetime.now()}")

    # Generate test data
    weather_data = generate_test_data(250000)  # Adjust size as needed
    print(f"Generated weather data: {len(weather_data):,} records")

    # Setup devices
    device_configs = create_device_configs()
    devices = ['air_conditioner', 'heating_system', 'refrigerator', 'water_heater']
    quantities = {'air_conditioner': 2, 'heating_system': 1, 'refrigerator': 1, 'water_heater': 1}

    # Create calculator
    calculator = DeviceLoadCalculator(device_configs)

    # This is the main function we want to profile
    print("Starting load calculation...")
    result = calculator.calculate_total_load(weather_data, devices, quantities)

    print(f"Calculation complete!")
    print(f"Result shape: {result.shape}")
    print(f"Total power mean: {result['total_power'].mean():.1f}W")
    print(f"Total power max: {result['total_power'].max():.1f}W")

    return result

if __name__ == "__main__":
    main()