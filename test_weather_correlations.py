"""
Weather-Energy Correlation Testing
=================================

Test script to compare weather-energy correlations between dirty and clean weather data
to validate the impact of quality control on energy disaggregation analysis.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from weather_energy_analyzer import WeatherEnergyAnalyzer


def load_energy_data():
    """Load energy consumption data for testing."""
    # For testing, we'll create synthetic but realistic energy data
    # In real use, this would load from load_profiles.xlsx
    
    # Create 2024 energy data with seasonal patterns
    dates = pd.date_range('2024-01-01', '2024-12-31 23:00:00', freq='15min')
    
    # Base load pattern (university building)
    base_load = 50  # kW
    
    # Seasonal variation
    seasonal_factor = []
    for date in dates:
        month = date.month
        if month in [12, 1, 2]:  # Winter
            factor = 1.3  # Higher heating load
        elif month in [6, 7, 8]:  # Summer
            factor = 1.1  # Higher cooling load
        else:  # Spring/Fall
            factor = 1.0
        seasonal_factor.append(factor)
    
    # Daily variation (business hours pattern)
    daily_factor = []
    for date in dates:
        hour = date.hour
        if 8 <= hour <= 18:  # Business hours
            factor = 1.4
        elif 6 <= hour <= 22:  # Extended hours
            factor = 1.1
        else:  # Night
            factor = 0.7
        daily_factor.append(factor)
    
    # Weekly variation (weekdays vs weekends)
    weekly_factor = []
    for date in dates:
        if date.weekday() < 5:  # Weekdays
            factor = 1.0
        else:  # Weekends
            factor = 0.6
        weekly_factor.append(factor)
    
    # Random noise
    noise = np.random.normal(0, 5, len(dates))
    
    # Combine all factors
    energy_values = (
        base_load * 
        np.array(seasonal_factor) * 
        np.array(daily_factor) * 
        np.array(weekly_factor) + 
        noise
    )
    
    # Ensure positive values
    energy_values = np.maximum(energy_values, 10)
    
    energy_df = pd.DataFrame({
        'Timestamp': dates,
        'Value': energy_values
    })
    
    return energy_df


def load_weather_data(db_path: str, location: str = "Bottrop"):
    """Load weather data from database."""
    with sqlite3.connect(db_path) as conn:
        query = '''
            SELECT datetime, temperature, humidity, condition, precipitation, weather_code
            FROM weather_data
            WHERE location LIKE ? AND datetime LIKE '2024%'
            ORDER BY datetime
        '''
        
        df = pd.read_sql_query(query, conn, params=[f'%{location}%'])
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.rename(columns={'datetime': 'Timestamp'})
        
        return df


def analyze_correlations(energy_data, weather_data, data_label):
    """Analyze weather-energy correlations."""
    print(f"\n=== {data_label} ===")
    
    if weather_data.empty:
        print("No weather data available")
        return None
    
    analyzer = WeatherEnergyAnalyzer()
    
    try:
        results = analyzer.analyze_weather_energy_relationship(energy_data, weather_data)
        
        print(f"Temperature Statistics:")
        temp_stats = results.temperature_statistics
        print(f"  Range: {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C")
        print(f"  Mean: {temp_stats['mean']:.1f}°C")
        print(f"  Std Dev: {temp_stats['std']:.1f}°C")
        
        print(f"\nEnergy Statistics:")
        energy_stats = results.energy_statistics
        print(f"  Range: {energy_stats['min']:.1f} to {energy_stats['max']:.1f} kW")
        print(f"  Mean: {energy_stats['mean']:.1f} kW")
        
        print(f"\nCorrelation Analysis:")
        corr = results.correlation_analysis
        print(f"  Linear correlation: {corr['linear_correlation']:.3f}")
        print(f"  Heating degree day correlation: {corr['heating_degree_day_correlation']:.3f}")
        print(f"  Cooling degree day correlation: {corr['cooling_degree_day_correlation']:.3f}")
        print(f"  Absolute temp difference correlation: {corr['abs_temp_diff_correlation']:.3f}")
        
        print(f"\nDegree Day Analysis:")
        dd = results.degree_day_analysis
        print(f"  Heating base temp: {dd['heating_base_temp']:.1f}°C")
        print(f"  Cooling base temp: {dd['cooling_base_temp']:.1f}°C")
        print(f"  Heating R²: {dd['heating']['r_squared']:.3f}")
        print(f"  Cooling R²: {dd['cooling']['r_squared']:.3f}")
        print(f"  Combined R²: {dd['combined']['r_squared']:.3f}")
        
        print(f"\nSignatures Found:")
        if results.heating_signature:
            print(f"  Heating: Coeff={results.heating_signature.coefficient:.3f}, R²={results.heating_signature.r_squared:.3f}")
        else:
            print("  Heating: No significant signature found")
            
        if results.cooling_signature:
            print(f"  Cooling: Coeff={results.cooling_signature.coefficient:.3f}, R²={results.cooling_signature.r_squared:.3f}")
        else:
            print("  Cooling: No significant signature found")
        
        return results
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None


def main():
    """Main comparison function."""
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("Weather-Energy Correlation Analysis")
    print("===================================")
    
    # Load energy data
    print("Loading energy data...")
    energy_data = load_energy_data()
    print(f"Energy data: {len(energy_data)} records from {energy_data['Timestamp'].min()} to {energy_data['Timestamp'].max()}")
    
    # Load dirty weather data
    print("\nLoading dirty weather data...")
    dirty_weather = load_weather_data('weather_data.db')
    print(f"Dirty weather data: {len(dirty_weather)} records")
    
    # Load clean weather data
    print("\nLoading clean weather data...")
    clean_weather = load_weather_data('weather_data_clean.db')
    print(f"Clean weather data: {len(clean_weather)} records")
    
    # Analyze correlations with dirty data
    dirty_results = analyze_correlations(energy_data, dirty_weather, "DIRTY WEATHER DATA ANALYSIS")
    
    # Analyze correlations with clean data
    clean_results = analyze_correlations(energy_data, clean_weather, "CLEAN WEATHER DATA ANALYSIS")
    
    # Summary comparison
    print("\n" + "="*60)
    print("QUALITY IMPROVEMENT SUMMARY")
    print("="*60)
    
    if dirty_results and clean_results:
        dirty_linear = dirty_results.correlation_analysis['linear_correlation']
        clean_linear = clean_results.correlation_analysis['linear_correlation']
        
        dirty_hdd = dirty_results.correlation_analysis['heating_degree_day_correlation']
        clean_hdd = clean_results.correlation_analysis['heating_degree_day_correlation']
        
        dirty_cdd = dirty_results.correlation_analysis['cooling_degree_day_correlation']
        clean_cdd = clean_results.correlation_analysis['cooling_degree_day_correlation']
        
        print(f"Linear correlation improvement: {dirty_linear:.3f} → {clean_linear:.3f} ({clean_linear-dirty_linear:+.3f})")
        print(f"Heating correlation improvement: {dirty_hdd:.3f} → {clean_hdd:.3f} ({clean_hdd-dirty_hdd:+.3f})")
        print(f"Cooling correlation improvement: {dirty_cdd:.3f} → {clean_cdd:.3f} ({clean_cdd-dirty_cdd:+.3f})")
        
        # Temperature range comparison
        dirty_temp_range = dirty_results.temperature_statistics['max'] - dirty_results.temperature_statistics['min']
        clean_temp_range = clean_results.temperature_statistics['max'] - clean_results.temperature_statistics['min']
        
        print(f"\nTemperature range: {dirty_temp_range:.1f}°C → {clean_temp_range:.1f}°C")
        print(f"Temperature std dev: {dirty_results.temperature_statistics['std']:.1f}°C → {clean_results.temperature_statistics['std']:.1f}°C")
        
        if abs(clean_linear) > abs(dirty_linear):
            print("\n✅ Weather data quality control IMPROVED correlations")
        else:
            print("\n⚠️ Weather data quality control did not improve correlations significantly")
    else:
        print("Could not complete comparison due to analysis failures")


if __name__ == "__main__":
    main()