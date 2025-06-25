"""
Enhanced Weather-Energy Correlation Test
=======================================

Test weather-energy correlations with a more realistic energy model that includes
proper temperature dependencies to validate weather correlation analysis.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from weather_energy_analyzer import WeatherEnergyAnalyzer
import matplotlib.pyplot as plt


def create_realistic_energy_data(weather_data):
    """Create realistic energy data with strong weather correlations."""
    if weather_data.empty:
        return pd.DataFrame()
    
    # Align timestamps
    weather_data = weather_data.copy()
    weather_data.set_index('Timestamp', inplace=True)
    
    # Base building load (continuous infrastructure)
    base_load = 35  # kW
    
    # Weather-dependent components
    energy_components = []
    
    for timestamp, row in weather_data.iterrows():
        temp = row['temperature']
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # 1. Base load (weather-independent)
        base = base_load
        
        # 2. Heating load (significant below 15°C)
        if temp < 15.0:
            heating_dd = max(0, 15.0 - temp)
            heating_load = heating_dd * 2.5  # 2.5 kW per degree day
        else:
            heating_load = 0
        
        # 3. Cooling load (significant above 22°C)
        if temp > 22.0:
            cooling_dd = max(0, temp - 22.0)
            cooling_load = cooling_dd * 3.0  # 3.0 kW per degree day
        else:
            cooling_load = 0
        
        # 4. Occupancy-dependent load (office equipment, lighting)
        if day_of_week < 5:  # Weekdays
            if 7 <= hour <= 19:  # Business hours
                occupancy_load = 20
            elif 6 <= hour <= 21:  # Extended hours
                occupancy_load = 10
            else:  # Night
                occupancy_load = 2
        else:  # Weekends
            occupancy_load = 3
        
        # 5. Seasonal adjustment
        if month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.1
        elif month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.05
        else:  # Spring/Fall
            seasonal_factor = 1.0
        
        # Total energy
        total_energy = (base + heating_load + cooling_load + occupancy_load) * seasonal_factor
        
        # Add realistic noise (±3%)
        noise = np.random.normal(0, total_energy * 0.03)
        total_energy += noise
        
        # Ensure minimum load
        total_energy = max(total_energy, 25)
        
        energy_components.append({
            'Timestamp': timestamp,
            'Value': total_energy,
            'base_load': base * seasonal_factor,
            'heating_load': heating_load * seasonal_factor,
            'cooling_load': cooling_load * seasonal_factor,
            'occupancy_load': occupancy_load * seasonal_factor,
            'temperature': temp
        })
    
    return pd.DataFrame(energy_components)


def analyze_and_plot_correlations(energy_data, weather_data, data_label, save_plots=False):
    """Analyze correlations and create visualization."""
    print(f"\n=== {data_label} ===")
    
    if weather_data.empty or energy_data.empty:
        print("Insufficient data for analysis")
        return None
    
    # Prepare data for analysis
    energy_analysis = energy_data[['Timestamp', 'Value']].copy()
    weather_analysis = weather_data.copy()
    
    analyzer = WeatherEnergyAnalyzer()
    
    try:
        results = analyzer.analyze_weather_energy_relationship(energy_analysis, weather_analysis)
        
        print(f"Correlation Analysis:")
        corr = results.correlation_analysis
        print(f"  Linear correlation: {corr['linear_correlation']:.3f}")
        print(f"  Heating degree day correlation: {corr['heating_degree_day_correlation']:.3f}")
        print(f"  Cooling degree day correlation: {corr['cooling_degree_day_correlation']:.3f}")
        
        print(f"\nDegree Day Analysis:")
        dd = results.degree_day_analysis
        print(f"  Heating R²: {dd['heating']['r_squared']:.3f}")
        print(f"  Cooling R²: {dd['cooling']['r_squared']:.3f}")
        print(f"  Combined R²: {dd['combined']['r_squared']:.3f}")
        
        print(f"\nSignatures Found:")
        if results.heating_signature:
            print(f"  Heating: Coeff={results.heating_signature.coefficient:.2f} kW/°C, R²={results.heating_signature.r_squared:.3f}")
        else:
            print("  Heating: No significant signature found")
            
        if results.cooling_signature:
            print(f"  Cooling: Coeff={results.cooling_signature.coefficient:.2f} kW/°C, R²={results.cooling_signature.r_squared:.3f}")
        else:
            print("  Cooling: No significant signature found")
        
        # Create visualization if requested
        if save_plots:
            create_correlation_plots(energy_data, weather_data, data_label)
        
        return results
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_correlation_plots(energy_data, weather_data, data_label):
    """Create correlation visualization plots."""
    try:
        # Merge data for plotting
        plot_data = pd.merge(energy_data, weather_data, on='Timestamp', how='inner')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Weather-Energy Correlations: {data_label}', fontsize=16)
        
        # Plot 1: Energy vs Temperature scatter
        axes[0, 0].scatter(plot_data['temperature'], plot_data['Value'], alpha=0.6, s=1)
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Energy (kW)')
        axes[0, 0].set_title('Energy vs Temperature')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Heating degree days vs Energy
        hdd = np.maximum(0, 15.0 - plot_data['temperature'])
        heating_mask = hdd > 0
        if heating_mask.sum() > 0:
            axes[0, 1].scatter(hdd[heating_mask], plot_data['Value'][heating_mask], alpha=0.6, s=1, color='red')
            axes[0, 1].set_xlabel('Heating Degree Days')
            axes[0, 1].set_ylabel('Energy (kW)')
            axes[0, 1].set_title('Heating Degree Days vs Energy')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cooling degree days vs Energy
        cdd = np.maximum(0, plot_data['temperature'] - 22.0)
        cooling_mask = cdd > 0
        if cooling_mask.sum() > 0:
            axes[1, 0].scatter(cdd[cooling_mask], plot_data['Value'][cooling_mask], alpha=0.6, s=1, color='blue')
            axes[1, 0].set_xlabel('Cooling Degree Days')
            axes[1, 0].set_ylabel('Energy (kW)')
            axes[1, 0].set_title('Cooling Degree Days vs Energy')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Time series (first week)
        week_data = plot_data.head(672)  # 7 days * 96 (15-min intervals)
        axes[1, 1].plot(week_data['Timestamp'], week_data['Value'], label='Energy', linewidth=1)
        ax2 = axes[1, 1].twinx()
        ax2.plot(week_data['Timestamp'], week_data['temperature'], color='orange', label='Temperature', linewidth=1)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Energy (kW)', color='blue')
        ax2.set_ylabel('Temperature (°C)', color='orange')
        axes[1, 1].set_title('Energy and Temperature (First Week)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        for ax in axes.flat:
            if hasattr(ax, 'tick_params'):
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"correlation_analysis_{data_label.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Correlation plots saved to {filename}")
        
    except Exception as e:
        print(f"  ⚠️ Plot creation failed: {e}")


def load_weather_data(db_path: str, location: str = "Bottrop"):
    """Load weather data from database."""
    try:
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
    except Exception as e:
        print(f"Error loading weather data from {db_path}: {e}")
        return pd.DataFrame()


def main():
    """Main enhanced correlation test."""
    logging.basicConfig(level=logging.WARNING)
    
    print("Enhanced Weather-Energy Correlation Analysis")
    print("===========================================")
    
    # Load weather data
    print("Loading weather data...")
    dirty_weather = load_weather_data('weather_data.db')
    clean_weather = load_weather_data('weather_data_clean.db')
    
    if dirty_weather.empty and clean_weather.empty:
        print("ERROR: No weather data available")
        return
    
    # Use clean weather for energy generation if available, otherwise dirty
    reference_weather = clean_weather if not clean_weather.empty else dirty_weather
    
    print(f"Generating realistic energy data with weather correlations...")
    energy_data = create_realistic_energy_data(reference_weather)
    
    if energy_data.empty:
        print("ERROR: Failed to generate energy data")
        return
    
    print(f"Energy data generated: {len(energy_data)} records")
    print(f"Energy range: {energy_data['Value'].min():.1f} to {energy_data['Value'].max():.1f} kW")
    print(f"Mean energy: {energy_data['Value'].mean():.1f} kW")
    
    # Analyze with dirty data if available
    if not dirty_weather.empty:
        dirty_results = analyze_and_plot_correlations(
            energy_data, dirty_weather, "DIRTY WEATHER DATA", save_plots=True
        )
    else:
        dirty_results = None
    
    # Analyze with clean data if available
    if not clean_weather.empty:
        clean_results = analyze_and_plot_correlations(
            energy_data, clean_weather, "CLEAN WEATHER DATA", save_plots=True
        )
    else:
        clean_results = None
    
    # Comparison summary
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*60)
    
    if dirty_results and clean_results:
        dirty_linear = dirty_results.correlation_analysis['linear_correlation']
        clean_linear = clean_results.correlation_analysis['linear_correlation']
        
        dirty_hdd = dirty_results.correlation_analysis['heating_degree_day_correlation']
        clean_hdd = clean_results.correlation_analysis['heating_degree_day_correlation']
        
        dirty_cdd = dirty_results.correlation_analysis['cooling_degree_day_correlation']
        clean_cdd = clean_results.correlation_analysis['cooling_degree_day_correlation']
        
        print(f"Linear correlation: {dirty_linear:.3f} → {clean_linear:.3f} ({clean_linear-dirty_linear:+.3f})")
        print(f"Heating correlation: {dirty_hdd:.3f} → {clean_hdd:.3f} ({clean_hdd-dirty_hdd:+.3f})")
        print(f"Cooling correlation: {clean_cdd:.3f} → {clean_cdd:.3f} ({clean_cdd-dirty_cdd:+.3f})")
        
        improvement_score = abs(clean_linear) - abs(dirty_linear)
        if improvement_score > 0.05:
            print(f"\\n✅ Significant correlation improvement: +{improvement_score:.3f}")
        elif improvement_score > 0.01:
            print(f"\\n🔄 Moderate correlation improvement: +{improvement_score:.3f}")
        else:
            print(f"\\n⚠️ Minimal correlation change: {improvement_score:+.3f}")
    
    elif clean_results:
        print("Only clean weather data analysis completed:")
        corr = clean_results.correlation_analysis
        print(f"Linear correlation: {corr['linear_correlation']:.3f}")
        print(f"Heating correlation: {corr['heating_degree_day_correlation']:.3f}")
        print(f"Cooling correlation: {corr['cooling_degree_day_correlation']:.3f}")
        
        if abs(corr['linear_correlation']) > 0.3:
            print("\\n✅ Strong weather-energy correlations detected")
        elif abs(corr['linear_correlation']) > 0.1:
            print("\\n🔄 Moderate weather-energy correlations detected")
        else:
            print("\\n⚠️ Weak weather-energy correlations")
    
    print("\\nAnalysis complete. Check generated plots for visual correlation assessment.")


if __name__ == "__main__":
    main()