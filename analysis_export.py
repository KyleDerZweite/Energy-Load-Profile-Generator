import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os

class LoadProfileAnalyzer:
    """Analyze and export energy load profile data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_load_profile(self, load_data: pd.DataFrame) -> Dict:
        """Comprehensive analysis of the load profile."""
        
        self.logger.info("Performing load profile analysis...")
        
        analysis = {
            'basic_statistics': self._calculate_basic_stats(load_data),
            'temporal_patterns': self._analyze_temporal_patterns(load_data),
            'weather_correlation': self._analyze_weather_correlation(load_data),
            'peak_analysis': self._analyze_peaks(load_data),
            'device_breakdown': self._analyze_device_breakdown(load_data)
        }
        
        return analysis
    
    def _calculate_basic_stats(self, load_data: pd.DataFrame) -> Dict:
        """Calculate basic load statistics."""
        total_power = load_data['total_power']
        
        # Time intervals (assuming 15-minute data)
        interval_hours = 0.25
        
        stats = {
            'total_energy_kwh': total_power.sum() * interval_hours / 1000,
            'average_power_w': total_power.mean(),
            'max_power_w': total_power.max(),
            'min_power_w': total_power.min(),
            'std_power_w': total_power.std(),
            'peak_to_average_ratio': total_power.max() / total_power.mean(),
            'load_factor': total_power.mean() / total_power.max(),
            'data_points': len(load_data),
            'time_span_days': (load_data.index.max() - load_data.index.min()).days,
            'temperature_range': {
                'min': load_data['temperature'].min(),
                'max': load_data['temperature'].max(),
                'mean': load_data['temperature'].mean(),
                'std': load_data['temperature'].std()
            }
        }
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        stats['power_percentiles'] = {
            f'p{p}': total_power.quantile(p/100) for p in percentiles
        }
        
        return stats
    
    def _analyze_temporal_patterns(self, load_data: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in energy consumption."""
        patterns = {}
        
        # Hourly patterns
        if self.config.get('analysis', {}).get('include_hourly_patterns', True):
            load_data['hour'] = load_data.index.hour
            hourly_avg = load_data.groupby('hour')['total_power'].agg(['mean', 'std', 'min', 'max'])
            patterns['hourly'] = hourly_avg.to_dict('index')
        
        # Daily patterns (day of week)
        load_data['weekday'] = load_data.index.day_name()
        daily_avg = load_data.groupby('weekday')['total_power'].agg(['mean', 'std'])
        patterns['weekly'] = daily_avg.to_dict('index')
        
        # Monthly patterns
        if self.config.get('analysis', {}).get('include_monthly_patterns', True):
            load_data['month'] = load_data.index.month
            monthly_avg = load_data.groupby('month')['total_power'].agg(['mean', 'std', 'sum'])
            patterns['monthly'] = monthly_avg.to_dict('index')
        
        # Seasonal patterns
        if self.config.get('analysis', {}).get('include_seasonal_analysis', True):
            def get_season(month):
                if month in [12, 1, 2]: return 'Winter'
                elif month in [3, 4, 5]: return 'Spring'
                elif month in [6, 7, 8]: return 'Summer'
                else: return 'Autumn'
            
            load_data['season'] = load_data['month'].apply(get_season)
            seasonal_avg = load_data.groupby('season')['total_power'].agg(['mean', 'std', 'sum'])
            patterns['seasonal'] = seasonal_avg.to_dict('index')
        
        return patterns
    
    def _analyze_weather_correlation(self, load_data: pd.DataFrame) -> Dict:
        """Analyze correlation between weather and energy consumption."""
        if not self.config.get('analysis', {}).get('include_temperature_correlation', True):
            return {}
        
        correlations = {}
        
        # Temperature correlation
        temp_corr = load_data['total_power'].corr(load_data['temperature'])
        correlations['temperature'] = temp_corr
        
        # Humidity correlation (if available)
        if 'humidity' in load_data.columns:
            humidity_corr = load_data['total_power'].corr(load_data['humidity'])
            correlations['humidity'] = humidity_corr
        
        # Temperature bins analysis
        load_data['temp_bin'] = pd.cut(load_data['temperature'], 
                                      bins=10, labels=False, duplicates='drop')
        if 'temp_bin' in load_data.columns and not load_data['temp_bin'].isna().all():
            temp_bin_avg = load_data.groupby('temp_bin')['total_power'].mean()
            correlations['temperature_bins'] = temp_bin_avg.to_dict()
        
        return correlations
    
    def _analyze_peaks(self, load_data: pd.DataFrame) -> Dict:
        """Analyze peak consumption patterns."""
        if not self.config.get('analysis', {}).get('include_peak_analysis', True):
            return {}
        
        total_power = load_data['total_power']
        
        # Define peak thresholds
        p95 = total_power.quantile(0.95)
        p99 = total_power.quantile(0.99)
        
        peak_analysis = {
            'peak_threshold_95': p95,
            'peak_threshold_99': p99,
            'peak_events_95': (total_power >= p95).sum(),
            'peak_events_99': (total_power >= p99).sum(),
            'peak_duration_hours_95': (total_power >= p95).sum() * 0.25,
            'peak_duration_hours_99': (total_power >= p99).sum() * 0.25
        }
        
        # Peak timing analysis
        peak_data = load_data[total_power >= p95].copy()
        if not peak_data.empty:
            peak_analysis['peak_hours'] = peak_data.groupby('hour_of_day').size().to_dict()
            peak_analysis['peak_months'] = peak_data.groupby(peak_data.index.month).size().to_dict()
        
        return peak_analysis
    
    def _analyze_device_breakdown(self, load_data: pd.DataFrame) -> Dict:
        """Analyze individual device contributions."""
        device_columns = [col for col in load_data.columns if col.endswith('_power')]
        
        breakdown = {}
        total_consumption = load_data['total_power'].sum() * 0.25 / 1000  # kWh
        
        for device_col in device_columns:
            device_name = device_col.replace('_power', '')
            device_consumption = load_data[device_col].sum() * 0.25 / 1000
            
            breakdown[device_name] = {
                'total_kwh': device_consumption,
                'percentage': (device_consumption / total_consumption * 100) if total_consumption > 0 else 0,
                'average_power_w': load_data[device_col].mean(),
                'max_power_w': load_data[device_col].max(),
                'capacity_factor': load_data[device_col].mean() / load_data[device_col].max() if load_data[device_col].max() > 0 else 0
            }
        
        return breakdown
    
    def generate_plots(self, load_data: pd.DataFrame, output_dir: str = "plots"):
        """Generate visualization plots."""
        if not self.config.get('analysis', {}).get('generate_plots', True):
            return
        
        os.makedirs(output_dir, exist_ok=True)
        plot_format = self.config.get('analysis', {}).get('plot_format', 'png')
        plot_dpi = self.config.get('analysis', {}).get('plot_dpi', 300)
        
        self.logger.info(f"Generating plots in {output_dir}/")
        
        # 1. Overall load profile (sample period)
        self._plot_load_profile(load_data, output_dir, plot_format, plot_dpi)
        
        # 2. Daily patterns
        self._plot_daily_patterns(load_data, output_dir, plot_format, plot_dpi)
        
        # 3. Temperature correlation
        self._plot_temperature_correlation(load_data, output_dir, plot_format, plot_dpi)
        
        # 4. Device breakdown
        self._plot_device_breakdown(load_data, output_dir, plot_format, plot_dpi)
        
        # 5. Monthly patterns
        self._plot_monthly_patterns(load_data, output_dir, plot_format, plot_dpi)
    
    def _plot_load_profile(self, load_data: pd.DataFrame, output_dir: str, 
                          plot_format: str, plot_dpi: int):
        """Plot overall load profile."""
        plot_days = self.config.get('analysis', {}).get('plot_days', 14)
        sample_data = load_data.head(plot_days * 24 * 4)  # 15-min intervals
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Power consumption
        ax1.plot(sample_data.index, sample_data['total_power'], linewidth=1, alpha=0.8)
        ax1.set_ylabel('Power Consumption (W)')
        ax1.set_title(f'Energy Load Profile ({plot_days} days sample)')
        ax1.grid(True, alpha=0.3)
        
        # Temperature
        ax2.plot(sample_data.index, sample_data['temperature'], color='red', linewidth=1)
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_xlabel('Date and Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(f"{output_dir}/load_profile.{plot_format}", dpi=plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_daily_patterns(self, load_data: pd.DataFrame, output_dir: str, 
                           plot_format: str, plot_dpi: int):
        """Plot daily consumption patterns."""
        hourly_avg = load_data.groupby(load_data.index.hour)['total_power'].mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Power Consumption (W)')
        plt.title('Average Daily Consumption Pattern')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/daily_patterns.{plot_format}", dpi=plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_temperature_correlation(self, load_data: pd.DataFrame, output_dir: str, 
                                    plot_format: str, plot_dpi: int):
        """Plot temperature vs power correlation."""
        plt.figure(figsize=(10, 6))
        
        # Scatter plot with trend line
        x = load_data['temperature']
        y = load_data['total_power']
        
        plt.scatter(x, y, alpha=0.1, s=1)
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8, linewidth=2)
        
        correlation = x.corr(y)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Power Consumption (W)')
        plt.title(f'Temperature vs Power Consumption (Correlation: {correlation:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/temperature_correlation.{plot_format}", dpi=plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_device_breakdown(self, load_data: pd.DataFrame, output_dir: str, 
                             plot_format: str, plot_dpi: int):
        """Plot device consumption breakdown."""
        device_columns = [col for col in load_data.columns if col.endswith('_power')]
        
        if not device_columns:
            return
        
        # Calculate total consumption per device
        device_totals = {}
        for col in device_columns:
            device_name = col.replace('_power', '').replace('_', ' ').title()
            device_totals[device_name] = load_data[col].sum() * 0.25 / 1000  # kWh
        
        # Pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(device_totals.values(), labels=device_totals.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('Energy Consumption by Device')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/device_breakdown.{plot_format}", dpi=plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_monthly_patterns(self, load_data: pd.DataFrame, output_dir: str, 
                             plot_format: str, plot_dpi: int):
        """Plot monthly consumption patterns."""
        monthly_consumption = load_data.groupby(load_data.index.month)['total_power'].sum() * 0.25 / 1000
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(1, len(monthly_consumption) + 1), monthly_consumption.values)
        plt.xlabel('Month')
        plt.ylabel('Total Energy Consumption (kWh)')
        plt.title('Monthly Energy Consumption')
        plt.xticks(range(1, len(monthly_consumption) + 1), 
                  [month_names[i-1] for i in monthly_consumption.index])
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, monthly_consumption.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monthly_patterns.{plot_format}", dpi=plot_dpi, bbox_inches='tight')
        plt.close()
    
    def export_to_csv(self, load_data: pd.DataFrame, analysis: Dict, filename: str):
        """Export data to CSV format."""
        self.logger.info(f"Exporting to CSV: {filename}")
        
        # Main data export
        load_data.to_csv(filename)
        
        # Export analysis summary
        summary_filename = filename.replace('.csv', '_summary.csv')
        
        # Flatten analysis data for CSV export
        summary_data = []
        
        # Basic statistics
        for key, value in analysis.get('basic_statistics', {}).items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    summary_data.append({'Category': key, 'Metric': subkey, 'Value': subvalue})
            else:
                summary_data.append({'Category': 'basic_statistics', 'Metric': key, 'Value': value})
        
        # Device breakdown
        for device, stats in analysis.get('device_breakdown', {}).items():
            for metric, value in stats.items():
                summary_data.append({'Category': f'device_{device}', 'Metric': metric, 'Value': value})
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_filename, index=False)
        
        self.logger.info(f"Summary exported to: {summary_filename}")
    
    def export_to_excel(self, load_data: pd.DataFrame, analysis: Dict, filename: str):
        """Export data to Excel format with multiple sheets."""
        self.logger.info(f"Exporting to Excel: {filename}")
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            # Main data
            load_data.to_excel(writer, sheet_name='Load_Profile')
            
            # Basic statistics
            if 'basic_statistics' in analysis:
                stats_df = pd.DataFrame([analysis['basic_statistics']]).T
                stats_df.columns = ['Value']
                stats_df.to_excel(writer, sheet_name='Statistics')
            
            # Temporal patterns
            if 'temporal_patterns' in analysis:
                patterns = analysis['temporal_patterns']
                
                if 'hourly' in patterns:
                    hourly_df = pd.DataFrame(patterns['hourly']).T
                    hourly_df.to_excel(writer, sheet_name='Hourly_Patterns')
                
                if 'monthly' in patterns:
                    monthly_df = pd.DataFrame(patterns['monthly']).T
                    monthly_df.to_excel(writer, sheet_name='Monthly_Patterns')
            
            # Device breakdown
            if 'device_breakdown' in analysis:
                device_df = pd.DataFrame(analysis['device_breakdown']).T
                device_df.to_excel(writer, sheet_name='Device_Breakdown')
            
            # Weather correlation
            if 'weather_correlation' in analysis:
                corr_data = []
                for key, value in analysis['weather_correlation'].items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            corr_data.append({'Variable': key, 'Bin/Type': subkey, 'Correlation': subvalue})
                    else:
                        corr_data.append({'Variable': key, 'Bin/Type': 'Overall', 'Correlation': value})
                
                if corr_data:
                    corr_df = pd.DataFrame(corr_data)
                    corr_df.to_excel(writer, sheet_name='Weather_Correlation', index=False)
        
        self.logger.info(f"Excel export completed: {filename}")