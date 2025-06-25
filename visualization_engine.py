"""
Advanced Visualization Engine for Energy Load Profile Generator
Creates comprehensive visual outputs and structured directories for energy analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Set matplotlib parameters for better plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

@dataclass
class VisualizationConfig:
    """Configuration for visualization output."""
    output_base_dir: str = "output"
    create_timestamp_dirs: bool = True
    plot_formats: List[str] = None
    excel_format: bool = True
    create_summary_report: bool = True
    color_palette: str = "Set2"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    
    def __post_init__(self):
        if self.plot_formats is None:
            self.plot_formats = ['png', 'pdf']

class EnergyVisualizationEngine:
    """
    Advanced visualization engine for energy disaggregation results.
    Creates comprehensive plots, structured outputs, and analysis reports.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set up seaborn style
        sns.set_style("whitegrid")
        sns.set_palette(self.config.color_palette)
        
        # Color schemes for different device categories
        self.category_colors = {
            'buero_verwaltung': '#2E86AB',
            'lehre_seminar': '#A23B72', 
            'labor_makerspace': '#F18F01',
            'allgemeine_infrastruktur': '#C73E1D'
        }
        
    def create_output_structure(self, location: str, start_date: str, end_date: str) -> str:
        """Create structured output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.create_timestamp_dirs:
            dir_name = f"energy_profiles_{location.lower().replace(' ', '_').replace(',', '')}_{start_date}_{end_date}_{timestamp}"
        else:
            dir_name = f"energy_profiles_{location.lower().replace(' ', '_').replace(',', '')}_{start_date}_{end_date}"
            
        output_dir = Path(self.config.output_base_dir) / dir_name
        
        # Create subdirectories
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (output_dir / "plots" / "devices").mkdir(exist_ok=True)
        (output_dir / "plots" / "analysis").mkdir(exist_ok=True)
        (output_dir / "plots" / "comparison").mkdir(exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
        (output_dir / "reports").mkdir(exist_ok=True)
        
        self.output_dir = output_dir
        self.logger.info(f"📁 Created output structure: {output_dir}")
        return str(output_dir)
    
    def visualize_disaggregation_results(self, 
                                       result,
                                       location: str,
                                       start_date: str, 
                                       end_date: str,
                                       weather_data: pd.DataFrame = None,
                                       devices_config: Dict = None) -> str:
        """
        Create comprehensive visualization suite for disaggregation results.
        
        Args:
            result: EnergyDisaggregationResult object
            location: Location string for naming
            start_date: Start date string
            end_date: End date string
            weather_data: Optional weather data for correlation plots
            devices_config: Device configuration from devices.json
            
        Returns:
            Path to output directory
        """
        # Create output structure
        output_dir = self.create_output_structure(location, start_date, end_date)
        
        # Extract device profiles and timestamps
        device_profiles = result.device_profiles
        timestamps = result.timestamps
        total_actual = result.total_actual
        total_predicted = result.total_predicted
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(device_profiles)
        df['timestamp'] = pd.to_datetime(timestamps)
        df['total_actual'] = total_actual
        df['total_predicted'] = total_predicted
        df.set_index('timestamp', inplace=True)
        
        # Load device categories if available
        device_categories = {}
        if devices_config and 'devices' in devices_config:
            for device_id, device_info in devices_config['devices'].items():
                device_categories[device_id] = device_info.get('category', 'unknown')
        
        # 1. Create overview plots
        self._create_overview_plots(df, device_categories)
        
        # 2. Create individual device plots
        self._create_device_plots(df, device_categories, devices_config)
        
        # 3. Create analysis plots
        self._create_analysis_plots(df, device_categories, weather_data)
        
        # 4. Create comparison plots
        self._create_comparison_plots(df, device_categories)
        
        # 5. Export data to Excel
        self._export_data_to_excel(df, device_categories, devices_config)
        
        # 6. Create summary report
        if self.config.create_summary_report:
            self._create_summary_report(df, device_categories, result, location, start_date, end_date)
        
        self.logger.info(f"✅ Visualization suite completed: {output_dir}")
        return output_dir
    
    def _create_overview_plots(self, df: pd.DataFrame, device_categories: Dict):
        """Create overview plots showing total energy and device breakdown."""
        
        # 1. Total Energy Overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Energy Load Profile Overview', fontsize=16, fontweight='bold')
        
        # Total energy comparison
        axes[0, 0].plot(df.index, df['total_actual'], label='Actual', linewidth=2, alpha=0.8)
        axes[0, 0].plot(df.index, df['total_predicted'], label='Predicted', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Total Energy: Actual vs Predicted')
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].legend()
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Error analysis
        error = df['total_predicted'] - df['total_actual']
        axes[0, 1].plot(df.index, error, color='red', alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Prediction Error')
        axes[0, 1].set_ylabel('Error (kW)')
        axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Daily energy consumption
        daily_energy = df.resample('D').mean()
        axes[1, 0].plot(daily_energy.index, daily_energy['total_actual'], 'o-', alpha=0.7, label='Actual')
        axes[1, 0].plot(daily_energy.index, daily_energy['total_predicted'], 's-', alpha=0.7, label='Predicted')
        axes[1, 0].set_title('Daily Average Energy Consumption')
        axes[1, 0].set_ylabel('Power (kW)')
        axes[1, 0].legend()
        axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Hourly pattern
        hourly_pattern = df.groupby(df.index.hour).mean()
        axes[1, 1].plot(hourly_pattern.index.astype(int), hourly_pattern['total_actual'], 'o-', label='Actual')
        axes[1, 1].plot(hourly_pattern.index.astype(int), hourly_pattern['total_predicted'], 's-', label='Predicted')
        axes[1, 1].set_title('Average Hourly Pattern')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Power (kW)')
        axes[1, 1].legend()
        axes[1, 1].set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        self._save_plot(fig, 'analysis/energy_overview')
        plt.close()
        
        # 2. Device Category Breakdown
        self._create_category_breakdown_plot(df, device_categories)
        
        # 3. Stacked Area Plot
        self._create_stacked_area_plot(df, device_categories)
    
    def _create_device_plots(self, df: pd.DataFrame, device_categories: Dict, devices_config: Dict):
        """Create individual plots for each device."""
        device_columns = [col for col in df.columns if col not in ['total_actual', 'total_predicted']]
        
        for device in device_columns:
            if device in df.columns and not df[device].isna().all():
                self._create_single_device_plot(df, device, device_categories, devices_config)
    
    def _create_single_device_plot(self, df: pd.DataFrame, device: str, 
                                 device_categories: Dict, devices_config: Dict):
        """Create comprehensive plot for a single device."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Get device info
        device_info = devices_config.get('devices', {}).get(device, {}) if devices_config else {}
        device_name = device_info.get('name', device.replace('_', ' ').title())
        category = device_categories.get(device, 'unknown')
        color = self.category_colors.get(category, '#333333')
        
        fig.suptitle(f'Device Analysis: {device_name}', fontsize=14, fontweight='bold')
        
        # Time series
        axes[0, 0].plot(df.index, df[device], color=color, linewidth=1.5, alpha=0.8)
        axes[0, 0].set_title('Energy Consumption Over Time')
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Daily pattern
        daily_pattern = df.groupby(df.index.hour)[device].mean()
        axes[0, 1].plot(daily_pattern.index.astype(int), daily_pattern.values, 'o-', color=color, linewidth=2)
        axes[0, 1].set_title('Average Daily Pattern')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Power (kW)')
        axes[0, 1].set_xticks(range(0, 24, 2))
        axes[0, 1].grid(True, alpha=0.3)
        
        # Weekly pattern - Fix broadcasting issue
        weekly_pattern = df.groupby(df.index.dayofweek)[device].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # Reindex to ensure all 7 days are present, fill missing with 0
        weekly_pattern_full = weekly_pattern.reindex(range(7), fill_value=0)
        axes[1, 0].bar(days, weekly_pattern_full.values, color=color, alpha=0.7)
        axes[1, 0].set_title('Average Weekly Pattern')
        axes[1, 0].set_ylabel('Power (kW)')
        
        # Distribution
        axes[1, 1].hist(df[device].dropna(), bins=30, color=color, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Power Distribution')
        axes[1, 1].set_xlabel('Power (kW)')
        axes[1, 1].set_ylabel('Frequency')
        
        # Add device info as text
        if device_info:
            info_text = f"""Category: {category.replace('_', ' ').title()}
Peak Power: {device_info.get('peak_power', 'N/A')} W
Typical Power: {device_info.get('typical_power', 'N/A')} W
Quantity: {device_info.get('quantity', 'N/A')}"""
            fig.text(0.02, 0.98, info_text, transform=fig.transFigure, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        self._save_plot(fig, f'devices/{device}')
        plt.close()
    
    def _create_analysis_plots(self, df: pd.DataFrame, device_categories: Dict, weather_data: pd.DataFrame = None):
        """Create advanced analysis plots."""
        
        # 1. Load Duration Curves
        self._create_load_duration_curve(df, device_categories)
        
        # 2. Correlation Matrix
        self._create_correlation_matrix(df)
        
        # 3. Weather Correlation (if weather data available)
        if weather_data is not None:
            self._create_weather_correlation_plots(df, weather_data)
        
        # 4. Peak Analysis
        self._create_peak_analysis(df, device_categories)
        
        # 5. Energy Balance Validation
        self._create_energy_balance_plot(df)
    
    def _create_category_breakdown_plot(self, df: pd.DataFrame, device_categories: Dict):
        """Create pie chart and bar chart for device categories."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate category totals
        category_totals = {}
        device_columns = [col for col in df.columns if col not in ['total_actual', 'total_predicted']]
        
        for device in device_columns:
            category = device_categories.get(device, 'unknown')
            if category not in category_totals:
                category_totals[category] = 0
            category_totals[category] += df[device].mean()
        
        categories = list(category_totals.keys())
        values = list(category_totals.values())
        colors = [self.category_colors.get(cat, '#333333') for cat in categories]
        
        # Pie chart
        axes[0].pie(values, labels=[cat.replace('_', ' ').title() for cat in categories], 
                   colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Energy Consumption by Category')
        
        # Bar chart
        bars = axes[1].bar([cat.replace('_', ' ').title() for cat in categories], values, color=colors, alpha=0.7)
        axes[1].set_title('Average Power Consumption by Category')
        axes[1].set_ylabel('Power (kW)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_plot(fig, 'analysis/category_breakdown')
        plt.close()
    
    def _create_stacked_area_plot(self, df: pd.DataFrame, device_categories: Dict):
        """Create stacked area plot showing device contributions over time."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        device_columns = [col for col in df.columns if col not in ['total_actual', 'total_predicted']]
        
        # Group devices by category
        categories = {}
        for device in device_columns:
            category = device_categories.get(device, 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(device)
        
        # Create stacked plot by category
        bottom = np.zeros(len(df))
        
        for category, devices in categories.items():
            category_total = df[devices].sum(axis=1)
            color = self.category_colors.get(category, '#333333')
            ax.fill_between(df.index, bottom, bottom + category_total, 
                           label=category.replace('_', ' ').title(), 
                           color=color, alpha=0.7)
            bottom += category_total
        
        # Plot total actual as overlay line (not stacked on top)
        ax.plot(df.index, df['total_actual'], 'k-', linewidth=3, label='Total Actual', alpha=0.9, zorder=10)
        
        ax.set_title('Energy Consumption Breakdown Over Time')
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, zorder=0)
        
        plt.tight_layout()
        self._save_plot(fig, 'analysis/stacked_energy_breakdown')
        plt.close()
    
    def _create_load_duration_curve(self, df: pd.DataFrame, device_categories: Dict):
        """Create load duration curves for total and category breakdowns."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Load Duration Curves', fontsize=16, fontweight='bold')
        
        # Total load duration curve
        total_sorted = np.sort(df['total_actual'])[::-1]
        duration_hours = np.arange(1, len(total_sorted) + 1) / len(total_sorted) * 100
        
        axes[0, 0].plot(duration_hours, total_sorted, linewidth=2)
        axes[0, 0].set_title('Total Load Duration Curve')
        axes[0, 0].set_xlabel('Duration (%)')
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Category load duration curves
        device_columns = [col for col in df.columns if col not in ['total_actual', 'total_predicted']]
        categories = {}
        for device in device_columns:
            category = device_categories.get(device, 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(device)
        
        for i, (category, devices) in enumerate(categories.items()):
            ax = axes[0, 1] if i < 2 else axes[1, i-2]
            category_total = df[devices].sum(axis=1)
            category_sorted = np.sort(category_total)[::-1]
            
            color = self.category_colors.get(category, '#333333')
            ax.plot(duration_hours, category_sorted, linewidth=2, color=color)
            ax.set_title(f'{category.replace("_", " ").title()} Load Duration')
            ax.set_xlabel('Duration (%)')
            ax.set_ylabel('Power (kW)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(fig, 'analysis/load_duration_curves')
        plt.close()
    
    def _create_correlation_matrix(self, df: pd.DataFrame):
        """Create correlation matrix of device energy consumption."""
        device_columns = [col for col in df.columns if col not in ['total_actual', 'total_predicted']]
        
        if len(device_columns) > 1:
            corr_matrix = df[device_columns].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            ax.set_title('Device Energy Consumption Correlation Matrix')
            
            plt.tight_layout()
            self._save_plot(fig, 'analysis/correlation_matrix')
            plt.close()
    
    def _create_weather_correlation_plots(self, df: pd.DataFrame, weather_data: pd.DataFrame):
        """Create weather correlation analysis plots."""
        if 'temperature' not in weather_data.columns:
            return
        
        # Align weather data with energy data
        weather_aligned = weather_data.reindex(df.index, method='nearest')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Weather Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Temperature vs Total Energy
        axes[0, 0].scatter(weather_aligned['temperature'], df['total_actual'], alpha=0.6)
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Total Energy (kW)')
        axes[0, 0].set_title('Temperature vs Energy Consumption')
        
        # Temperature time series with energy overlay
        ax2 = axes[0, 1].twinx()
        axes[0, 1].plot(df.index, weather_aligned['temperature'], 'r-', alpha=0.7, label='Temperature')
        ax2.plot(df.index, df['total_actual'], 'b-', alpha=0.7, label='Energy')
        axes[0, 1].set_ylabel('Temperature (°C)', color='r')
        ax2.set_ylabel('Energy (kW)', color='b')
        axes[0, 1].set_title('Temperature and Energy Over Time')
        
        # Daily temperature vs energy correlation
        daily_temp = weather_aligned.resample('D')['temperature'].mean()
        daily_energy = df.resample('D')['total_actual'].mean()
        axes[1, 0].scatter(daily_temp, daily_energy, alpha=0.7)
        axes[1, 0].set_xlabel('Daily Avg Temperature (°C)')
        axes[1, 0].set_ylabel('Daily Avg Energy (kW)')
        axes[1, 0].set_title('Daily Temperature vs Energy')
        
        # Heating/Cooling degree days analysis
        heating_dd = np.maximum(18 - weather_aligned['temperature'], 0)
        cooling_dd = np.maximum(weather_aligned['temperature'] - 22, 0)
        
        axes[1, 1].scatter(heating_dd + cooling_dd, df['total_actual'], alpha=0.6)
        axes[1, 1].set_xlabel('Degree Days (Heating + Cooling)')
        axes[1, 1].set_ylabel('Energy (kW)')
        axes[1, 1].set_title('Degree Days vs Energy')
        
        plt.tight_layout()
        self._save_plot(fig, 'analysis/weather_correlation')
        plt.close()
    
    def _create_peak_analysis(self, df: pd.DataFrame, device_categories: Dict):
        """Create peak demand analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Peak Demand Analysis', fontsize=16, fontweight='bold')
        
        # Peak hours identification
        peak_threshold = df['total_actual'].quantile(0.95)
        peak_hours = df[df['total_actual'] >= peak_threshold]
        
        # Peak hours by hour of day
        peak_by_hour = peak_hours.groupby(peak_hours.index.hour).size()
        axes[0, 0].bar(peak_by_hour.index.astype(int), peak_by_hour.values, alpha=0.7)
        axes[0, 0].set_title('Peak Hours Distribution by Hour of Day')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Number of Peak Hours')
        
        # Peak hours by day of week - Fix broadcasting issue
        peak_by_dow = peak_hours.groupby(peak_hours.index.dayofweek).size()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # Reindex to ensure all 7 days are present, fill missing with 0
        peak_by_dow_full = peak_by_dow.reindex(range(7), fill_value=0)
        axes[0, 1].bar(days, peak_by_dow_full.values, alpha=0.7)
        axes[0, 1].set_title('Peak Hours Distribution by Day of Week')
        axes[0, 1].set_ylabel('Number of Peak Hours')
        
        # Device contribution during peak hours
        device_columns = [col for col in df.columns if col not in ['total_actual', 'total_predicted']]
        peak_contributions = {}
        
        for device in device_columns:
            category = device_categories.get(device, 'unknown')
            if category not in peak_contributions:
                peak_contributions[category] = 0
            peak_contributions[category] += peak_hours[device].mean()
        
        categories = list(peak_contributions.keys())
        contributions = list(peak_contributions.values())
        colors = [self.category_colors.get(cat, '#333333') for cat in categories]
        
        axes[1, 0].pie(contributions, labels=[cat.replace('_', ' ').title() for cat in categories],
                      colors=colors, autopct='%1.1f%%')
        axes[1, 0].set_title('Device Category Contribution During Peak Hours')
        
        # Peak vs average comparison
        avg_contributions = {}
        for device in device_columns:
            category = device_categories.get(device, 'unknown')
            if category not in avg_contributions:
                avg_contributions[category] = 0
            avg_contributions[category] += df[device].mean()
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, [avg_contributions[cat] for cat in categories], 
                      width, label='Average', alpha=0.7)
        axes[1, 1].bar(x + width/2, contributions, width, label='Peak Hours', alpha=0.7)
        axes[1, 1].set_xlabel('Device Categories')
        axes[1, 1].set_ylabel('Power (kW)')
        axes[1, 1].set_title('Average vs Peak Hour Consumption')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        self._save_plot(fig, 'analysis/peak_analysis')
        plt.close()
    
    def _create_energy_balance_plot(self, df: pd.DataFrame):
        """Create energy balance validation plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Energy Balance Validation', fontsize=16, fontweight='bold')
        
        # Actual vs Predicted scatter
        axes[0, 0].scatter(df['total_actual'], df['total_predicted'], alpha=0.6)
        max_val = max(df['total_actual'].max(), df['total_predicted'].max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Energy (kW)')
        axes[0, 0].set_ylabel('Predicted Energy (kW)')
        axes[0, 0].set_title('Actual vs Predicted Energy')
        
        # Error distribution
        error = df['total_predicted'] - df['total_actual']
        axes[0, 1].hist(error, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(error.mean(), color='red', linestyle='--', label=f'Mean: {error.mean():.3f}')
        axes[0, 1].set_xlabel('Prediction Error (kW)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Prediction Error Distribution')
        axes[0, 1].legend()
        
        # Cumulative error
        cumulative_error = error.cumsum()
        axes[1, 0].plot(df.index, cumulative_error)
        axes[1, 0].set_title('Cumulative Prediction Error')
        axes[1, 0].set_ylabel('Cumulative Error (kW)')
        axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Error statistics
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        mape = np.mean(np.abs(error / df['total_actual'])) * 100
        r2 = 1 - np.sum(error**2) / np.sum((df['total_actual'] - df['total_actual'].mean())**2)
        
        stats_text = f"""Error Statistics:
MAE: {mae:.3f} kW
RMSE: {rmse:.3f} kW
MAPE: {mape:.2f}%
R²: {r2:.4f}"""
        
        axes[1, 1].text(0.1, 0.7, stats_text, transform=axes[1, 1].transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Validation Metrics')
        
        plt.tight_layout()
        self._save_plot(fig, 'analysis/energy_balance_validation')
        plt.close()
    
    def _create_comparison_plots(self, df: pd.DataFrame, device_categories: Dict):
        """Create comparison plots between different devices and time periods."""
        
        # Top energy consumers
        device_columns = [col for col in df.columns if col not in ['total_actual', 'total_predicted']]
        device_totals = {device: df[device].sum() for device in device_columns}
        top_devices = sorted(device_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        devices, totals = zip(*top_devices)
        colors = [self.category_colors.get(device_categories.get(device, 'unknown'), '#333333') 
                 for device in devices]
        
        bars = ax.barh([device.replace('_', ' ').title() for device in devices], totals, color=colors, alpha=0.7)
        ax.set_xlabel('Total Energy Consumption (kWh)')
        ax.set_title('Top 10 Energy Consuming Devices')
        
        plt.tight_layout()
        self._save_plot(fig, 'comparison/top_energy_consumers')
        plt.close()
        
        # Weekday vs Weekend comparison
        weekday_data = df[df.index.dayofweek < 5]
        weekend_data = df[df.index.dayofweek >= 5]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Category comparison
        categories = set(device_categories.values())
        weekday_cat_totals = {}
        weekend_cat_totals = {}
        
        for category in categories:
            category_devices = [device for device, cat in device_categories.items() if cat == category]
            weekday_cat_totals[category] = weekday_data[category_devices].sum().sum()
            weekend_cat_totals[category] = weekend_data[category_devices].sum().sum()
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[0].bar(x - width/2, [weekday_cat_totals[cat] for cat in categories], 
                   width, label='Weekday', alpha=0.7)
        axes[0].bar(x + width/2, [weekend_cat_totals[cat] for cat in categories], 
                   width, label='Weekend', alpha=0.7)
        axes[0].set_xlabel('Device Categories')
        axes[0].set_ylabel('Total Energy (kWh)')
        axes[0].set_title('Weekday vs Weekend Energy Consumption')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45)
        axes[0].legend()
        
        # Hourly pattern comparison
        weekday_hourly = weekday_data.groupby(weekday_data.index.hour)['total_actual'].mean()
        weekend_hourly = weekend_data.groupby(weekend_data.index.hour)['total_actual'].mean()
        
        axes[1].plot(weekday_hourly.index.astype(int), weekday_hourly.values, 'o-', label='Weekday', linewidth=2)
        axes[1].plot(weekend_hourly.index.astype(int), weekend_hourly.values, 's-', label='Weekend', linewidth=2)
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Average Power (kW)')
        axes[1].set_title('Weekday vs Weekend Hourly Pattern')
        axes[1].legend()
        axes[1].set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        self._save_plot(fig, 'comparison/weekday_vs_weekend')
        plt.close()
    
    def _export_data_to_excel(self, df: pd.DataFrame, device_categories: Dict, devices_config: Dict):
        """Export disaggregation results to structured Excel file."""
        if not self.config.excel_format:
            return
        
        excel_path = self.output_dir / "data" / "energy_load_profiles_detailed.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Energy_Profiles', index=True)
            
            # Device summary sheet
            device_columns = [col for col in df.columns if col not in ['total_actual', 'total_predicted']]
            summary_data = []
            
            for device in device_columns:
                device_info = devices_config.get('devices', {}).get(device, {}) if devices_config else {}
                category = device_categories.get(device, 'unknown')
                
                summary_data.append({
                    'Device_ID': device,
                    'Device_Name': device_info.get('name', device.replace('_', ' ').title()),
                    'Category': category.replace('_', ' ').title(),
                    'Average_Power_kW': df[device].mean(),
                    'Max_Power_kW': df[device].max(),
                    'Min_Power_kW': df[device].min(),
                    'Total_Energy_kWh': df[device].sum() * 0.25,  # Assuming 15-min intervals
                    'Peak_Power_W': device_info.get('peak_power', 'N/A'),
                    'Typical_Power_W': device_info.get('typical_power', 'N/A'),
                    'Quantity': device_info.get('quantity', 'N/A'),
                    'Allocation_Percent': (df[device].mean() / df['total_actual'].mean()) * 100
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Device_Summary', index=False)
            
            # Category summary sheet
            categories = set(device_categories.values())
            category_summary = []
            
            for category in categories:
                category_devices = [device for device, cat in device_categories.items() if cat == category]
                category_total = df[category_devices].sum(axis=1)
                
                category_summary.append({
                    'Category': category.replace('_', ' ').title(),
                    'Device_Count': len(category_devices),
                    'Average_Power_kW': category_total.mean(),
                    'Max_Power_kW': category_total.max(),
                    'Total_Energy_kWh': category_total.sum() * 0.25,
                    'Allocation_Percent': (category_total.mean() / df['total_actual'].mean()) * 100
                })
            
            category_df = pd.DataFrame(category_summary)
            category_df.to_excel(writer, sheet_name='Category_Summary', index=False)
            
            # Hourly patterns
            hourly_patterns = df.groupby(df.index.hour).mean()
            hourly_patterns.to_excel(writer, sheet_name='Hourly_Patterns', index=True)
            
            # Daily patterns
            daily_patterns = df.resample('D').mean()
            daily_patterns.to_excel(writer, sheet_name='Daily_Patterns', index=True)
        
        self.logger.info(f"📊 Excel export completed: {excel_path}")
    
    def _create_summary_report(self, df: pd.DataFrame, device_categories: Dict, 
                             result, location: str, start_date: str, end_date: str):
        """Create comprehensive summary report."""
        report_path = self.output_dir / "reports" / "analysis_summary.md"
        
        # Calculate key metrics
        total_energy = df['total_actual'].sum() * 0.25  # kWh
        avg_power = df['total_actual'].mean()
        peak_power = df['total_actual'].max()
        load_factor = avg_power / peak_power
        
        device_columns = [col for col in df.columns if col not in ['total_actual', 'total_predicted']]
        device_count = len(device_columns)
        
        # Category breakdown
        categories = set(device_categories.values())
        category_breakdown = {}
        for category in categories:
            category_devices = [device for device, cat in device_categories.items() if cat == category]
            category_total = df[category_devices].sum(axis=1).mean()
            category_breakdown[category] = (category_total / avg_power) * 100
        
        # Error metrics
        error = df['total_predicted'] - df['total_actual']
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        mape = np.mean(np.abs(error / df['total_actual'])) * 100
        r2 = 1 - np.sum(error**2) / np.sum((df['total_actual'] - df['total_actual'].mean())**2)
        
        # Generate report
        report_content = f"""# Energy Load Profile Analysis Report

## Analysis Overview
- **Location**: {location}
- **Analysis Period**: {start_date} to {end_date}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Energy Metrics
- **Total Energy Consumption**: {total_energy:.2f} kWh
- **Average Power**: {avg_power:.2f} kW
- **Peak Power**: {peak_power:.2f} kW
- **Load Factor**: {load_factor:.3f}
- **Number of Devices**: {device_count}

## Model Performance
- **Mean Absolute Error (MAE)**: {mae:.3f} kW
- **Root Mean Square Error (RMSE)**: {rmse:.3f} kW
- **Mean Absolute Percentage Error (MAPE)**: {mape:.2f}%
- **R² Score**: {r2:.4f}

## Device Category Breakdown
"""
        
        for category, percentage in sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True):
            report_content += f"- **{category.replace('_', ' ').title()}**: {percentage:.1f}%\n"
        
        report_content += f"""
## Data Quality
- **Total Data Points**: {len(df):,}
- **Missing Data Points**: {df.isnull().sum().sum()}
- **Data Completeness**: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%

## Generated Outputs
- Device-specific energy profiles and plots
- Comprehensive visualization suite
- Excel data export with multiple analysis sheets
- Category-based energy breakdown
- Peak demand analysis
- Weather correlation analysis (if applicable)

## File Structure
```
output/
├── plots/
│   ├── devices/          # Individual device plots
│   ├── analysis/         # Analysis and overview plots
│   └── comparison/       # Comparative analysis plots
├── data/
│   └── energy_load_profiles_detailed.xlsx
└── reports/
    └── analysis_summary.md
```
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"📋 Summary report created: {report_path}")
    
    def _save_plot(self, fig, filename_without_ext: str):
        """Save plot in configured formats."""
        for fmt in self.config.plot_formats:
            filepath = self.output_dir / "plots" / f"{filename_without_ext}.{fmt}"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')

def create_visualization_config(output_dir: str = "output", 
                              timestamp_dirs: bool = True,
                              formats: List[str] = None) -> VisualizationConfig:
    """Helper function to create visualization configuration."""
    return VisualizationConfig(
        output_base_dir=output_dir,
        create_timestamp_dirs=timestamp_dirs,
        plot_formats=formats or ['png', 'pdf']
    )