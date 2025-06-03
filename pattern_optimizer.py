"""
Pattern Optimization System using Reinforcement Learning
=======================================================

This system optimizes device load patterns by comparing generated profiles
with real measured load data using reinforcement learning techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import copy
import json
import yaml
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Reduce excessive debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# For live monitoring
import threading
import time
import webbrowser
from flask import Flask, render_template, jsonify
import plotly.graph_objs as go
import plotly.utils

# Import existing modules
from config_manager import ConfigManager
from device_calculator import DeviceLoadCalculator
from weather_database import WeatherDatabase
from weather_fetcher import MultiSourceWeatherFetcher

class PatternOptimizer:
    """
    Reinforcement Learning-based pattern optimizer with live monitoring.
    """

    def __init__(self, config_path: str = "config.yaml", optimization_config_path: str = "optimization_config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.optimization_config = self._load_optimization_config(optimization_config_path)
        self.logger = logging.getLogger(__name__)

        # Initialize device calculator with current patterns
        self.device_calculator = DeviceLoadCalculator(self.config.get('devices', {}))

        # Weather fetcher for generating synthetic data
        db_path = self.config.get('database', {}).get('path', 'energy_weather.db')
        self.weather_fetcher = MultiSourceWeatherFetcher(self.config, db_path)

        # RL Parameters
        self.learning_rate = self.optimization_config.get('learning_rate', 0.01)
        self.exploration_rate = self.optimization_config.get('exploration_rate', 0.1)
        self.exploration_decay = self.optimization_config.get('exploration_decay', 0.995)
        self.min_exploration_rate = self.optimization_config.get('min_exploration_rate', 0.01)

        # Pattern constraints
        self.pattern_bounds = (0.0, 1.0)
        self.max_change_per_step = self.optimization_config.get('max_change_per_step', 0.05)

        # Optimization tracking
        self.best_patterns = {}
        self.best_score = float('inf')
        self.training_history = []
        self.current_patterns = {}
        self.original_patterns = {}

        # Load target data
        self.target_data = None
        self.location = None
        self.weather_data_cache = None
        self.evaluation_periods = []  # Multiple evaluation periods

        # Live monitoring
        self.live_monitoring = False
        self.flask_app = None
        self.monitoring_thread = None
        self.current_generation = 0
        self.generation_scores = []
        self.best_scores_history = []
        self.population_stats = {}

    def _load_optimization_config(self, config_path: str) -> Dict:
        """Load optimization configuration."""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        else:
            # Create default optimization config
            default_config = self._get_default_optimization_config()
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(default_config, file, default_flow_style=False, indent=2)
            self.logger.info(f"Created default optimization config: {config_path}")
            return default_config

    def _get_default_optimization_config(self) -> Dict:
        """Get default optimization configuration."""
        return {
            'learning_rate': 0.01,
            'exploration_rate': 0.1,
            'exploration_decay': 0.995,
            'min_exploration_rate': 0.01,
            'max_episodes': 1000,
            'convergence_threshold': 0.001,
            'max_change_per_step': 0.05,

            'training_data': {
                'input_file': 'load_profiles.xlsx',
                'timestamp_column': 'Timestamp',
                'value_column': 'Value',
                'value_unit': 'kW',
                'timezone': 'UTC',
                'years_to_use': [2018, 2019, 2020, 2023, 2024]
            },

            'evaluation': {
                'use_full_dataset': False,  # Use full dataset or sample periods
                'sample_periods': [
                    {'start_month': 1, 'days': 14},   # Winter sample
                    {'start_month': 4, 'days': 14},   # Spring sample
                    {'start_month': 7, 'days': 14},   # Summer sample
                    {'start_month': 10, 'days': 14}   # Autumn sample
                ],
                'progressive_evaluation': True,  # Start small, then increase
                'generations_per_stage': [25, 25, 25, 25]  # Generations for each stage
            },

            'device_constraints': {
                'air_conditioner': {
                    'seasonal_percentages': {
                        'winter': 0.05,
                        'spring': 0.15,
                        'summer': 0.50,
                        'autumn': 0.20
                    },
                    'optimization_weight': 1.0,
                    'allow_optimization': True
                },
                'heater': {
                    'seasonal_percentages': {
                        'winter': 0.40,
                        'spring': 0.20,
                        'summer': 0.05,
                        'autumn': 0.25
                    },
                    'optimization_weight': 1.0,
                    'allow_optimization': True
                },
                'general_load': {
                    'year_round_percentage': 0.25,
                    'optimization_weight': 0.8,
                    'allow_optimization': True
                },
                'refrigeration': {
                    'year_round_percentage': 0.10,
                    'optimization_weight': 0.5,
                    'allow_optimization': False
                },
                'lighting': {
                    'year_round_percentage': 0.08,
                    'optimization_weight': 0.6,
                    'allow_optimization': True
                },
                'water_heater': {
                    'year_round_percentage': 0.12,
                    'optimization_weight': 0.7,
                    'allow_optimization': True
                }
            },

            'reward_weights': {
                'mse_weight': 0.4,
                'trend_weight': 0.3,
                'peak_weight': 0.2,
                'constraint_weight': 0.1
            },

            'optimization': {
                'algorithm': 'genetic',
                'population_size': 30,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            }
        }

    def load_training_data(self, file_path: str, location: str) -> pd.DataFrame:
        """Load training data from Excel file with multiple year sheets."""
        training_config = self.optimization_config.get('training_data', {})
        timestamp_col = training_config.get('timestamp_column', 'Timestamp')
        value_col = training_config.get('value_column', 'Value')
        value_unit = training_config.get('value_unit', 'kW')
        timezone = training_config.get('timezone', 'UTC')
        years_to_use = training_config.get('years_to_use', [2018, 2019, 2020, 2023, 2024])

        self.logger.info(f"Loading training data from {file_path}")

        all_data = []

        try:
            # Read Excel file and get all sheet names
            xl_file = pd.ExcelFile(file_path)
            available_sheets = xl_file.sheet_names

            self.logger.info(f"Available sheets: {available_sheets}")

            for year in years_to_use:
                sheet_name = str(year)
                if sheet_name in available_sheets:
                    self.logger.info(f"Loading data for year {year}")

                    # Read the sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name)

                    # Validate columns
                    if timestamp_col not in df.columns or value_col not in df.columns:
                        self.logger.error(f"Required columns {timestamp_col}, {value_col} not found in sheet {sheet_name}")
                        self.logger.info(f"Available columns: {list(df.columns)}")
                        continue

                    # Process the data
                    df = df[[timestamp_col, value_col]].copy()
                    df.columns = ['datetime', 'total_power']

                    # Convert timestamp
                    df['datetime'] = pd.to_datetime(df['datetime'])

                    # Handle timezone
                    if timezone.upper() == 'UTC':
                        if df['datetime'].dt.tz is None:
                            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
                        else:
                            df['datetime'] = df['datetime'].dt.tz_convert('UTC')
                    else:
                        if df['datetime'].dt.tz is None:
                            df['datetime'] = df['datetime'].dt.tz_localize('Europe/Berlin')
                        else:
                            df['datetime'] = df['datetime'].dt.tz_convert('Europe/Berlin')

                    # Convert units if necessary (kW to W)
                    if value_unit.upper() == 'KW':
                        df['total_power'] = df['total_power'] * 1000

                    # Remove any invalid data
                    df = df.dropna()
                    df = df[df['total_power'] >= 0]

                    all_data.append(df)
                    self.logger.info(f"Loaded {len(df)} records for year {year}")
                else:
                    self.logger.warning(f"Sheet '{sheet_name}' not found in Excel file")

            if not all_data:
                raise ValueError("No valid data found in any sheets")

            # Combine all years
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('datetime').reset_index(drop=True)

            # Set datetime as index
            combined_data.set_index('datetime', inplace=True)

            # Ensure 15-minute intervals by resampling if necessary
            combined_data = combined_data.resample('15min').mean().interpolate()

            self.logger.info(f"Successfully loaded {len(combined_data)} total records")
            self.logger.info(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
            self.logger.info(f"Power range: {combined_data['total_power'].min():.1f}W to {combined_data['total_power'].max():.1f}W")
            self.logger.info(f"Average power: {combined_data['total_power'].mean():.1f}W")

            self.target_data = combined_data
            self.location = location

            # Store original patterns for comparison
            self.original_patterns = {}
            for device_name, device_config in self.config.get('devices', {}).items():
                self.original_patterns[device_name] = device_config.get('daily_pattern', [0.5] * 96)

            # Adjust device peak powers based on real data
            self._adjust_device_peak_powers()

            # Prepare evaluation periods
            self._prepare_evaluation_periods()

            return combined_data

        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            raise

    def _adjust_device_peak_powers(self):
        """Adjust device peak powers based on real load data statistics."""
        if self.target_data is None:
            return

        # Calculate statistics from real data
        max_power = self.target_data['total_power'].max()
        avg_power = self.target_data['total_power'].mean()
        p95_power = self.target_data['total_power'].quantile(0.95)

        self.logger.info(f"Real data statistics:")
        self.logger.info(f"  Max power: {max_power:.0f}W")
        self.logger.info(f"  Average power: {avg_power:.0f}W")
        self.logger.info(f"  95th percentile: {p95_power:.0f}W")

        # Get current device configuration
        devices = self.config.get('devices', {})
        device_quantities = self.config.get('load_profile', {}).get('device_quantities', {})

        # Calculate total configured peak power
        current_total_peak = 0
        for device_name, device_config in devices.items():
            quantity = device_quantities.get(device_name, 1)
            current_total_peak += device_config.get('peak_power', 0) * quantity

        self.logger.info(f"Current total peak power: {current_total_peak:.0f}W")

        # Calculate scaling factor to match real data
        target_total_peak = p95_power * 1.2
        scaling_factor = target_total_peak / current_total_peak if current_total_peak > 0 else 1.0

        self.logger.info(f"Applying scaling factor: {scaling_factor:.3f}")

        # Apply scaling to all devices
        for device_name, device_config in devices.items():
            old_peak = device_config.get('peak_power', 0)
            new_peak = int(old_peak * scaling_factor)
            device_config['peak_power'] = new_peak
            self.logger.info(f"  {device_name}: {old_peak}W -> {new_peak}W")

        # Update device calculator with new peak powers
        self.device_calculator = DeviceLoadCalculator(devices)

    def _prepare_evaluation_periods(self):
        """Prepare multiple evaluation periods for comprehensive testing."""
        if self.target_data is None:
            return

        eval_config = self.optimization_config.get('evaluation', {})

        if eval_config.get('use_full_dataset', False):
            # Use the entire dataset
            self.evaluation_periods = [(
                self.target_data.index.min().strftime('%Y-%m-%d'),
                self.target_data.index.max().strftime('%Y-%m-%d')
            )]
            self.logger.info("Using full dataset for evaluation")
        else:
            # Use sample periods from different seasons
            sample_periods = eval_config.get('sample_periods', [
                {'start_month': 1, 'days': 14},
                {'start_month': 4, 'days': 14},
                {'start_month': 7, 'days': 14},
                {'start_month': 10, 'days': 14}
            ])

            self.evaluation_periods = []

            for period in sample_periods:
                start_month = period['start_month']
                days = period['days']

                # Find a period in each available year
                for year in [2018, 2019, 2020, 2023, 2024]:
                    try:
                        start_date = datetime(year, start_month, 1, tzinfo=self.target_data.index.tz)
                        end_date = start_date + timedelta(days=days)

                        # Check if this period exists in our data
                        period_data = self.target_data.loc[start_date:end_date]
                        if len(period_data) > days * 24 * 2:  # At least some data
                            self.evaluation_periods.append((
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d')
                            ))
                            break
                    except:
                        continue

            self.logger.info(f"Prepared {len(self.evaluation_periods)} evaluation periods:")
            for i, (start, end) in enumerate(self.evaluation_periods):
                self.logger.info(f"  Period {i+1}: {start} to {end}")

        # Pre-load weather data for all periods
        self._prepare_weather_data()

    def _prepare_weather_data(self):
        """Pre-load weather data for all evaluation periods."""
        self.weather_data_cache = {}

        for i, (start_date, end_date) in enumerate(self.evaluation_periods):
            try:
                weather_data = self.weather_fetcher.get_weather_data(
                    self.location, start_date, end_date
                )

                if weather_data.empty:
                    # Generate synthetic weather data
                    target_tz = self.target_data.index.tz
                    date_range = pd.date_range(start=start_date, end=end_date, freq='15min', tz=target_tz)
                    weather_data = pd.DataFrame({
                        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / (24 * 4)),
                        'humidity': 50 + 20 * np.random.normal(0, 0.1, len(date_range)),
                        'condition': 'Synthetic'
                    }, index=date_range)
                else:
                    # Ensure consistent timezone
                    target_tz = self.target_data.index.tz
                    if weather_data.index.tz != target_tz:
                        if weather_data.index.tz is None:
                            weather_data.index = weather_data.index.tz_localize(target_tz)
                        else:
                            weather_data.index = weather_data.index.tz_convert(target_tz)

                self.weather_data_cache[i] = weather_data
                self.logger.info(f"Cached weather data for period {i+1}: {len(weather_data)} records")

            except Exception as e:
                self.logger.error(f"Failed to prepare weather data for period {i+1}: {e}")
                self.weather_data_cache[i] = None

    def evaluate_patterns(self, patterns: Dict[str, List[float]],
                          evaluation_stage: int = 0) -> float:
        """Evaluate patterns using progressive evaluation strategy."""
        if self.target_data is None:
            return float('inf')

        try:
            eval_config = self.optimization_config.get('evaluation', {})
            progressive = eval_config.get('progressive_evaluation', True)

            if progressive:
                # Use only first few periods initially, then expand
                stages_per_generation = eval_config.get('generations_per_stage', [25, 25, 25, 25])
                current_stage = min(evaluation_stage // 25, len(stages_per_generation) - 1)
                periods_to_use = min(current_stage + 1, len(self.evaluation_periods))
            else:
                periods_to_use = len(self.evaluation_periods)

            total_score = 0
            valid_periods = 0

            # Evaluate across multiple periods
            for period_idx in range(periods_to_use):
                if period_idx >= len(self.evaluation_periods):
                    continue

                start_date, end_date = self.evaluation_periods[period_idx]
                weather_data = self.weather_data_cache.get(period_idx)

                if weather_data is None or weather_data.empty:
                    continue

                period_score = self._evaluate_single_period(patterns, weather_data, start_date, end_date)

                if period_score != float('inf'):
                    total_score += period_score
                    valid_periods += 1

            if valid_periods == 0:
                return float('inf')

            # Average score across periods
            final_score = total_score / valid_periods

            # Log evaluation details for first few evaluations
            if hasattr(self, '_eval_count'):
                self._eval_count += 1
            else:
                self._eval_count = 1

            if self._eval_count <= 5:
                self.logger.debug(f"Evaluation {self._eval_count}: Used {periods_to_use}/{len(self.evaluation_periods)} periods, Score: {final_score:.6f}")

            return final_score

        except Exception as e:
            self.logger.error(f"Error in evaluate_patterns: {e}")
            return float('inf')

    def _evaluate_single_period(self, patterns: Dict[str, List[float]],
                                weather_data: pd.DataFrame,
                                start_date: str, end_date: str) -> float:
        """Evaluate patterns for a single time period."""
        try:
            # Update device calculator with new patterns
            devices = copy.deepcopy(self.config.get('devices', {}))
            for device_name, pattern in patterns.items():
                if device_name in devices and len(pattern) == 96:
                    devices[device_name]['daily_pattern'] = pattern

            temp_calculator = DeviceLoadCalculator(devices)

            # Generate synthetic load profile
            enabled_devices = [name for name in devices.keys() if devices[name].get('enabled', True)]
            device_quantities = self.config.get('load_profile', {}).get('device_quantities', {})

            synthetic_data = temp_calculator.calculate_total_load(
                weather_data, enabled_devices, device_quantities
            )

            if synthetic_data.empty:
                return float('inf')

            # Get corresponding target data
            target_subset = self.target_data.loc[start_date:end_date]

            if target_subset.empty:
                return float('inf')

            # Ensure same timezone
            if synthetic_data.index.tz != target_subset.index.tz:
                if synthetic_data.index.tz is None:
                    synthetic_data.index = synthetic_data.index.tz_localize(target_subset.index.tz)
                else:
                    synthetic_data.index = synthetic_data.index.tz_convert(target_subset.index.tz)

            # Find overlapping indices
            common_index = synthetic_data.index.intersection(target_subset.index)

            if len(common_index) == 0:
                # Try rounding approach
                synthetic_rounded = synthetic_data.copy()
                target_rounded = target_subset.copy()
                synthetic_rounded.index = synthetic_rounded.index.round('15min')
                target_rounded.index = target_rounded.index.round('15min')
                common_index = synthetic_rounded.index.intersection(target_rounded.index)

                if len(common_index) > 0:
                    synthetic_aligned = synthetic_rounded.loc[common_index]['total_power']
                    target_aligned = target_rounded.loc[common_index]['total_power']
                else:
                    return float('inf')
            else:
                synthetic_aligned = synthetic_data.loc[common_index]['total_power']
                target_aligned = target_subset.loc[common_index]['total_power']

            # Remove NaN values
            valid_mask = ~(synthetic_aligned.isna() | target_aligned.isna())
            synthetic_aligned = synthetic_aligned[valid_mask]
            target_aligned = target_aligned[valid_mask]

            if len(synthetic_aligned) == 0:
                return float('inf')

            # Calculate metrics
            mse = mean_squared_error(target_aligned, synthetic_aligned)
            target_mean = target_aligned.mean()
            mse_normalized = mse / (target_mean ** 2) if target_mean > 0 else mse

            correlation, _ = pearsonr(target_aligned, synthetic_aligned)
            correlation = 0 if np.isnan(correlation) else correlation
            trend_score = 1 - abs(correlation)

            mae = mean_absolute_error(target_aligned, synthetic_aligned)
            mae_normalized = mae / target_mean if target_mean > 0 else mae

            synthetic_mean = synthetic_aligned.mean()
            scale_diff = abs(synthetic_mean - target_mean) / target_mean if target_mean > 0 else 1.0

            # Combine scores
            total_score = (
                    0.3 * mse_normalized +
                    0.3 * trend_score +
                    0.2 * mae_normalized +
                    0.2 * scale_diff
            )

            # Penalty for unrealistic values
            if synthetic_aligned.max() > target_aligned.max() * 3:
                total_score += 10.0

            return total_score

        except Exception as e:
            self.logger.warning(f"Error evaluating single period {start_date}-{end_date}: {e}")
            return float('inf')

    def start_live_monitoring(self, port: int = 5000):
        """Start the enhanced live monitoring web interface."""
        self.live_monitoring = True

        # Create Flask app
        self.flask_app = Flask(__name__)

        @self.flask_app.route('/')
        def dashboard():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pattern Optimization Monitor</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #1e1e1e; color: white; }
                    .container { max-width: 1400px; margin: 0 auto; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
                    .stat-card { background: #2d2d2d; padding: 20px; border-radius: 8px; text-align: center; }
                    .stat-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
                    .stat-label { color: #ccc; margin-top: 5px; }
                    .chart-container { background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                    .progress-bar { width: 100%; height: 20px; background: #444; border-radius: 10px; overflow: hidden; }
                    .progress-fill { height: 100%; background: linear-gradient(90deg, #4CAF50, #81C784); transition: width 0.3s; }
                    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🔬 Enhanced Pattern Optimization Monitor</h1>
                        <p>Real-time monitoring of genetic algorithm optimization with pattern evolution</p>
                    </div>
                    
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-value" id="generation">0</div>
                            <div class="stat-label">Generation</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="best-score">∞</div>
                            <div class="stat-label">Best Score</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="avg-score">-</div>
                            <div class="stat-label">Avg Score</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="progress">0%</div>
                            <div class="stat-label">Progress</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="eval-records">-</div>
                            <div class="stat-label">Records Used</div>
                        </div>
                    </div>
                    
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
                    </div>
                    
                    <div class="chart-grid">
                        <div class="chart-container">
                            <div id="score-chart"></div>
                        </div>
                        <div class="chart-container">
                            <div id="population-chart"></div>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <div id="pattern-evolution-chart"></div>
                    </div>
                </div>
                
                <script>
                    // Initialize charts
                    var scoreLayout = {
                        title: { text: 'Optimization Progress', font: { color: 'white' } },
                        xaxis: { title: 'Generation', color: 'white' },
                        yaxis: { title: 'Score (lower is better)', color: 'white' },
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: { color: 'white' },
                        margin: { t: 50, b: 50, l: 50, r: 50 }
                    };
                    
                    var populationLayout = {
                        title: { text: 'Population Score Distribution', font: { color: 'white' } },
                        xaxis: { title: 'Score', color: 'white' },
                        yaxis: { title: 'Count', color: 'white' },
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: { color: 'white' },
                        margin: { t: 50, b: 50, l: 50, r: 50 }
                    };
                    
                    var patternLayout = {
                        title: { text: 'Pattern Evolution (Original vs Current Best)', font: { color: 'white' } },
                        xaxis: { title: 'Time of Day (15-min intervals)', color: 'white' },
                        yaxis: { title: 'Usage Factor (0-1)', color: 'white' },
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: { color: 'white' },
                        margin: { t: 50, b: 50, l: 50, r: 50 }
                    };
                    
                    Plotly.newPlot('score-chart', [], scoreLayout);
                    Plotly.newPlot('population-chart', [], populationLayout);
                    Plotly.newPlot('pattern-evolution-chart', [], patternLayout);
                    
                    // Update function
                    function updateDashboard() {
                        $.get('/api/status', function(data) {
                            $('#generation').text(data.generation);
                            $('#best-score').text(data.best_score < 999999 ? data.best_score.toFixed(4) : '∞');
                            $('#avg-score').text(data.avg_score > 0 ? data.avg_score.toFixed(4) : '-');
                            $('#progress').text(data.progress + '%');
                            $('#progress-fill').css('width', data.progress + '%');
                            $('#eval-records').text(data.eval_records || '-');
                            
                            // Update score chart
                            if (data.score_history && data.score_history.length > 0) {
                                var scoreTrace = {
                                    x: data.score_history.map((_, i) => i + 1),
                                    y: data.score_history,
                                    type: 'scatter',
                                    mode: 'lines+markers',
                                    name: 'Best Score',
                                    line: { color: '#4CAF50', width: 2 },
                                    marker: { size: 4 }
                                };
                                Plotly.react('score-chart', [scoreTrace], scoreLayout);
                            }
                            
                            // Update population chart
                            if (data.population_scores && data.population_scores.length > 0) {
                                var populationTrace = {
                                    x: data.population_scores,
                                    type: 'histogram',
                                    name: 'Population',
                                    marker: { color: '#81C784' },
                                    nbinsx: 20
                                };
                                Plotly.react('population-chart', [populationTrace], populationLayout);
                            }
                            
                            // Update pattern evolution chart
                            if (data.pattern_evolution && data.pattern_evolution.length > 0) {
                                var traces = [];
                                var timeLabels = [];
                                for (var h = 0; h < 24; h++) {
                                    for (var m = 0; m < 60; m += 15) {
                                        timeLabels.push(h + ':' + (m < 10 ? '0' + m : m));
                                    }
                                }
                                
                                data.pattern_evolution.forEach(function(device, idx) {
                                    if (device.original) {
                                        traces.push({
                                            x: timeLabels,
                                            y: device.original,
                                            type: 'scatter',
                                            mode: 'lines',
                                            name: device.name + ' (Original)',
                                            line: { color: '#888', width: 1, dash: 'dot' }
                                        });
                                    }
                                    if (device.current) {
                                        traces.push({
                                            x: timeLabels,
                                            y: device.current,
                                            type: 'scatter',
                                            mode: 'lines',
                                            name: device.name + ' (Optimized)',
                                            line: { width: 2 }
                                        });
                                    }
                                });
                                
                                if (traces.length > 0) {
                                    Plotly.react('pattern-evolution-chart', traces, patternLayout);
                                }
                            }
                        }).fail(function() {
                            console.log('Failed to fetch status');
                        });
                    }
                    
                    // Update every 2 seconds
                    setInterval(updateDashboard, 2000);
                    updateDashboard(); // Initial load
                </script>
            </body>
            </html>
            '''

        @self.flask_app.route('/api/status')
        def api_status():
            max_generations = self.optimization_config.get('optimization', {}).get('generations', 100)
            progress = (self.current_generation / max_generations * 100) if max_generations > 0 else 0

            # Calculate total records being used for evaluation
            eval_records = 0
            for period_idx, (start_date, end_date) in enumerate(self.evaluation_periods):
                if hasattr(self, 'current_generation') and period_idx < (self.current_generation // 25 + 1):
                    try:
                        period_data = self.target_data.loc[start_date:end_date]
                        eval_records += len(period_data)
                    except:
                        pass

            # Prepare pattern evolution data
            pattern_evolution = []
            if hasattr(self, 'best_patterns') and self.best_patterns:
                for device_name in list(self.best_patterns.keys())[:3]:  # Show first 3 devices
                    original = self.original_patterns.get(device_name, [])
                    current = self.best_patterns.get(device_name, [])

                    if original and current:
                        pattern_evolution.append({
                            'name': device_name.replace('_', ' ').title(),
                            'original': original,
                            'current': current
                        })

            return jsonify({
                'generation': self.current_generation,
                'best_score': float(self.best_score) if self.best_score != float('inf') else 999999,
                'avg_score': float(np.mean(self.generation_scores)) if self.generation_scores else 0,
                'progress': min(100, progress),
                'score_history': self.best_scores_history,
                'population_scores': self.generation_scores,
                'eval_records': f"{eval_records:,}" if eval_records > 0 else "Loading...",
                'pattern_evolution': pattern_evolution
            })

        # Start Flask in a separate thread
        def run_flask():
            try:
                self.flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
            except Exception as e:
                self.logger.error(f"Flask app failed to start: {e}")

        self.monitoring_thread = threading.Thread(target=run_flask)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        # Open browser
        try:
            webbrowser.open(f'http://localhost:{port}')
        except:
            pass

        self.logger.info(f"Enhanced live monitoring started at http://localhost:{port}")

    def optimize_patterns_genetic(self, enable_live_monitoring: bool = True) -> Dict[str, List[float]]:
        """Optimize patterns using genetic algorithm with progressive evaluation."""
        self.logger.info("Starting genetic algorithm optimization with progressive evaluation")

        if enable_live_monitoring:
            self.start_live_monitoring()

        optimization_config = self.optimization_config.get('optimization', {})
        population_size = optimization_config.get('population_size', 30)
        generations = optimization_config.get('generations', 100)
        mutation_rate = optimization_config.get('mutation_rate', 0.1)
        crossover_rate = optimization_config.get('crossover_rate', 0.8)

        # Get devices to optimize
        devices_to_optimize = []
        fixed_patterns = {}

        for device_name, device_config in self.config.get('devices', {}).items():
            device_constraints = self.optimization_config.get('device_constraints', {}).get(device_name, {})
            allow_optimization = device_constraints.get('allow_optimization', True)
            is_fixed = device_config.get('fixed_pattern', False)

            if allow_optimization and not is_fixed:
                devices_to_optimize.append(device_name)
            else:
                fixed_patterns[device_name] = device_config.get('daily_pattern', [0.5] * 96)
                self.logger.info(f"Device '{device_name}' has fixed pattern (not optimizing)")

        self.logger.info(f"Optimizing patterns for devices: {devices_to_optimize}")

        if not devices_to_optimize:
            self.logger.warning("No devices to optimize!")
            return {}

        # Log evaluation strategy
        total_records = len(self.target_data)
        eval_config = self.optimization_config.get('evaluation', {})
        if eval_config.get('use_full_dataset', False):
            self.logger.info(f"Using full dataset: {total_records:,} records")
        else:
            progressive = eval_config.get('progressive_evaluation', True)
            if progressive:
                self.logger.info(f"Using progressive evaluation across {len(self.evaluation_periods)} periods")
                self.logger.info("Stage 1 (Gen 1-25): 1 period, Stage 2 (Gen 26-50): 2 periods, etc.")
            else:
                total_eval_records = sum(len(self.target_data.loc[start:end]) for start, end in self.evaluation_periods)
                self.logger.info(f"Using {len(self.evaluation_periods)} evaluation periods: {total_eval_records:,} records total")

        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for device_name in devices_to_optimize:
                current_pattern = self.config['devices'][device_name].get('daily_pattern', [0.5] * 96)
                individual[device_name] = [
                    max(0.0, min(1.0, x + np.random.normal(0, 0.05)))
                    for x in current_pattern
                ]
            population.append(individual)

        best_individual = None
        best_score = float('inf')
        self.best_scores_history = []

        for generation in range(generations):
            self.current_generation = generation + 1

            # Evaluate population
            scores = []
            for i, individual in enumerate(population):
                all_patterns = {**fixed_patterns, **individual}
                score = self.evaluate_patterns(all_patterns, generation)
                scores.append(score)

                if score < best_score:
                    best_score = score
                    best_individual = individual.copy()
                    self.best_patterns = {**fixed_patterns, **best_individual}
                    self.logger.info(f"New best score: {best_score:.6f}")

            # Update monitoring data
            self.generation_scores = [s for s in scores if s != float('inf')]
            self.best_scores_history.append(best_score if best_score != float('inf') else 999999)
            self.best_score = best_score

            # Log progress with stage info
            valid_scores = [s for s in scores if s != float('inf')]
            if valid_scores:
                avg_score = np.mean(valid_scores)
                min_score = min(valid_scores)
                max_score = max(valid_scores)

                # Determine current stage
                stage = (generation // 25) + 1
                periods_used = min(stage, len(self.evaluation_periods))

                self.logger.info(f"Generation {generation + 1}/{generations} (Stage {stage}, {periods_used} periods): "
                                 f"Best={min_score:.6f}, Avg={avg_score:.6f}, Worst={max_score:.6f}")
            else:
                self.logger.warning(f"Generation {generation + 1}/{generations}: All scores are infinite!")

            # Selection, crossover, and mutation
            new_population = []

            # Keep best individuals (elitism)
            sorted_indices = np.argsort(scores)
            elite_count = max(1, population_size // 10)
            for i in range(elite_count):
                if scores[sorted_indices[i]] != float('inf'):
                    new_population.append(population[sorted_indices[i]].copy())

            # Generate new individuals
            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, scores)
                parent2 = self._tournament_selection(population, scores)

                if np.random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                child1 = self._mutate(child1, mutation_rate)
                child2 = self._mutate(child2, mutation_rate)

                new_population.extend([child1, child2])

            population = new_population[:population_size]

            # Early stopping if converged
            if len(self.best_scores_history) > 15:
                recent_improvement = self.best_scores_history[-15] - self.best_scores_history[-1]
                if recent_improvement < 0.001:
                    self.logger.info("Converged - stopping early")
                    break

        if best_individual is not None:
            self.best_patterns = {**fixed_patterns, **best_individual}
        else:
            self.logger.error("Optimization failed to find any valid solution")
            self.best_patterns = fixed_patterns

        # Final evaluation with all periods
        if len(self.evaluation_periods) > 1:
            self.logger.info("Performing final evaluation with all periods...")
            final_score = self.evaluate_patterns(self.best_patterns, 999)  # Use all periods
            self.logger.info(f"Final comprehensive score: {final_score:.6f}")

        self.logger.info(f"Optimization complete. Best score: {best_score:.6f}")

        if enable_live_monitoring:
            self.logger.info("Live monitoring will continue running. Close the browser tab when done.")

        return self.best_patterns

    def _tournament_selection(self, population: List[Dict], scores: List[float],
                              tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm."""
        valid_indices = [i for i, score in enumerate(scores) if score != float('inf')]

        if not valid_indices:
            return population[np.random.randint(len(population))].copy()

        tournament_size = min(tournament_size, len(valid_indices))
        tournament_indices = np.random.choice(valid_indices, tournament_size, replace=False)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_scores)]
        return population[winner_idx].copy()

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Crossover operation for genetic algorithm."""
        child1, child2 = {}, {}

        for device_name in parent1.keys():
            pattern1 = parent1[device_name]
            pattern2 = parent2[device_name]

            crossover_point = np.random.randint(1, len(pattern1))

            child1[device_name] = pattern1[:crossover_point] + pattern2[crossover_point:]
            child2[device_name] = pattern2[:crossover_point] + pattern1[crossover_point:]

        return child1, child2

    def _mutate(self, individual: Dict, mutation_rate: float) -> Dict:
        """Mutation operation for genetic algorithm."""
        mutated = {}

        for device_name, pattern in individual.items():
            mutated_pattern = []
            for value in pattern:
                if np.random.random() < mutation_rate:
                    new_value = value + np.random.normal(0, 0.02)
                    new_value = max(0.0, min(1.0, new_value))
                    mutated_pattern.append(new_value)
                else:
                    mutated_pattern.append(value)
            mutated[device_name] = mutated_pattern

        return mutated

    def save_optimized_config(self, output_path: str = "optimized_config.yaml.tmp"):
        """Save optimized patterns to a new config file."""
        if not self.best_patterns:
            self.logger.warning("No optimized patterns to save")
            return

        optimized_config = copy.deepcopy(self.config)

        # Update device patterns
        for device_name, pattern in self.best_patterns.items():
            if device_name in optimized_config['devices']:
                optimized_config['devices'][device_name]['daily_pattern'] = pattern

        # Add optimization metadata
        total_eval_records = sum(len(self.target_data.loc[start:end]) for start, end in self.evaluation_periods)

        optimized_config['optimization_metadata'] = {
            'optimization_date': datetime.now().isoformat(),
            'best_score': float(self.best_score),
            'total_target_records': len(self.target_data),
            'evaluation_records_used': total_eval_records,
            'evaluation_periods': len(self.evaluation_periods),
            'training_data_period': {
                'start': self.target_data.index.min().isoformat() if self.target_data is not None else None,
                'end': self.target_data.index.max().isoformat() if self.target_data is not None else None
            },
            'location': self.location,
            'optimized_devices': list(self.best_patterns.keys()),
            'generations_run': len(self.best_scores_history)
        }

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(optimized_config, file, default_flow_style=False, indent=2)

        self.logger.info(f"Optimized configuration saved to: {output_path}")
        return output_path

    def generate_comparison_plots(self, output_dir: str = "optimization_plots"):
        """Generate enhanced plots comparing optimized vs original patterns."""
        if not self.best_patterns:
            self.logger.warning("No optimized patterns to plot")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Plot device patterns comparison
        devices_to_plot = list(self.best_patterns.keys())

        fig, axes = plt.subplots(len(devices_to_plot), 1, figsize=(15, 4 * len(devices_to_plot)))
        if len(devices_to_plot) == 1:
            axes = [axes]

        time_labels = []
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                time_labels.append(f"{hour:02d}:{minute:02d}")

        for i, device_name in enumerate(devices_to_plot):
            ax = axes[i]

            original_pattern = self.original_patterns.get(device_name, [0.5] * 96)
            optimized_pattern = self.best_patterns[device_name]

            x_pos = range(96)
            ax.plot(x_pos, original_pattern, label='Original', linewidth=2, alpha=0.7, color='#ff7f0e')
            ax.plot(x_pos, optimized_pattern, label='Optimized', linewidth=2, alpha=0.9, color='#2ca02c')

            ax.set_title(f'{device_name.replace("_", " ").title()} - Pattern Comparison')
            ax.set_xlabel('Time of Day (15-minute intervals)')
            ax.set_ylabel('Usage Factor (0-1)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Set x-axis ticks (every 2 hours)
            tick_positions = range(0, 96, 8)
            tick_labels_filtered = [time_labels[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels_filtered, rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/pattern_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot optimization progress with stage information
        if self.best_scores_history:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Score progress
            ax1.plot(self.best_scores_history, linewidth=2, color='#1f77b4')
            ax1.set_title('Optimization Progress')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Best Score (lower is better)')
            ax1.grid(True, alpha=0.3)

            # Add stage markers
            for stage in range(1, 5):
                gen = stage * 25
                if gen < len(self.best_scores_history):
                    ax1.axvline(x=gen, color='red', linestyle='--', alpha=0.5)
                    ax1.text(gen, max(self.best_scores_history) * 0.9, f'Stage {stage+1}',
                             rotation=90, alpha=0.7)

            # Records used over time
            records_used = []
            for gen in range(len(self.best_scores_history)):
                stage = (gen // 25) + 1
                periods = min(stage, len(self.evaluation_periods))
                total_records = 0
                for period_idx in range(periods):
                    if period_idx < len(self.evaluation_periods):
                        start_date, end_date = self.evaluation_periods[period_idx]
                        try:
                            period_data = self.target_data.loc[start_date:end_date]
                            total_records += len(period_data)
                        except:
                            pass
                records_used.append(total_records)

            ax2.plot(records_used, linewidth=2, color='#ff7f0e')
            ax2.set_title('Evaluation Dataset Size Over Time')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Records Used for Evaluation')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/optimization_progress.png", dpi=300, bbox_inches='tight')
            plt.close()

        self.logger.info(f"Enhanced comparison plots saved to: {output_dir}")

def main():
    """Main function for standalone pattern optimization."""
    import argparse

    parser = argparse.ArgumentParser(description='Optimize device load patterns using real load data')
    parser.add_argument('--training-data', required=True, help='Path to Excel file with training data')
    parser.add_argument('--location', required=True, help='Location for weather data')
    parser.add_argument('--config', default='config.yaml', help='Main configuration file')
    parser.add_argument('--optimization-config', default='optimization_config.yaml', help='Optimization configuration file')
    parser.add_argument('--output', default='optimized_config.yaml.tmp', help='Output path for optimized config')
    parser.add_argument('--plots-dir', default='optimization_plots', help='Directory for output plots')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-live-monitor', action='store_true', help='Disable live monitoring web interface')
    parser.add_argument('--full-dataset', action='store_true', help='Use full dataset instead of progressive evaluation')

    args = parser.parse_args()

    # Setup logging with reduced matplotlib verbosity
    if args.verbose:
        log_level = logging.DEBUG
        # Still reduce matplotlib/PIL noise even in verbose mode
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger('PIL').setLevel(logging.INFO)
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    try:
        # Initialize optimizer
        logger.info("Initializing enhanced pattern optimizer...")
        optimizer = PatternOptimizer(args.config, args.optimization_config)

        # Override full dataset setting if specified
        if args.full_dataset:
            optimizer.optimization_config['evaluation']['use_full_dataset'] = True
            logger.info("Overriding to use full dataset for evaluation")

        # Load training data
        logger.info("Loading training data...")
        optimizer.load_training_data(args.training_data, args.location)

        # Test evaluation function
        logger.info("Testing evaluation function...")
        current_patterns = {}
        for device_name, device_config in optimizer.config.get('devices', {}).items():
            current_patterns[device_name] = device_config.get('daily_pattern', [0.5] * 96)

        test_score = optimizer.evaluate_patterns(current_patterns)
        logger.info(f"Initial evaluation score: {test_score}")

        if test_score == float('inf'):
            logger.error("Evaluation function returns infinite score - check your data and configuration")
            return

        # Run optimization
        logger.info("Starting enhanced pattern optimization...")
        best_patterns = optimizer.optimize_patterns_genetic(enable_live_monitoring=not args.no_live_monitor)

        if best_patterns:
            # Save optimized config
            output_path = optimizer.save_optimized_config(args.output)

            # Generate comparison plots
            optimizer.generate_comparison_plots(args.plots_dir)

            logger.info("=== Optimization Complete ===")
            logger.info(f"Best score: {optimizer.best_score:.6f}")
            logger.info(f"Optimized config saved to: {output_path}")
            logger.info(f"Comparison plots saved to: {args.plots_dir}")
            logger.info(f"To use optimized patterns: python main.py --config {output_path} [other args]")
        else:
            logger.error("Optimization failed to produce results")

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)

if __name__ == '__main__':
    main()