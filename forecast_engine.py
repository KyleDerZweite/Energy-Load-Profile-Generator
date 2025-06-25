"""
Energy Forecasting Engine
=========================

This module provides advanced energy forecasting capabilities for building energy systems.
It can generate energy forecasts for future periods using trained energy disaggregation models
and weather forecasts or scenarios.

Key Features:
- Multi-scenario forecasting (normal, warm, cold weather scenarios)
- Uncertainty quantification and confidence intervals
- Long-term trend extrapolation
- Seasonal pattern forecasting
- Device-level energy forecasting
- Climate change scenario modeling
- Forecast validation and performance tracking

This engine extends the building energy model with sophisticated forecasting capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy import stats
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class WeatherScenario(Enum):
    """Weather scenario types for forecasting."""
    HISTORICAL = "historical"           # Use historical weather patterns
    CLIMATE_NORMAL = "climate_normal"   # Use 30-year climate normal
    WARM_SCENARIO = "warm_scenario"     # +2°C warming scenario
    COLD_SCENARIO = "cold_scenario"     # -2°C cooling scenario
    EXTREME_HEAT = "extreme_heat"       # Extreme heat scenario
    EXTREME_COLD = "extreme_cold"       # Extreme cold scenario
    CUSTOM = "custom"                   # User-defined scenario


class ForecastHorizon(Enum):
    """Forecast time horizons."""
    SHORT_TERM = "short_term"           # Days to weeks
    MEDIUM_TERM = "medium_term"         # Months
    LONG_TERM = "long_term"             # Years to decades


@dataclass
class ForecastScenario:
    """Definition of a forecast scenario."""
    name: str
    description: str
    weather_scenario: WeatherScenario
    temperature_adjustment: float       # °C adjustment from baseline
    seasonal_adjustment: float          # Seasonal pattern adjustment factor
    trend_factor: float                # Long-term trend factor
    uncertainty_factor: float          # Additional uncertainty factor


@dataclass
class ForecastResult:
    """Result of energy forecasting."""
    scenario_name: str
    forecast_years: List[int]
    total_energy_forecast: np.ndarray
    device_energy_forecasts: Dict[str, np.ndarray]
    confidence_intervals: Dict[str, Dict[str, np.ndarray]]
    forecast_statistics: Dict[str, float]
    weather_data_used: pd.DataFrame
    forecast_metadata: Dict[str, Any]


@dataclass
class UncertaintyAnalysis:
    """Uncertainty analysis results."""
    total_uncertainty: float
    weather_uncertainty: float
    model_uncertainty: float
    scenario_uncertainty: float
    confidence_intervals_68: Tuple[np.ndarray, np.ndarray]
    confidence_intervals_95: Tuple[np.ndarray, np.ndarray]
    uncertainty_breakdown: Dict[str, float]


class EnergyForecastEngine:
    """
    Advanced energy forecasting engine for building energy systems.
    
    This engine provides sophisticated forecasting capabilities including
    multi-scenario analysis, uncertainty quantification, and long-term trends.
    """
    
    def __init__(self, building_model=None, logger=None):
        self.building_model = building_model
        self.logger = logger or logging.getLogger(__name__)
        
        # Forecast parameters
        self.default_scenarios = self._initialize_default_scenarios()
        self.forecast_cache = {}
        self.validation_results = {}
        
        # Climate data for scenario generation
        self.climate_normals = {}
        self.historical_weather_stats = {}
        
        self.logger.info("🔮 Energy Forecast Engine initialized")
    
    def _initialize_default_scenarios(self) -> Dict[str, ForecastScenario]:
        """Initialize default forecast scenarios."""
        return {
            'baseline': ForecastScenario(
                name='Baseline',
                description='Normal weather conditions based on historical patterns',
                weather_scenario=WeatherScenario.CLIMATE_NORMAL,
                temperature_adjustment=0.0,
                seasonal_adjustment=1.0,
                trend_factor=1.0,
                uncertainty_factor=1.0
            ),
            'warm_climate': ForecastScenario(
                name='Warm Climate',
                description='Climate warming scenario (+3°C, increased cooling demand)',
                weather_scenario=WeatherScenario.WARM_SCENARIO,
                temperature_adjustment=3.0,
                seasonal_adjustment=1.2,  # Enhanced summer patterns
                trend_factor=1.05,  # 5% growth trend
                uncertainty_factor=1.3
            ),
            'cold_climate': ForecastScenario(
                name='Cold Climate',
                description='Unusually cold conditions (-3°C, increased heating)',
                weather_scenario=WeatherScenario.COLD_SCENARIO,
                temperature_adjustment=-3.0,
                seasonal_adjustment=1.2,  # Enhanced winter patterns
                trend_factor=1.03,  # 3% growth trend
                uncertainty_factor=1.3
            ),
            'extreme_heat': ForecastScenario(
                name='Extreme Heat',
                description='Extreme heat wave conditions (+6°C, major cooling impact)',
                weather_scenario=WeatherScenario.EXTREME_HEAT,
                temperature_adjustment=6.0,
                seasonal_adjustment=1.4,  # Significantly enhanced summer demand
                trend_factor=1.08,  # 8% growth trend
                uncertainty_factor=1.6
            )
        }
    
    def generate_weather_scenarios(self, base_weather: pd.DataFrame,
                                 forecast_years: List[int],
                                 scenarios: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate weather scenarios for forecasting.
        
        Args:
            base_weather: Historical weather data to use as baseline
            forecast_years: Years to generate forecasts for
            scenarios: List of scenario names to generate
            
        Returns:
            Dictionary mapping scenario names to weather DataFrames
        """
        if scenarios is None:
            scenarios = ['baseline', 'warm_climate', 'cold_climate']
        
        self.logger.info(f"🌤️ Generating weather scenarios: {scenarios}")
        
        # Analyze historical weather patterns
        weather_stats = self._analyze_historical_weather(base_weather)
        self.historical_weather_stats = weather_stats
        
        # Generate time series for forecast years
        forecast_timestamps = self._generate_forecast_timestamps(forecast_years)
        
        # Generate scenarios
        weather_scenarios = {}
        for scenario_name in scenarios:
            if scenario_name not in self.default_scenarios:
                self.logger.warning(f"⚠️ Unknown scenario '{scenario_name}', using baseline")
                scenario_name = 'baseline'
            
            scenario = self.default_scenarios[scenario_name]
            weather_data = self._generate_scenario_weather(
                forecast_timestamps, weather_stats, scenario
            )
            weather_scenarios[scenario_name] = weather_data
        
        self.logger.info(f"✅ Generated {len(weather_scenarios)} weather scenarios")
        return weather_scenarios
    
    def forecast_multiple_scenarios(self, base_weather: pd.DataFrame,
                                  forecast_years: List[int],
                                  scenarios: List[str] = None,
                                  include_uncertainty: bool = True) -> Dict[str, ForecastResult]:
        """
        Generate energy forecasts for multiple scenarios.
        
        Args:
            base_weather: Historical weather data
            forecast_years: Years to forecast
            scenarios: Scenarios to run
            include_uncertainty: Whether to include uncertainty analysis
            
        Returns:
            Dictionary mapping scenario names to forecast results
        """
        if self.building_model is None or not self.building_model.is_trained:
            raise ValueError("Building model must be trained before forecasting")
        
        self.logger.info(f"🎯 Forecasting energy for scenarios: {scenarios}")
        
        # Generate weather scenarios
        weather_scenarios = self.generate_weather_scenarios(
            base_weather, forecast_years, scenarios
        )
        
        # Run forecasts for each scenario
        forecast_results = {}
        for scenario_name, weather_data in weather_scenarios.items():
            self.logger.info(f"🔮 Running forecast for scenario: {scenario_name}")
            
            # Generate forecast using building model
            scenario_obj = self.default_scenarios[scenario_name]
            forecast_config = self._create_forecast_config(scenario_obj, forecast_years)
            
            try:
                building_forecast = self.building_model.forecast_energy(
                    weather_data, forecast_config
                )
                
                # Add uncertainty analysis if requested
                uncertainty_analysis = None
                if include_uncertainty:
                    uncertainty_analysis = self._analyze_forecast_uncertainty(
                        building_forecast, scenario_obj
                    )
                
                # Compile forecast result
                forecast_result = ForecastResult(
                    scenario_name=scenario_name,
                    forecast_years=forecast_years,
                    total_energy_forecast=building_forecast['total_energy_forecast'],
                    device_energy_forecasts=building_forecast['device_profiles_forecast'],
                    confidence_intervals=uncertainty_analysis.confidence_intervals_95 if uncertainty_analysis else {},
                    forecast_statistics=building_forecast['forecast_metrics'],
                    weather_data_used=weather_data,
                    forecast_metadata={
                        'scenario': asdict(scenario_obj),
                        'forecast_date': datetime.now().isoformat(),
                        'uncertainty_included': include_uncertainty,
                        'building_model_version': self.building_model.model_metadata.get('model_version', '1.0.0')
                    }
                )
                
                forecast_results[scenario_name] = forecast_result
                
            except Exception as e:
                self.logger.error(f"❌ Error forecasting scenario {scenario_name}: {e}")
                continue
        
        self.logger.info(f"✅ Completed forecasts for {len(forecast_results)} scenarios")
        return forecast_results
    
    def _analyze_historical_weather(self, weather_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze historical weather patterns for scenario generation."""
        # Find temperature column
        temp_cols = ['temperature', 'Temperature', 'temp', 'Temp']
        temp_col = None
        for col in temp_cols:
            if col in weather_data.columns:
                temp_col = col
                break
        
        if temp_col is None:
            self.logger.warning("⚠️ No temperature column found, using default statistics")
            return self._get_default_weather_stats()
        
        # Extract timestamp
        timestamp_cols = ['timestamp', 'Timestamp', 'time', 'Time']
        ts_col = None
        for col in timestamp_cols:
            if col in weather_data.columns:
                ts_col = col
                break
        
        if ts_col is None:
            weather_data['timestamp'] = weather_data.index
            ts_col = 'timestamp'
        
        weather_data = weather_data.copy()
        weather_data[ts_col] = pd.to_datetime(weather_data[ts_col])
        weather_data['month'] = weather_data[ts_col].dt.month
        weather_data['day_of_year'] = weather_data[ts_col].dt.dayofyear
        weather_data['hour'] = weather_data[ts_col].dt.hour
        
        temperature = weather_data[temp_col].values
        
        # Calculate statistics
        stats = {
            'overall': {
                'mean': float(np.mean(temperature)),
                'std': float(np.std(temperature)),
                'min': float(np.min(temperature)),
                'max': float(np.max(temperature)),
                'percentiles': {
                    p: float(np.percentile(temperature, p)) 
                    for p in [5, 10, 25, 50, 75, 90, 95]
                }
            },
            'monthly': {},
            'daily_cycle': {},
            'seasonal_pattern': []
        }
        
        # Monthly statistics
        for month in range(1, 13):
            month_data = weather_data[weather_data['month'] == month][temp_col]
            if len(month_data) > 0:
                stats['monthly'][month] = {
                    'mean': float(np.mean(month_data)),
                    'std': float(np.std(month_data)),
                    'min': float(np.min(month_data)),
                    'max': float(np.max(month_data))
                }
            else:
                stats['monthly'][month] = stats['overall'].copy()
        
        # Daily cycle (24-hour pattern)
        for hour in range(24):
            hour_data = weather_data[weather_data['hour'] == hour][temp_col]
            if len(hour_data) > 0:
                stats['daily_cycle'][hour] = {
                    'mean': float(np.mean(hour_data)),
                    'std': float(np.std(hour_data))
                }
            else:
                stats['daily_cycle'][hour] = {
                    'mean': stats['overall']['mean'],
                    'std': stats['overall']['std']
                }
        
        # Seasonal pattern (day of year)
        if len(weather_data) > 365:
            daily_means = weather_data.groupby('day_of_year')[temp_col].mean()
            stats['seasonal_pattern'] = daily_means.tolist()
        
        return stats
    
    def _get_default_weather_stats(self) -> Dict[str, Any]:
        """Get default weather statistics when no data is available."""
        # Default temperate climate statistics
        return {
            'overall': {
                'mean': 15.0,
                'std': 8.0,
                'min': -10.0,
                'max': 35.0,
                'percentiles': {5: -5, 10: -2, 25: 8, 50: 15, 75: 22, 90: 28, 95: 32}
            },
            'monthly': {
                month: {
                    'mean': 15.0 + 10 * np.sin(2 * np.pi * (month - 1) / 12),
                    'std': 8.0,
                    'min': -10.0,
                    'max': 35.0
                } for month in range(1, 13)
            },
            'daily_cycle': {
                hour: {
                    'mean': 15.0 + 5 * np.sin(2 * np.pi * (hour - 6) / 24),
                    'std': 3.0
                } for hour in range(24)
            },
            'seasonal_pattern': [15.0 + 10 * np.sin(2 * np.pi * day / 365) for day in range(365)]
        }
    
    def _generate_forecast_timestamps(self, forecast_years: List[int]) -> pd.DatetimeIndex:
        """Generate timestamps for forecast period."""
        start_year = min(forecast_years)
        end_year = max(forecast_years)
        
        start_date = f"{start_year}-01-01 00:00:00"
        end_date = f"{end_year}-12-31 23:45:00"
        
        return pd.date_range(start=start_date, end=end_date, freq='15min')
    
    def _generate_scenario_weather(self, timestamps: pd.DatetimeIndex,
                                 weather_stats: Dict[str, Any],
                                 scenario: ForecastScenario) -> pd.DataFrame:
        """Generate weather data for a specific scenario."""
        n_points = len(timestamps)
        
        # Extract time components
        months = timestamps.month
        days_of_year = timestamps.dayofyear
        hours = timestamps.hour
        
        # Generate base temperature using seasonal pattern
        base_temperature = np.zeros(n_points)
        
        if weather_stats['seasonal_pattern']:
            # Use learned seasonal pattern
            seasonal_pattern = np.array(weather_stats['seasonal_pattern'])
            for i, day in enumerate(days_of_year):
                day_index = min(day - 1, len(seasonal_pattern) - 1)
                base_temperature[i] = seasonal_pattern[day_index]
        else:
            # Use monthly averages
            for i, month in enumerate(months):
                base_temperature[i] = weather_stats['monthly'][month]['mean']
        
        # Add daily cycle
        daily_cycle_amplitude = 3.0  # °C daily temperature variation
        for i, hour in enumerate(hours):
            daily_factor = np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 3 PM
            base_temperature[i] += daily_cycle_amplitude * daily_factor
        
        # Apply scenario adjustments
        scenario_temperature = base_temperature + scenario.temperature_adjustment
        
        # Add random variations
        noise_std = weather_stats['overall']['std'] * 0.3  # 30% of historical variability
        noise = np.random.normal(0, noise_std, n_points)
        scenario_temperature += noise * scenario.uncertainty_factor
        
        # Apply seasonal adjustment for extreme scenarios
        if scenario.seasonal_adjustment != 1.0:
            seasonal_deviation = scenario_temperature - weather_stats['overall']['mean']
            scenario_temperature = weather_stats['overall']['mean'] + seasonal_deviation * scenario.seasonal_adjustment
        
        # Create weather DataFrame
        weather_df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': scenario_temperature
        })
        
        # Add other weather variables if needed (simplified)
        weather_df['humidity'] = 50.0  # Default humidity
        weather_df['wind_speed'] = 5.0  # Default wind speed
        weather_df['solar_radiation'] = self._generate_solar_radiation(timestamps)
        
        return weather_df
    
    def _generate_solar_radiation(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Generate synthetic solar radiation data."""
        n_points = len(timestamps)
        hours = timestamps.hour
        days_of_year = timestamps.dayofyear
        
        solar_radiation = np.zeros(n_points)
        
        for i, (hour, day) in enumerate(zip(hours, days_of_year)):
            # Solar elevation factor (simplified)
            hour_factor = max(0, np.sin(np.pi * (hour - 6) / 12))  # Daylight hours
            
            # Seasonal factor
            seasonal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (day - 80) / 365)  # Peak in summer
            
            # Maximum solar radiation (W/m²)
            max_radiation = 1000
            
            solar_radiation[i] = max_radiation * hour_factor * seasonal_factor
        
        return solar_radiation
    
    def _create_forecast_config(self, scenario: ForecastScenario, 
                              forecast_years: List[int]):
        """Create forecast configuration from scenario."""
        from building_model import ForecastConfig
        
        config = ForecastConfig(
            forecast_years=forecast_years,
            weather_scenario=scenario.weather_scenario.value,
            uncertainty_estimation=True,
            seasonal_adjustment=scenario.seasonal_adjustment != 1.0,
            trend_extrapolation=scenario.trend_factor != 1.0
        )
        
        # Add scenario-specific parameters
        config.seasonal_factor = scenario.seasonal_adjustment
        config.trend_factor = scenario.trend_factor
        
        return config
    
    def _analyze_forecast_uncertainty(self, building_forecast: Dict[str, Any],
                                    scenario: ForecastScenario) -> UncertaintyAnalysis:
        """Analyze uncertainty in forecast results."""
        total_energy = building_forecast['total_energy_forecast']
        
        # Get validation metrics for uncertainty estimation
        if hasattr(self.building_model, 'validation_metrics'):
            validation_metrics = self.building_model.validation_metrics
            base_rmse = validation_metrics.get('rmse', np.std(total_energy) * 0.1)
        else:
            base_rmse = np.std(total_energy) * 0.1
        
        # Calculate uncertainty components
        model_uncertainty = base_rmse
        weather_uncertainty = base_rmse * 0.5  # Weather forecast uncertainty
        scenario_uncertainty = base_rmse * scenario.uncertainty_factor
        
        # Total uncertainty (combine in quadrature)
        total_uncertainty = np.sqrt(
            model_uncertainty**2 + 
            weather_uncertainty**2 + 
            scenario_uncertainty**2
        )
        
        # Generate confidence intervals
        ci_68_lower = total_energy - total_uncertainty
        ci_68_upper = total_energy + total_uncertainty
        ci_95_lower = total_energy - 2 * total_uncertainty
        ci_95_upper = total_energy + 2 * total_uncertainty
        
        return UncertaintyAnalysis(
            total_uncertainty=float(total_uncertainty),
            weather_uncertainty=float(weather_uncertainty),
            model_uncertainty=float(model_uncertainty),
            scenario_uncertainty=float(scenario_uncertainty),
            confidence_intervals_68=(ci_68_lower, ci_68_upper),
            confidence_intervals_95=(ci_95_lower, ci_95_upper),
            uncertainty_breakdown={
                'model': float(model_uncertainty / total_uncertainty * 100),
                'weather': float(weather_uncertainty / total_uncertainty * 100),
                'scenario': float(scenario_uncertainty / total_uncertainty * 100)
            }
        )
    
    def validate_forecast_accuracy(self, historical_data: pd.DataFrame,
                                 historical_weather: pd.DataFrame,
                                 validation_years: List[int]) -> Dict[str, Any]:
        """
        Validate forecast accuracy using historical data.
        
        Args:
            historical_data: Historical energy consumption data
            historical_weather: Historical weather data
            validation_years: Years to use for validation
            
        Returns:
            Validation results and metrics
        """
        self.logger.info(f"🔍 Validating forecast accuracy on years {validation_years}")
        
        if self.building_model is None or not self.building_model.is_trained:
            raise ValueError("Building model must be trained before validation")
        
        validation_results = {}
        
        for year in validation_years:
            self.logger.info(f"📊 Validating forecast for year {year}")
            
            # Extract actual data for the year
            actual_energy = self._filter_data_by_year(historical_data, year)
            actual_weather = self._filter_data_by_year(historical_weather, year)
            
            if len(actual_energy) == 0:
                self.logger.warning(f"⚠️ No data found for year {year}")
                continue
            
            # Generate forecast for the year using historical weather
            try:
                # Create forecast scenarios
                forecast_scenarios = self.generate_weather_scenarios(
                    actual_weather, [year], ['baseline']
                )
                
                # Run forecast
                forecast_results = self.forecast_multiple_scenarios(
                    actual_weather, [year], ['baseline'], include_uncertainty=True
                )
                
                if 'baseline' in forecast_results:
                    forecast_result = forecast_results['baseline']
                    
                    # Compare forecast vs actual
                    forecast_energy = forecast_result.total_energy_forecast
                    actual_energy_values = self._extract_energy_values(actual_energy)
                    
                    # Align data lengths
                    min_length = min(len(forecast_energy), len(actual_energy_values))
                    forecast_energy = forecast_energy[:min_length]
                    actual_energy_values = actual_energy_values[:min_length]
                    
                    # Calculate validation metrics
                    year_metrics = self._calculate_forecast_validation_metrics(
                        forecast_energy, actual_energy_values
                    )
                    
                    validation_results[year] = {
                        'metrics': year_metrics,
                        'data_points': min_length,
                        'forecast_result': forecast_result
                    }
                
            except Exception as e:
                self.logger.error(f"❌ Error validating year {year}: {e}")
                continue
        
        # Calculate overall validation summary
        overall_summary = self._summarize_validation_results(validation_results)
        
        self.validation_results = {
            'individual_years': validation_results,
            'overall_summary': overall_summary,
            'validation_metadata': {
                'validation_date': datetime.now().isoformat(),
                'years_validated': validation_years,
                'successful_validations': len(validation_results)
            }
        }
        
        self.logger.info("✅ Forecast validation completed")
        return self.validation_results
    
    def _filter_data_by_year(self, data: pd.DataFrame, year: int) -> pd.DataFrame:
        """Filter DataFrame to include only specified year."""
        # Find timestamp column
        ts_cols = ['timestamp', 'Timestamp', 'time', 'Time']
        ts_col = None
        for col in ts_cols:
            if col in data.columns:
                ts_col = col
                break
        
        if ts_col is None:
            data = data.copy()
            data['timestamp'] = data.index
            ts_col = 'timestamp'
        
        data = data.copy()
        data[ts_col] = pd.to_datetime(data[ts_col])
        year_mask = data[ts_col].dt.year == year
        
        return data[year_mask]
    
    def _extract_energy_values(self, data: pd.DataFrame) -> np.ndarray:
        """Extract energy values from DataFrame."""
        energy_cols = ['Value', 'value', 'energy', 'Energy', 'total_energy']
        
        for col in energy_cols:
            if col in data.columns:
                return data[col].values
        
        raise ValueError("No energy column found in data")
    
    def _calculate_forecast_validation_metrics(self, forecast: np.ndarray,
                                             actual: np.ndarray) -> Dict[str, float]:
        """Calculate validation metrics for forecast accuracy."""
        # Basic metrics
        mae = np.mean(np.abs(forecast - actual))
        rmse = np.sqrt(np.mean((forecast - actual) ** 2))
        mape = np.mean(np.abs((forecast - actual) / actual)) * 100
        
        # Bias metrics
        bias = np.mean(forecast - actual)
        percent_bias = bias / np.mean(actual) * 100
        
        # R-squared
        ss_res = np.sum((actual - forecast) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Peak prediction accuracy
        actual_peak = np.max(actual)
        forecast_peak = np.max(forecast)
        peak_error = abs(forecast_peak - actual_peak) / actual_peak * 100
        
        # Energy conservation
        total_actual = np.sum(actual)
        total_forecast = np.sum(forecast)
        energy_error = abs(total_forecast - total_actual) / total_actual * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2),
            'bias': float(bias),
            'percent_bias': float(percent_bias),
            'peak_error': float(peak_error),
            'energy_conservation_error': float(energy_error),
            'mean_actual': float(np.mean(actual)),
            'mean_forecast': float(np.mean(forecast))
        }
    
    def _summarize_validation_results(self, validation_results: Dict[int, Dict]) -> Dict[str, float]:
        """Summarize validation results across multiple years."""
        if not validation_results:
            return {}
        
        # Extract metrics from all years
        all_metrics = [result['metrics'] for result in validation_results.values()]
        
        # Calculate averages
        summary = {}
        metric_names = ['mae', 'rmse', 'mape', 'r2_score', 'bias', 'percent_bias', 
                       'peak_error', 'energy_conservation_error']
        
        for metric in metric_names:
            values = [metrics[metric] for metrics in all_metrics if metric in metrics]
            if values:
                summary[f'mean_{metric}'] = float(np.mean(values))
                summary[f'std_{metric}'] = float(np.std(values))
        
        return summary
    
    def plot_forecast_scenarios(self, forecast_results: Dict[str, ForecastResult],
                              save_path: Optional[str] = None) -> None:
        """Create visualization of forecast scenarios."""
        if not forecast_results:
            self.logger.warning("⚠️ No forecast results to plot")
            return
        
        # Set up the plotting
        plt.style.use('default')
        n_scenarios = len(forecast_results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Energy Forecast Scenarios Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        scenario_names = list(forecast_results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, n_scenarios))
        
        # Plot 1: Total energy forecasts
        ax1 = axes[0, 0]
        for i, (scenario_name, result) in enumerate(forecast_results.items()):
            total_energy = result.total_energy_forecast
            time_index = np.arange(len(total_energy))
            
            # Plot first 2000 points for visibility
            plot_length = min(2000, len(total_energy))
            ax1.plot(time_index[:plot_length], total_energy[:plot_length], 
                    label=scenario_name, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Energy (kW)')
        ax1.set_title('Total Energy Forecasts')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy statistics comparison
        ax2 = axes[0, 1]
        metrics = ['mean', 'peak', 'min']
        scenario_stats = {}
        
        for scenario_name, result in forecast_results.items():
            total_energy = result.total_energy_forecast
            scenario_stats[scenario_name] = {
                'mean': np.mean(total_energy),
                'peak': np.max(total_energy),
                'min': np.min(total_energy)
            }
        
        x_positions = np.arange(len(scenario_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [scenario_stats[name][metric] for name in scenario_names]
            ax2.bar(x_positions + i * width, values, width, label=metric, alpha=0.8)
        
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Energy (kW)')
        ax2.set_title('Energy Statistics by Scenario')
        ax2.set_xticks(x_positions + width)
        ax2.set_xticklabels(scenario_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temperature scenarios
        ax3 = axes[1, 0]
        for i, (scenario_name, result) in enumerate(forecast_results.items()):
            weather_data = result.weather_data_used
            if 'temperature' in weather_data.columns:
                temperature = weather_data['temperature'].values
                time_index = np.arange(len(temperature))
                
                # Plot first 2000 points for visibility
                plot_length = min(2000, len(temperature))
                ax3.plot(time_index[:plot_length], temperature[:plot_length],
                        label=scenario_name, color=colors[i], alpha=0.8)
        
        ax3.set_xlabel('Time Index')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('Temperature Scenarios')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Uncertainty comparison (if available)
        ax4 = axes[1, 1]
        if any('confidence_intervals' in result.forecast_metadata for result in forecast_results.values()):
            # Plot uncertainty ranges for first scenario with confidence intervals
            for i, (scenario_name, result) in enumerate(forecast_results.items()):
                if 'confidence_intervals' in result.forecast_metadata:
                    # This would need to be implemented based on confidence interval structure
                    pass
            
            ax4.set_title('Forecast Uncertainty (Placeholder)')
        else:
            # Show annual energy totals instead
            annual_totals = {}
            for scenario_name, result in forecast_results.items():
                # Calculate annual totals for each forecast year
                total_energy = result.total_energy_forecast
                # Assuming 15-minute intervals: 4 * 24 * 365 = 35,040 intervals per year
                intervals_per_year = 35040
                
                for year in result.forecast_years:
                    year_start = (year - min(result.forecast_years)) * intervals_per_year
                    year_end = min(year_start + intervals_per_year, len(total_energy))
                    
                    if year_end > year_start:
                        annual_total = np.sum(total_energy[year_start:year_end]) * 0.25  # kWh
                        if scenario_name not in annual_totals:
                            annual_totals[scenario_name] = {}
                        annual_totals[scenario_name][year] = annual_total
            
            # Plot annual totals
            years = sorted(set().union(*[years.keys() for years in annual_totals.values()]))
            x_positions = np.arange(len(years))
            width = 0.8 / len(forecast_results)
            
            for i, (scenario_name, year_data) in enumerate(annual_totals.items()):
                values = [year_data.get(year, 0) for year in years]
                ax4.bar(x_positions + i * width, values, width, label=scenario_name, alpha=0.8)
            
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Annual Energy (kWh)')
            ax4.set_title('Annual Energy Totals by Scenario')
            ax4.set_xticks(x_positions + width * (len(forecast_results) - 1) / 2)
            ax4.set_xticklabels(years)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Forecast scenario plots saved to {save_path}")
        
        plt.show()
    
    def export_forecast_results(self, forecast_results: Dict[str, ForecastResult],
                              filepath: str) -> None:
        """Export forecast results to file."""
        import json
        
        # Prepare export data
        export_data = {
            'export_metadata': {
                'export_date': datetime.now().isoformat(),
                'forecast_engine_version': '1.0.0',
                'number_of_scenarios': len(forecast_results)
            },
            'scenarios': {}
        }
        
        for scenario_name, result in forecast_results.items():
            # Convert numpy arrays to lists for JSON serialization
            scenario_data = {
                'forecast_years': result.forecast_years,
                'total_energy_forecast': result.total_energy_forecast.tolist(),
                'device_energy_forecasts': {
                    device: energy.tolist() 
                    for device, energy in result.device_energy_forecasts.items()
                },
                'forecast_statistics': result.forecast_statistics,
                'forecast_metadata': result.forecast_metadata
            }
            
            export_data['scenarios'][scenario_name] = scenario_data
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"📄 Forecast results exported to {filepath}")
    
    def get_forecast_summary(self, forecast_results: Dict[str, ForecastResult]) -> Dict[str, Any]:
        """Get summary of forecast results across scenarios."""
        if not forecast_results:
            return {}
        
        # Calculate cross-scenario statistics
        all_totals = [np.sum(result.total_energy_forecast) for result in forecast_results.values()]
        all_peaks = [np.max(result.total_energy_forecast) for result in forecast_results.values()]
        all_means = [np.mean(result.total_energy_forecast) for result in forecast_results.values()]
        
        summary = {
            'number_of_scenarios': len(forecast_results),
            'scenario_names': list(forecast_results.keys()),
            'forecast_years': forecast_results[list(forecast_results.keys())[0]].forecast_years,
            'cross_scenario_statistics': {
                'total_energy': {
                    'mean': float(np.mean(all_totals)),
                    'std': float(np.std(all_totals)),
                    'min': float(np.min(all_totals)),
                    'max': float(np.max(all_totals))
                },
                'peak_demand': {
                    'mean': float(np.mean(all_peaks)),
                    'std': float(np.std(all_peaks)),
                    'min': float(np.min(all_peaks)),
                    'max': float(np.max(all_peaks))
                },
                'average_load': {
                    'mean': float(np.mean(all_means)),
                    'std': float(np.std(all_means)),
                    'min': float(np.min(all_means)),
                    'max': float(np.max(all_means))
                }
            },
            'scenario_comparison': {
                scenario_name: {
                    'total_energy': float(np.sum(result.total_energy_forecast)),
                    'peak_demand': float(np.max(result.total_energy_forecast)),
                    'average_load': float(np.mean(result.total_energy_forecast)),
                    'energy_statistics': result.forecast_statistics
                }
                for scenario_name, result in forecast_results.items()
            }
        }
        
        return summary