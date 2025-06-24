"""
Weather-Energy Relationship Analyzer
====================================

This module analyzes the relationship between weather conditions (primarily temperature)
and building energy consumption. It identifies heating/cooling signatures, seasonal patterns,
and weather-dependent energy components for energy disaggregation.

Key Features:
- Degree-day analysis for heating and cooling
- Temperature response modeling
- Seasonal energy pattern identification
- Weather-dependent vs weather-independent load separation
- Statistical analysis of weather-energy correlations

Used by the energy disaggregator to understand how building energy responds to weather.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class WeatherEnergySignature:
    """Weather-energy relationship signature for a specific pattern."""
    signature_type: str                    # 'heating', 'cooling', 'baseload', 'weather_independent'
    temperature_threshold: float           # Temperature threshold for response
    coefficient: float                     # Energy response coefficient (kW/°C or kW/DD)
    intercept: float                      # Base energy at threshold
    r_squared: float                      # Goodness of fit
    temperature_range: Tuple[float, float] # Valid temperature range
    seasonal_factor: float                # Seasonal adjustment factor
    time_of_day_factor: List[float]       # 24-hour time-of-day factors


@dataclass
class WeatherEnergyAnalysisResult:
    """Result of comprehensive weather-energy analysis."""
    heating_signature: Optional[WeatherEnergySignature]
    cooling_signature: Optional[WeatherEnergySignature]
    baseload_signature: WeatherEnergySignature
    weather_independent_load: float       # kW, constant load independent of weather
    seasonal_patterns: Dict[str, List[float]]  # Monthly patterns by component
    temperature_statistics: Dict[str, float]   # Temperature data statistics
    energy_statistics: Dict[str, float]        # Energy data statistics
    correlation_analysis: Dict[str, float]     # Various correlation metrics
    degree_day_analysis: Dict[str, Any]        # Heating and cooling degree day analysis
    load_decomposition: Dict[str, np.ndarray]  # Decomposed load components


class WeatherEnergyAnalyzer:
    """
    Analyzes weather-energy relationships for building energy disaggregation.
    
    This analyzer identifies how building energy consumption responds to weather
    conditions, particularly temperature, to support accurate energy disaggregation.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Analysis parameters
        self.heating_base_temp = 15.0      # °C, base temperature for heating degree days
        self.cooling_base_temp = 22.0      # °C, base temperature for cooling degree days
        self.min_correlation_threshold = 0.3  # Minimum correlation to consider significant
        self.min_data_points = 100         # Minimum data points for reliable analysis
        
        # Results storage
        self.analysis_results: Optional[WeatherEnergyAnalysisResult] = None
        self.raw_data: Optional[pd.DataFrame] = None
        
        self.logger.info("🌡️ Weather-Energy Analyzer initialized")
    
    def analyze_weather_energy_relationship(self, energy_data: pd.DataFrame, 
                                          weather_data: pd.DataFrame) -> WeatherEnergyAnalysisResult:
        """
        Perform comprehensive weather-energy relationship analysis.
        
        Args:
            energy_data: DataFrame with timestamp and energy consumption
            weather_data: DataFrame with timestamp and weather parameters
            
        Returns:
            WeatherEnergyAnalysisResult with comprehensive analysis
        """
        self.logger.info("🔍 Starting comprehensive weather-energy analysis")
        
        # Align and prepare data
        aligned_data = self._align_weather_energy_data(energy_data, weather_data)
        self.raw_data = aligned_data.copy()
        
        if len(aligned_data) < self.min_data_points:
            raise ValueError(f"Insufficient data points: {len(aligned_data)} < {self.min_data_points}")
        
        # Extract energy and temperature arrays
        energy = self._extract_energy_values(aligned_data)
        temperature = self._extract_temperature_values(aligned_data)
        
        # Basic statistical analysis
        energy_stats = self._calculate_energy_statistics(energy)
        temp_stats = self._calculate_temperature_statistics(temperature)
        correlation_analysis = self._calculate_correlation_metrics(energy, temperature, aligned_data)
        
        # Degree day analysis
        degree_day_analysis = self._analyze_degree_days(energy, temperature, aligned_data)
        
        # Load decomposition
        load_decomposition = self._decompose_energy_loads(energy, temperature, aligned_data)
        
        # Generate weather-energy signatures
        heating_signature = self._analyze_heating_signature(energy, temperature, aligned_data)
        cooling_signature = self._analyze_cooling_signature(energy, temperature, aligned_data)
        baseload_signature = self._analyze_baseload_signature(energy, temperature, aligned_data)
        
        # Weather-independent load estimation
        weather_independent_load = self._estimate_weather_independent_load(energy, temperature)
        
        # Seasonal pattern analysis
        seasonal_patterns = self._analyze_seasonal_patterns(aligned_data)
        
        # Compile results
        self.analysis_results = WeatherEnergyAnalysisResult(
            heating_signature=heating_signature,
            cooling_signature=cooling_signature,
            baseload_signature=baseload_signature,
            weather_independent_load=weather_independent_load,
            seasonal_patterns=seasonal_patterns,
            temperature_statistics=temp_stats,
            energy_statistics=energy_stats,
            correlation_analysis=correlation_analysis,
            degree_day_analysis=degree_day_analysis,
            load_decomposition=load_decomposition
        )
        
        self.logger.info("✅ Weather-energy analysis completed")
        self._log_analysis_summary()
        
        return self.analysis_results
    
    def _align_weather_energy_data(self, energy_data: pd.DataFrame, 
                                  weather_data: pd.DataFrame) -> pd.DataFrame:
        """Align energy and weather data temporally."""
        # Find timestamp columns
        energy_ts_col = self._find_timestamp_column(energy_data)
        weather_ts_col = self._find_timestamp_column(weather_data)
        
        # Convert to datetime
        energy_data = energy_data.copy()
        weather_data = weather_data.copy()
        energy_data[energy_ts_col] = pd.to_datetime(energy_data[energy_ts_col])
        weather_data[weather_ts_col] = pd.to_datetime(weather_data[weather_ts_col])
        
        # Merge on timestamp
        aligned = pd.merge(
            energy_data, 
            weather_data, 
            left_on=energy_ts_col, 
            right_on=weather_ts_col,
            how='inner',
            suffixes=('_energy', '_weather')
        )
        
        # Use energy timestamp as primary
        aligned['timestamp'] = aligned[energy_ts_col]
        
        self.logger.info(f"📊 Aligned {len(aligned)} data points")
        return aligned
    
    def _find_timestamp_column(self, df: pd.DataFrame) -> str:
        """Find timestamp column in DataFrame."""
        timestamp_cols = ['timestamp', 'Timestamp', 'time', 'Time', 'date', 'Date']
        
        for col in timestamp_cols:
            if col in df.columns:
                return col
        
        # If no timestamp column found, use index
        if isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index
            return 'timestamp'
        
        raise ValueError("No timestamp column found in data")
    
    def _extract_energy_values(self, data: pd.DataFrame) -> np.ndarray:
        """Extract energy values from aligned data."""
        energy_cols = ['Value', 'value', 'energy', 'Energy', 'total_energy', 'load']
        
        for col in energy_cols:
            if col in data.columns:
                return data[col].values
        
        raise ValueError("No energy column found in data")
    
    def _extract_temperature_values(self, data: pd.DataFrame) -> np.ndarray:
        """Extract temperature values from aligned data."""
        temp_cols = ['temperature', 'Temperature', 'temp', 'Temp', 'outdoor_temp']
        
        for col in temp_cols:
            if col in data.columns:
                return data[col].values
        
        # Return default temperature if not found
        self.logger.warning("⚠️ No temperature column found, using default 20°C")
        return np.full(len(data), 20.0)
    
    def _calculate_energy_statistics(self, energy: np.ndarray) -> Dict[str, float]:
        """Calculate basic energy statistics."""
        return {
            'mean': float(np.mean(energy)),
            'median': float(np.median(energy)),
            'std': float(np.std(energy)),
            'min': float(np.min(energy)),
            'max': float(np.max(energy)),
            'percentile_5': float(np.percentile(energy, 5)),
            'percentile_95': float(np.percentile(energy, 95)),
            'range': float(np.max(energy) - np.min(energy)),
            'coefficient_of_variation': float(np.std(energy) / np.mean(energy)) if np.mean(energy) > 0 else 0
        }
    
    def _calculate_temperature_statistics(self, temperature: np.ndarray) -> Dict[str, float]:
        """Calculate basic temperature statistics."""
        return {
            'mean': float(np.mean(temperature)),
            'median': float(np.median(temperature)),
            'std': float(np.std(temperature)),
            'min': float(np.min(temperature)),
            'max': float(np.max(temperature)),
            'range': float(np.max(temperature) - np.min(temperature))
        }
    
    def _calculate_correlation_metrics(self, energy: np.ndarray, temperature: np.ndarray,
                                     data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various correlation metrics between energy and temperature."""
        # Linear correlation
        linear_corr = np.corrcoef(energy, temperature)[0, 1]
        
        # Spearman rank correlation
        if HAS_SCIPY:
            spearman_corr, spearman_p = stats.spearmanr(energy, temperature)
        else:
            spearman_corr, spearman_p = linear_corr, 0.05
        
        # Temperature difference correlations
        temp_abs_diff = np.abs(temperature - np.mean(temperature))
        abs_diff_corr = np.corrcoef(energy, temp_abs_diff)[0, 1]
        
        # Heating degree day correlation
        hdd = np.maximum(0, self.heating_base_temp - temperature)
        hdd_corr = np.corrcoef(energy, hdd)[0, 1] if np.sum(hdd) > 0 else 0
        
        # Cooling degree day correlation
        cdd = np.maximum(0, temperature - self.cooling_base_temp)
        cdd_corr = np.corrcoef(energy, cdd)[0, 1] if np.sum(cdd) > 0 else 0
        
        return {
            'linear_correlation': float(linear_corr),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'abs_temp_diff_correlation': float(abs_diff_corr),
            'heating_degree_day_correlation': float(hdd_corr),
            'cooling_degree_day_correlation': float(cdd_corr)
        }
    
    def _analyze_degree_days(self, energy: np.ndarray, temperature: np.ndarray, 
                           data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze heating and cooling degree days relationships."""
        # Calculate degree days
        hdd = np.maximum(0, self.heating_base_temp - temperature)
        cdd = np.maximum(0, temperature - self.cooling_base_temp)
        
        # Heating degree day analysis
        heating_analysis = self._fit_degree_day_model(energy, hdd, "heating")
        
        # Cooling degree day analysis
        cooling_analysis = self._fit_degree_day_model(energy, cdd, "cooling")
        
        # Combined model (base + heating + cooling)
        combined_analysis = self._fit_combined_degree_day_model(energy, hdd, cdd)
        
        return {
            'heating': heating_analysis,
            'cooling': cooling_analysis,
            'combined': combined_analysis,
            'heating_base_temp': self.heating_base_temp,
            'cooling_base_temp': self.cooling_base_temp
        }
    
    def _fit_degree_day_model(self, energy: np.ndarray, degree_days: np.ndarray, 
                             model_type: str) -> Dict[str, Any]:
        """Fit linear model: energy = base + coefficient * degree_days."""
        # Only use data where degree days > 0
        mask = degree_days > 0
        
        if np.sum(mask) < 10:  # Not enough data
            return {
                'coefficient': 0.0,
                'intercept': np.mean(energy),
                'r_squared': 0.0,
                'data_points': 0
            }
        
        dd_subset = degree_days[mask]
        energy_subset = energy[mask]
        
        if HAS_SKLEARN:
            # Use sklearn for regression
            model = LinearRegression()
            X = dd_subset.reshape(-1, 1)
            model.fit(X, energy_subset)
            
            coefficient = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(X, energy_subset)
        else:
            # Simple linear regression using numpy
            A = np.vstack([dd_subset, np.ones(len(dd_subset))]).T
            coefficient, intercept = np.linalg.lstsq(A, energy_subset, rcond=None)[0]
            
            # Calculate R²
            predictions = coefficient * dd_subset + intercept
            ss_res = np.sum((energy_subset - predictions) ** 2)
            ss_tot = np.sum((energy_subset - np.mean(energy_subset)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'coefficient': float(coefficient),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'data_points': int(np.sum(mask))
        }
    
    def _fit_combined_degree_day_model(self, energy: np.ndarray, hdd: np.ndarray, 
                                     cdd: np.ndarray) -> Dict[str, Any]:
        """Fit combined model: energy = base + heating_coeff*HDD + cooling_coeff*CDD."""
        if HAS_SKLEARN:
            # Prepare feature matrix
            X = np.column_stack([hdd, cdd])
            
            # Fit model
            model = LinearRegression()
            model.fit(X, energy)
            
            heating_coeff = model.coef_[0]
            cooling_coeff = model.coef_[1]
            intercept = model.intercept_
            r_squared = model.score(X, energy)
        else:
            # Simple multiple regression
            A = np.column_stack([hdd, cdd, np.ones(len(energy))])
            coeffs = np.linalg.lstsq(A, energy, rcond=None)[0]
            
            heating_coeff = coeffs[0]
            cooling_coeff = coeffs[1]
            intercept = coeffs[2]
            
            # Calculate R²
            predictions = heating_coeff * hdd + cooling_coeff * cdd + intercept
            ss_res = np.sum((energy - predictions) ** 2)
            ss_tot = np.sum((energy - np.mean(energy)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'heating_coefficient': float(heating_coeff),
            'cooling_coefficient': float(cooling_coeff),
            'base_load': float(intercept),
            'r_squared': float(r_squared),
            'total_explained_variance': float(r_squared)
        }
    
    def _decompose_energy_loads(self, energy: np.ndarray, temperature: np.ndarray,
                              data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Decompose energy into components: base, heating, cooling, residual."""
        # Calculate degree days
        hdd = np.maximum(0, self.heating_base_temp - temperature)
        cdd = np.maximum(0, temperature - self.cooling_base_temp)
        
        # Fit combined model
        combined_model = self._fit_combined_degree_day_model(energy, hdd, cdd)
        
        # Decompose energy
        base_load = np.full_like(energy, combined_model['base_load'])
        heating_load = combined_model['heating_coefficient'] * hdd
        cooling_load = combined_model['cooling_coefficient'] * cdd
        
        # Calculate residual
        predicted_total = base_load + heating_load + cooling_load
        residual = energy - predicted_total
        
        return {
            'base_load': base_load,
            'heating_load': heating_load,
            'cooling_load': cooling_load,
            'predicted_total': predicted_total,
            'residual': residual,
            'actual_total': energy
        }
    
    def _analyze_heating_signature(self, energy: np.ndarray, temperature: np.ndarray,
                                 data: pd.DataFrame) -> Optional[WeatherEnergySignature]:
        """Analyze heating energy signature."""
        hdd = np.maximum(0, self.heating_base_temp - temperature)
        heating_model = self._fit_degree_day_model(energy, hdd, "heating")
        
        if heating_model['r_squared'] < self.min_correlation_threshold:
            return None
        
        # Generate time-of-day factors (simplified)
        time_factors = self._extract_time_of_day_factors(data, 'heating')
        
        return WeatherEnergySignature(
            signature_type='heating',
            temperature_threshold=self.heating_base_temp,
            coefficient=heating_model['coefficient'],
            intercept=heating_model['intercept'],
            r_squared=heating_model['r_squared'],
            temperature_range=(np.min(temperature), self.heating_base_temp),
            seasonal_factor=1.0,  # Could be enhanced with seasonal analysis
            time_of_day_factor=time_factors
        )
    
    def _analyze_cooling_signature(self, energy: np.ndarray, temperature: np.ndarray,
                                 data: pd.DataFrame) -> Optional[WeatherEnergySignature]:
        """Analyze cooling energy signature."""
        cdd = np.maximum(0, temperature - self.cooling_base_temp)
        cooling_model = self._fit_degree_day_model(energy, cdd, "cooling")
        
        if cooling_model['r_squared'] < self.min_correlation_threshold:
            return None
        
        # Generate time-of-day factors (simplified)
        time_factors = self._extract_time_of_day_factors(data, 'cooling')
        
        return WeatherEnergySignature(
            signature_type='cooling',
            temperature_threshold=self.cooling_base_temp,
            coefficient=cooling_model['coefficient'],
            intercept=cooling_model['intercept'],
            r_squared=cooling_model['r_squared'],
            temperature_range=(self.cooling_base_temp, np.max(temperature)),
            seasonal_factor=1.0,  # Could be enhanced with seasonal analysis
            time_of_day_factor=time_factors
        )
    
    def _analyze_baseload_signature(self, energy: np.ndarray, temperature: np.ndarray,
                                  data: pd.DataFrame) -> WeatherEnergySignature:
        """Analyze baseload energy signature."""
        # Baseload is temperature-independent component
        base_energy = np.percentile(energy, 10)  # 10th percentile as baseload estimate
        
        # Time-of-day factors for baseload
        time_factors = self._extract_time_of_day_factors(data, 'baseload')
        
        return WeatherEnergySignature(
            signature_type='baseload',
            temperature_threshold=20.0,  # Neutral temperature
            coefficient=0.0,  # No temperature dependence
            intercept=base_energy,
            r_squared=1.0,  # Perfect fit for constant baseload
            temperature_range=(np.min(temperature), np.max(temperature)),
            seasonal_factor=1.0,
            time_of_day_factor=time_factors
        )
    
    def _extract_time_of_day_factors(self, data: pd.DataFrame, load_type: str) -> List[float]:
        """Extract time-of-day factors for different load types."""
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        
        # Find energy column
        energy_col = None
        for col in ['Value', 'value', 'energy', 'Energy']:
            if col in data.columns:
                energy_col = col
                break
        
        if energy_col is None:
            return [1.0] * 24  # Default flat pattern
        
        # Calculate hourly average energy
        hourly_avg = data.groupby('hour')[energy_col].mean()
        
        # Ensure all 24 hours are present
        time_factors = []
        for hour in range(24):
            if hour in hourly_avg.index:
                time_factors.append(hourly_avg[hour])
            else:
                time_factors.append(np.mean(hourly_avg.values))
        
        # Normalize to mean = 1.0
        time_factors = np.array(time_factors)
        if np.mean(time_factors) > 0:
            time_factors = time_factors / np.mean(time_factors)
        
        return time_factors.tolist()
    
    def _estimate_weather_independent_load(self, energy: np.ndarray, 
                                         temperature: np.ndarray) -> float:
        """Estimate the weather-independent portion of energy load."""
        # Use the minimum energy consumption as a proxy for weather-independent load
        # This represents the base building systems that run regardless of weather
        
        # Take 5th percentile to avoid outliers
        weather_independent = float(np.percentile(energy, 5))
        
        return weather_independent
    
    def _analyze_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Analyze seasonal patterns in energy consumption."""
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['month'] = data['timestamp'].dt.month
        
        # Find energy column
        energy_col = None
        for col in ['Value', 'value', 'energy', 'Energy']:
            if col in data.columns:
                energy_col = col
                break
        
        if energy_col is None:
            # Return default patterns
            return {
                'total_energy': [1.0] * 12,
                'heating_component': [1.0] * 12,
                'cooling_component': [1.0] * 12,
                'baseload_component': [1.0] * 12
            }
        
        # Monthly average energy
        monthly_avg = data.groupby('month')[energy_col].mean()
        
        # Ensure all 12 months
        total_pattern = []
        for month in range(1, 13):
            if month in monthly_avg.index:
                total_pattern.append(monthly_avg[month])
            else:
                total_pattern.append(np.mean(monthly_avg.values))
        
        # Normalize patterns
        total_pattern = np.array(total_pattern)
        if np.mean(total_pattern) > 0:
            total_pattern = total_pattern / np.mean(total_pattern)
        
        # Simplified seasonal components (could be enhanced)
        # Heating: higher in winter months
        heating_pattern = np.array([1.5, 1.4, 1.2, 0.8, 0.5, 0.3, 0.3, 0.3, 0.5, 0.8, 1.2, 1.4])
        
        # Cooling: higher in summer months
        cooling_pattern = np.array([0.3, 0.3, 0.5, 0.8, 1.2, 1.5, 1.8, 1.7, 1.3, 0.8, 0.5, 0.3])
        
        # Baseload: relatively constant
        baseload_pattern = np.ones(12)
        
        return {
            'total_energy': total_pattern.tolist(),
            'heating_component': heating_pattern.tolist(),
            'cooling_component': cooling_pattern.tolist(),
            'baseload_component': baseload_pattern.tolist()
        }
    
    def _log_analysis_summary(self) -> None:
        """Log summary of analysis results."""
        if self.analysis_results is None:
            return
        
        results = self.analysis_results
        
        self.logger.info("📋 Weather-Energy Analysis Summary:")
        self.logger.info(f"   🌡️ Temperature range: {results.temperature_statistics['min']:.1f}°C to {results.temperature_statistics['max']:.1f}°C")
        self.logger.info(f"   ⚡ Energy range: {results.energy_statistics['min']:.1f} to {results.energy_statistics['max']:.1f} kW")
        self.logger.info(f"   📊 Linear correlation: {results.correlation_analysis['linear_correlation']:.3f}")
        
        if results.heating_signature:
            self.logger.info(f"   🔥 Heating signature: {results.heating_signature.coefficient:.2f} kW/°C (R²={results.heating_signature.r_squared:.3f})")
        else:
            self.logger.info("   🔥 No significant heating signature found")
            
        if results.cooling_signature:
            self.logger.info(f"   ❄️ Cooling signature: {results.cooling_signature.coefficient:.2f} kW/°C (R²={results.cooling_signature.r_squared:.3f})")
        else:
            self.logger.info("   ❄️ No significant cooling signature found")
            
        self.logger.info(f"   🏠 Weather-independent load: {results.weather_independent_load:.1f} kW")
    
    def plot_analysis_results(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of weather-energy analysis."""
        if self.analysis_results is None or self.raw_data is None:
            self.logger.warning("⚠️ No analysis results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Weather-Energy Relationship Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        energy = self._extract_energy_values(self.raw_data)
        temperature = self._extract_temperature_values(self.raw_data)
        hdd = np.maximum(0, self.heating_base_temp - temperature)
        cdd = np.maximum(0, temperature - self.cooling_base_temp)
        
        # Plot 1: Energy vs Temperature scatter
        axes[0, 0].scatter(temperature, energy, alpha=0.6, s=1)
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Energy (kW)')
        axes[0, 0].set_title('Energy vs Temperature')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Heating Degree Days
        if np.sum(hdd) > 0:
            mask = hdd > 0
            axes[0, 1].scatter(hdd[mask], energy[mask], alpha=0.6, s=1, color='red')
            axes[0, 1].set_xlabel('Heating Degree Days')
            axes[0, 1].set_ylabel('Energy (kW)')
            axes[0, 1].set_title(f'Heating Response (Base: {self.heating_base_temp}°C)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cooling Degree Days
        if np.sum(cdd) > 0:
            mask = cdd > 0
            axes[0, 2].scatter(cdd[mask], energy[mask], alpha=0.6, s=1, color='blue')
            axes[0, 2].set_xlabel('Cooling Degree Days')
            axes[0, 2].set_ylabel('Energy (kW)')
            axes[0, 2].set_title(f'Cooling Response (Base: {self.cooling_base_temp}°C)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Load Decomposition
        if 'base_load' in self.analysis_results.load_decomposition:
            decomp = self.analysis_results.load_decomposition
            time_indices = np.arange(len(energy))[:1000]  # Show first 1000 points
            
            axes[1, 0].plot(time_indices, energy[:1000], label='Actual', alpha=0.8)
            axes[1, 0].plot(time_indices, decomp['predicted_total'][:1000], label='Predicted', alpha=0.8)
            axes[1, 0].set_xlabel('Time Index')
            axes[1, 0].set_ylabel('Energy (kW)')
            axes[1, 0].set_title('Load Decomposition')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Seasonal Pattern
        months = list(range(1, 13))
        seasonal = self.analysis_results.seasonal_patterns['total_energy']
        axes[1, 1].plot(months, seasonal, marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Relative Energy')
        axes[1, 1].set_title('Seasonal Energy Pattern')
        axes[1, 1].set_xticks(months)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Correlation Summary
        corr_data = self.analysis_results.correlation_analysis
        correlations = [
            corr_data['linear_correlation'],
            corr_data['heating_degree_day_correlation'],
            corr_data['cooling_degree_day_correlation'],
            corr_data['abs_temp_diff_correlation']
        ]
        labels = ['Linear', 'Heating DD', 'Cooling DD', 'Abs Temp Diff']
        
        bars = axes[1, 2].bar(labels, correlations, color=['gray', 'red', 'blue', 'orange'])
        axes[1, 2].set_ylabel('Correlation Coefficient')
        axes[1, 2].set_title('Temperature Correlations')
        axes[1, 2].set_ylim(-1, 1)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, correlations):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Analysis plots saved to {save_path}")
        
        plt.show()
    
    def export_analysis_report(self, filepath: str) -> None:
        """Export detailed analysis report to file."""
        if self.analysis_results is None:
            self.logger.warning("⚠️ No analysis results to export")
            return
        
        import json
        
        # Prepare export data
        export_data = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'heating_base_temp': self.heating_base_temp,
                'cooling_base_temp': self.cooling_base_temp
            },
            'temperature_statistics': self.analysis_results.temperature_statistics,
            'energy_statistics': self.analysis_results.energy_statistics,
            'correlation_analysis': self.analysis_results.correlation_analysis,
            'degree_day_analysis': self.analysis_results.degree_day_analysis,
            'weather_independent_load': self.analysis_results.weather_independent_load,
            'seasonal_patterns': self.analysis_results.seasonal_patterns,
            'signatures': {
                'heating': {
                    'found': self.analysis_results.heating_signature is not None,
                    'signature': self.analysis_results.heating_signature.__dict__ if self.analysis_results.heating_signature else None
                },
                'cooling': {
                    'found': self.analysis_results.cooling_signature is not None,
                    'signature': self.analysis_results.cooling_signature.__dict__ if self.analysis_results.cooling_signature else None
                },
                'baseload': self.analysis_results.baseload_signature.__dict__
            }
        }
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"📄 Analysis report exported to {filepath}")