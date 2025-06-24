"""
Building-Agnostic Energy Model
==============================

This module provides a comprehensive building energy model that can be trained
on any building type and used for energy disaggregation and forecasting.
It integrates weather-energy relationships, device energy models, and temporal
patterns to create accurate building energy profiles.

Key Features:
- Building-agnostic design (works for any building type)
- Integrated weather-energy analysis
- Device-level energy modeling
- Energy balance constraints
- Forecasting capabilities for future periods
- Model persistence and loading

This is the high-level interface for the energy disaggregation system.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import copy
import warnings
warnings.filterwarnings('ignore')

# Import our energy disaggregation components
from energy_disaggregator import EnergyDisaggregator, EnergyDisaggregationResult
from weather_energy_analyzer import WeatherEnergyAnalyzer, WeatherEnergyAnalysisResult


@dataclass
class BuildingProfile:
    """Profile information for a building."""
    building_type: str                     # 'office', 'university', 'residential', etc.
    total_floor_area: float               # m² 
    occupancy_type: str                   # 'commercial', 'institutional', 'residential'
    construction_year: Optional[int]      # Year built
    climate_zone: Optional[str]           # Climate classification
    heating_system_type: Optional[str]    # 'gas', 'electric', 'heat_pump', etc.
    cooling_system_type: Optional[str]    # 'central_ac', 'heat_pump', 'none', etc.
    insulation_rating: Optional[float]    # 0-1 scale
    equipment_density: Optional[float]    # W/m² typical equipment load


@dataclass
class ModelTrainingConfig:
    """Configuration for model training."""
    training_years: List[int]             # Years to use for training
    validation_years: List[int]           # Years to use for validation
    energy_balance_tolerance: float      # Acceptable energy balance error (%)
    min_correlation_threshold: float     # Minimum weather correlation for significance
    cross_validation_folds: int          # Number of CV folds
    feature_engineering: bool            # Whether to apply advanced feature engineering
    device_learning_method: str          # 'signature_based', 'pattern_based', 'hybrid'


@dataclass
class ForecastConfig:
    """Configuration for energy forecasting."""
    forecast_years: List[int]             # Years to forecast
    weather_scenario: str                 # 'historical', 'climate_normal', 'custom'
    uncertainty_estimation: bool         # Whether to estimate forecast uncertainty
    seasonal_adjustment: bool            # Whether to apply seasonal adjustments
    trend_extrapolation: bool            # Whether to extrapolate long-term trends


class BuildingEnergyModel:
    """
    High-level building energy model for disaggregation and forecasting.
    
    This class provides a complete solution for:
    1. Training on historical building energy + weather data
    2. Disaggregating total energy into device-level profiles
    3. Forecasting future energy consumption
    4. Validating model performance
    
    The model is building-agnostic and can be applied to any building type.
    """
    
    def __init__(self, building_profile: Optional[BuildingProfile] = None,
                 config_manager=None, accelerator=None, logger=None):
        self.building_profile = building_profile
        self.config_manager = config_manager
        self.accelerator = accelerator
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize core components
        self.energy_disaggregator = EnergyDisaggregator(config_manager, accelerator, logger)
        self.weather_analyzer = WeatherEnergyAnalyzer(logger)
        
        # Model state
        self.is_trained = False
        self.training_config: Optional[ModelTrainingConfig] = None
        self.weather_analysis: Optional[WeatherEnergyAnalysisResult] = None
        self.validation_results: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        
        # Performance metrics
        self.training_metrics: Dict[str, float] = {}
        self.validation_metrics: Dict[str, float] = {}
        
        self.logger.info("🏢 Building Energy Model initialized")
    
    def train(self, energy_data: pd.DataFrame, weather_data: pd.DataFrame,
              devices_config: Dict, training_config: ModelTrainingConfig) -> Dict[str, Any]:
        """
        Train the building energy model on historical data.
        
        Args:
            energy_data: Historical total energy consumption data
            weather_data: Historical weather data
            devices_config: Device configuration dictionary
            training_config: Training configuration parameters
            
        Returns:
            Dictionary with training results and performance metrics
        """
        self.logger.info("🎯 Starting building energy model training")
        self.training_config = training_config
        
        # Store training metadata
        self.model_metadata = {
            'training_start_time': datetime.now().isoformat(),
            'training_years': training_config.training_years,
            'validation_years': training_config.validation_years,
            'building_profile': asdict(self.building_profile) if self.building_profile else None,
            'model_version': '1.0.0'
        }
        
        # Step 1: Weather-Energy Relationship Analysis
        self.logger.info("🌡️ Analyzing weather-energy relationships")
        self.weather_analysis = self.weather_analyzer.analyze_weather_energy_relationship(
            energy_data, weather_data
        )
        
        # Step 2: Initialize device models from configuration
        self.logger.info("🔧 Initializing device energy models")
        self.energy_disaggregator.initialize_device_models(devices_config)
        
        # Step 3: Train energy disaggregation model
        self.logger.info("⚡ Training energy disaggregation model")
        disaggregation_results = self.energy_disaggregator.train(
            energy_data, weather_data, training_config.training_years
        )
        
        # Set trained flag after core training is complete
        self.training_metrics = disaggregation_results['validation_metrics']
        self.is_trained = True
        
        # Step 4: Validation on held-out data
        if training_config.validation_years:
            self.logger.info("✅ Validating model on held-out years")
            validation_results = self._validate_model(
                energy_data, weather_data, training_config.validation_years
            )
            self.validation_results = validation_results
        
        # Step 5: Cross-validation if requested
        if training_config.cross_validation_folds > 1:
            self.logger.info(f"🔄 Performing {training_config.cross_validation_folds}-fold cross-validation")
            cv_results = self._perform_cross_validation(
                energy_data, weather_data, training_config
            )
            self.validation_results['cross_validation'] = cv_results
        
        training_results = {
            'weather_analysis': self.weather_analysis,
            'disaggregation_results': disaggregation_results,
            'validation_results': self.validation_results,
            'training_metrics': self.training_metrics,
            'model_metadata': self.model_metadata
        }
        
        self.logger.info("✅ Building energy model training completed")
        self._log_training_summary()
        
        return training_results
    
    def disaggregate_energy(self, total_energy: Union[pd.DataFrame, np.ndarray],
                          weather_data: pd.DataFrame,
                          time_data: Optional[pd.DataFrame] = None) -> EnergyDisaggregationResult:
        """
        Disaggregate total energy consumption into device-level profiles.
        
        Args:
            total_energy: Total energy consumption data
            weather_data: Weather data for the same period
            time_data: Time data (optional, will be extracted from other sources)
            
        Returns:
            EnergyDisaggregationResult with device profiles and metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before disaggregation")
        
        self.logger.info("⚡ Disaggregating energy into device profiles")
        
        # Convert total_energy to numpy array if needed
        if isinstance(total_energy, pd.DataFrame):
            # Find energy column
            energy_cols = ['Value', 'value', 'energy', 'Energy', 'total_energy']
            energy_col = None
            for col in energy_cols:
                if col in total_energy.columns:
                    energy_col = col
                    break
            
            if energy_col is None:
                raise ValueError("No energy column found in total_energy DataFrame")
            
            energy_array = total_energy[energy_col].values
            
            # Use timestamp from energy data if time_data not provided
            if time_data is None:
                time_data = total_energy[['Timestamp']] if 'Timestamp' in total_energy.columns else total_energy
        else:
            energy_array = np.array(total_energy)
            
            # Create time_data if not provided
            if time_data is None:
                time_data = pd.DataFrame({
                    'timestamp': pd.date_range(start='2024-01-01', periods=len(energy_array), freq='15min')
                })
        
        # Perform disaggregation
        result = self.energy_disaggregator.disaggregate(
            energy_array, weather_data, time_data
        )
        
        self.logger.info(f"✅ Disaggregation completed with {result.energy_balance_error:.3f}% energy balance error")
        return result
    
    def forecast_energy(self, forecast_weather: pd.DataFrame,
                       forecast_config: ForecastConfig) -> Dict[str, Any]:
        """
        Forecast energy consumption for future periods.
        
        Args:
            forecast_weather: Weather data for forecast period
            forecast_config: Forecasting configuration
            
        Returns:
            Dictionary with forecasted energy profiles and uncertainty estimates
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        self.logger.info(f"🔮 Forecasting energy for years {forecast_config.forecast_years}")
        
        # Generate forecast time series
        forecast_time_data = self._generate_forecast_time_data(forecast_config.forecast_years)
        
        # Extract temperature and create total energy estimates
        temperature = self._extract_temperature_from_weather(forecast_weather)
        
        # Generate total energy forecast using weather relationships
        total_energy_forecast = self._forecast_total_energy(
            temperature, forecast_time_data, forecast_config
        )
        
        # Disaggregate forecasted total energy
        forecast_result = self.disaggregate_energy(
            total_energy_forecast, forecast_weather, forecast_time_data
        )
        
        # Add uncertainty estimation if requested
        uncertainty_estimates = None
        if forecast_config.uncertainty_estimation:
            uncertainty_estimates = self._estimate_forecast_uncertainty(
                forecast_result, forecast_config
            )
        
        forecast_output = {
            'forecast_years': forecast_config.forecast_years,
            'total_energy_forecast': total_energy_forecast,
            'device_profiles_forecast': forecast_result.device_profiles,
            'forecast_metrics': {
                'total_energy_mean': float(np.mean(total_energy_forecast)),
                'total_energy_peak': float(np.max(total_energy_forecast)),
                'total_energy_min': float(np.min(total_energy_forecast)),
                'energy_balance_error': forecast_result.energy_balance_error
            },
            'uncertainty_estimates': uncertainty_estimates,
            'forecast_metadata': {
                'forecast_date': datetime.now().isoformat(),
                'weather_scenario': forecast_config.weather_scenario,
                'model_version': self.model_metadata.get('model_version', '1.0.0')
            }
        }
        
        self.logger.info("✅ Energy forecasting completed")
        return forecast_output
    
    def _validate_model(self, energy_data: pd.DataFrame, weather_data: pd.DataFrame,
                       validation_years: List[int]) -> Dict[str, Any]:
        """Validate model performance on held-out data."""
        # Filter data for validation years
        validation_energy = self._filter_data_by_years(energy_data, validation_years)
        validation_weather = self._filter_data_by_years(weather_data, validation_years)
        
        if len(validation_energy) == 0:
            self.logger.warning("⚠️ No validation data found")
            return {}
        
        # Perform disaggregation on validation data
        validation_result = self.disaggregate_energy(
            validation_energy, validation_weather
        )
        
        # Calculate additional validation metrics
        validation_metrics = self._calculate_comprehensive_validation_metrics(
            validation_result
        )
        
        return {
            'validation_years': validation_years,
            'validation_data_points': len(validation_energy),
            'energy_balance_error': validation_result.energy_balance_error,
            'validation_metrics': validation_metrics,
            'device_allocations': validation_result.allocation_summary
        }
    
    def _perform_cross_validation(self, energy_data: pd.DataFrame, 
                                weather_data: pd.DataFrame,
                                training_config: ModelTrainingConfig) -> Dict[str, Any]:
        """Perform k-fold cross-validation."""
        all_years = sorted(set(training_config.training_years + training_config.validation_years))
        n_folds = training_config.cross_validation_folds
        
        # Create year folds
        year_folds = np.array_split(all_years, n_folds)
        
        cv_results = []
        
        for fold_idx, test_years in enumerate(year_folds):
            train_years = [year for year in all_years if year not in test_years]
            
            self.logger.info(f"📊 CV Fold {fold_idx + 1}: Train on {train_years}, Test on {list(test_years)}")
            
            # Create temporary disaggregator for this fold
            temp_disaggregator = EnergyDisaggregator(
                self.config_manager, self.accelerator, self.logger
            )
            temp_disaggregator.device_models = self.energy_disaggregator.device_models.copy()
            
            # Train on fold training data
            temp_disaggregator.train(energy_data, weather_data, train_years)
            
            # Test on fold test data
            test_energy = self._filter_data_by_years(energy_data, list(test_years))
            test_weather = self._filter_data_by_years(weather_data, list(test_years))
            
            if len(test_energy) > 0:
                test_result = temp_disaggregator.disaggregate(
                    test_energy['Value'].values if 'Value' in test_energy.columns else test_energy.iloc[:, 1].values,
                    test_weather,
                    test_energy
                )
                
                cv_results.append({
                    'fold': fold_idx + 1,
                    'train_years': train_years,
                    'test_years': list(test_years),
                    'energy_balance_error': test_result.energy_balance_error,
                    'test_data_points': len(test_energy),
                    'validation_metrics': test_result.validation_metrics
                })
        
        # Calculate cross-validation summary
        cv_summary = self._summarize_cross_validation(cv_results)
        
        return {
            'fold_results': cv_results,
            'cv_summary': cv_summary,
            'n_folds': n_folds
        }
    
    def _summarize_cross_validation(self, cv_results: List[Dict]) -> Dict[str, float]:
        """Summarize cross-validation results."""
        if not cv_results:
            return {}
        
        # Extract metrics from all folds
        energy_balance_errors = [result['energy_balance_error'] for result in cv_results]
        r2_scores = [result['validation_metrics']['r2_score'] for result in cv_results]
        mae_values = [result['validation_metrics']['mae'] for result in cv_results]
        
        return {
            'mean_energy_balance_error': float(np.mean(energy_balance_errors)),
            'std_energy_balance_error': float(np.std(energy_balance_errors)),
            'mean_r2_score': float(np.mean(r2_scores)),
            'std_r2_score': float(np.std(r2_scores)),
            'mean_mae': float(np.mean(mae_values)),
            'std_mae': float(np.std(mae_values))
        }
    
    def _filter_data_by_years(self, data: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """Filter DataFrame to include only specified years."""
        # Find timestamp column
        ts_cols = ['timestamp', 'Timestamp', 'time', 'Time']
        ts_col = None
        for col in ts_cols:
            if col in data.columns:
                ts_col = col
                break
        
        if ts_col is None:
            # Use index if no timestamp column
            data_copy = data.copy()
            data_copy['timestamp'] = data_copy.index
            ts_col = 'timestamp'
        else:
            data_copy = data.copy()
            
        data_copy[ts_col] = pd.to_datetime(data_copy[ts_col])
        year_mask = data_copy[ts_col].dt.year.isin(years)
        
        return data_copy[year_mask]
    
    def _calculate_comprehensive_validation_metrics(self, result: EnergyDisaggregationResult) -> Dict[str, float]:
        """Calculate comprehensive validation metrics."""
        predicted = result.total_predicted
        actual = result.total_actual
        
        # Basic metrics
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        mape = np.mean(np.abs((predicted - actual) / actual)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Peak prediction accuracy
        actual_peak = np.max(actual)
        predicted_peak = np.max(predicted)
        peak_error = abs(predicted_peak - actual_peak) / actual_peak * 100
        
        # Energy conservation check
        total_actual_energy = np.sum(actual)
        total_predicted_energy = np.sum(predicted)
        energy_conservation_error = abs(total_predicted_energy - total_actual_energy) / total_actual_energy * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2),
            'peak_prediction_error': float(peak_error),
            'energy_conservation_error': float(energy_conservation_error),
            'mean_actual': float(np.mean(actual)),
            'mean_predicted': float(np.mean(predicted))
        }
    
    def _generate_forecast_time_data(self, forecast_years: List[int]) -> pd.DataFrame:
        """Generate time data for forecast period."""
        start_date = f"{min(forecast_years)}-01-01"
        end_date = f"{max(forecast_years)}-12-31 23:45:00"
        
        # Generate 15-minute intervals for forecast period
        time_range = pd.date_range(start=start_date, end=end_date, freq='15min')
        
        return pd.DataFrame({'timestamp': time_range})
    
    def _extract_temperature_from_weather(self, weather_data: pd.DataFrame) -> np.ndarray:
        """Extract temperature array from weather data."""
        temp_cols = ['temperature', 'Temperature', 'temp', 'Temp']
        
        for col in temp_cols:
            if col in weather_data.columns:
                return weather_data[col].values
        
        # Default temperature if not found
        self.logger.warning("⚠️ No temperature data found in forecast weather, using default 20°C")
        return np.full(len(weather_data), 20.0)
    
    def _forecast_total_energy(self, temperature: np.ndarray, time_data: pd.DataFrame,
                             forecast_config: ForecastConfig) -> np.ndarray:
        """Forecast total energy consumption using learned weather relationships."""
        # Use learned parameters from weather analysis
        if self.weather_analysis is None:
            raise ValueError("Weather analysis not available for forecasting")
        
        # Extract learned parameters
        base_load = self.weather_analysis.weather_independent_load
        degree_day_params = self.weather_analysis.degree_day_analysis.get('combined', {})
        
        heating_coeff = degree_day_params.get('heating_coefficient', 0)
        cooling_coeff = degree_day_params.get('cooling_coefficient', 0)
        
        # Calculate degree days
        heating_base = self.weather_analyzer.heating_base_temp
        cooling_base = self.weather_analyzer.cooling_base_temp
        
        hdd = np.maximum(0, heating_base - temperature)
        cdd = np.maximum(0, temperature - cooling_base)
        
        # Base forecast
        total_energy = base_load + heating_coeff * hdd + cooling_coeff * cdd
        
        # Apply time-of-day patterns
        if hasattr(self.energy_disaggregator, 'learned_parameters'):
            time_patterns = self.energy_disaggregator.learned_parameters.get('temporal_patterns', {})
            time_features = time_patterns.get('time_features', {})
            
            if 'interval_pattern_96' in time_features:
                pattern_96 = np.array(time_features['interval_pattern_96'])
                
                # Map to time data
                time_data['timestamp'] = pd.to_datetime(time_data['timestamp'])
                time_data['interval_15min'] = (time_data['timestamp'].dt.hour * 4 + 
                                             time_data['timestamp'].dt.minute // 15)
                
                time_factors = []
                for interval in time_data['interval_15min']:
                    if 0 <= interval < 96:
                        time_factors.append(pattern_96[interval])
                    else:
                        time_factors.append(1.0)
                
                time_factors = np.array(time_factors)
                total_energy = total_energy * time_factors
        
        # Apply seasonal adjustments if requested
        if forecast_config.seasonal_adjustment:
            total_energy = self._apply_seasonal_adjustments(total_energy, time_data)
        
        # Ensure positive energy values
        total_energy = np.maximum(total_energy, base_load * 0.5)
        
        return total_energy
    
    def _apply_seasonal_adjustments(self, energy: np.ndarray, time_data: pd.DataFrame) -> np.ndarray:
        """Apply seasonal adjustments to energy forecast."""
        if self.weather_analysis is None:
            return energy
        
        seasonal_patterns = self.weather_analysis.seasonal_patterns.get('total_energy', [1.0] * 12)
        
        time_data['month'] = pd.to_datetime(time_data['timestamp']).dt.month
        
        seasonal_factors = []
        for month in time_data['month']:
            if 1 <= month <= 12:
                seasonal_factors.append(seasonal_patterns[month - 1])
            else:
                seasonal_factors.append(1.0)
        
        seasonal_factors = np.array(seasonal_factors)
        return energy * seasonal_factors
    
    def _estimate_forecast_uncertainty(self, forecast_result: EnergyDisaggregationResult,
                                     forecast_config: ForecastConfig) -> Dict[str, Any]:
        """Estimate uncertainty in energy forecasts."""
        # Simple uncertainty estimation based on validation performance
        validation_metrics = self.validation_metrics
        
        if not validation_metrics:
            self.logger.warning("⚠️ No validation metrics available for uncertainty estimation")
            return {}
        
        # Use RMSE as uncertainty measure
        rmse = validation_metrics.get('rmse', 0)
        mae = validation_metrics.get('mae', 0)
        
        # Estimate confidence intervals (simple approach)
        total_forecast = forecast_result.total_predicted
        
        # ±1 standard deviation confidence intervals
        lower_bound = total_forecast - rmse
        upper_bound = total_forecast + rmse
        
        # ±2 standard deviation confidence intervals
        lower_bound_95 = total_forecast - 2 * rmse
        upper_bound_95 = total_forecast + 2 * rmse
        
        return {
            'forecast_rmse': float(rmse),
            'forecast_mae': float(mae),
            'confidence_intervals': {
                '68_percent': {
                    'lower_bound': lower_bound.tolist(),
                    'upper_bound': upper_bound.tolist()
                },
                '95_percent': {
                    'lower_bound': lower_bound_95.tolist(),
                    'upper_bound': upper_bound_95.tolist()
                }
            },
            'uncertainty_method': 'validation_rmse_based'
        }
    
    def _log_training_summary(self) -> None:
        """Log summary of training results."""
        if not self.is_trained:
            return
        
        self.logger.info("📋 Building Energy Model Training Summary:")
        
        if self.weather_analysis:
            self.logger.info(f"   🌡️ Weather correlation: {self.weather_analysis.correlation_analysis['linear_correlation']:.3f}")
            if self.weather_analysis.heating_signature:
                self.logger.info(f"   🔥 Heating signature: {self.weather_analysis.heating_signature.coefficient:.2f} kW/°C")
            if self.weather_analysis.cooling_signature:
                self.logger.info(f"   ❄️ Cooling signature: {self.weather_analysis.cooling_signature.coefficient:.2f} kW/°C")
        
        if self.training_metrics:
            self.logger.info(f"   ⚡ Training energy balance error: {self.training_metrics['energy_balance_error']:.3f}%")
            self.logger.info(f"   📊 Training R²: {self.training_metrics['r2_score']:.3f}")
        
        if self.validation_results and 'energy_balance_error' in self.validation_results:
            self.logger.info(f"   ✅ Validation energy balance error: {self.validation_results['energy_balance_error']:.3f}%")
    
    def save_model(self, filepath: str) -> None:
        """Save complete trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        import json
        import pickle
        
        # Prepare model data
        model_data = {
            'model_metadata': self.model_metadata,
            'building_profile': asdict(self.building_profile) if self.building_profile else None,
            'training_config': asdict(self.training_config) if self.training_config else None,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'validation_results': self.validation_results,
            'is_trained': self.is_trained
        }
        
        # Save disaggregator separately
        disaggregator_path = filepath.replace('.json', '_disaggregator.json')
        self.energy_disaggregator.save_model(disaggregator_path)
        
        # Save weather analysis
        weather_analysis_path = filepath.replace('.json', '_weather_analysis.json')
        if self.weather_analysis:
            # Convert weather analysis to dict for JSON serialization
            weather_dict = {
                'heating_signature': asdict(self.weather_analysis.heating_signature) if self.weather_analysis.heating_signature else None,
                'cooling_signature': asdict(self.weather_analysis.cooling_signature) if self.weather_analysis.cooling_signature else None,
                'baseload_signature': asdict(self.weather_analysis.baseload_signature),
                'weather_independent_load': self.weather_analysis.weather_independent_load,
                'seasonal_patterns': self.weather_analysis.seasonal_patterns,
                'temperature_statistics': self.weather_analysis.temperature_statistics,
                'energy_statistics': self.weather_analysis.energy_statistics,
                'correlation_analysis': self.weather_analysis.correlation_analysis,
                'degree_day_analysis': self.weather_analysis.degree_day_analysis
            }
            
            with open(weather_analysis_path, 'w') as f:
                json.dump(weather_dict, f, indent=2, default=str)
        
        # Save main model data
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Complete model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load complete trained model from file."""
        import json
        
        # Load main model data
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Restore model attributes
        self.model_metadata = model_data['model_metadata']
        self.training_metrics = model_data['training_metrics']
        self.validation_metrics = model_data['validation_metrics']
        self.validation_results = model_data['validation_results']
        self.is_trained = model_data['is_trained']
        
        # Restore building profile if present
        if model_data['building_profile']:
            self.building_profile = BuildingProfile(**model_data['building_profile'])
        
        # Restore training config if present
        if model_data['training_config']:
            self.training_config = ModelTrainingConfig(**model_data['training_config'])
        
        # Load disaggregator
        disaggregator_path = filepath.replace('.json', '_disaggregator.json')
        self.energy_disaggregator.load_model(disaggregator_path)
        
        # Load weather analysis
        weather_analysis_path = filepath.replace('.json', '_weather_analysis.json')
        try:
            with open(weather_analysis_path, 'r') as f:
                weather_dict = json.load(f)
            
            # Reconstruct weather analysis (simplified - would need full reconstruction)
            self.weather_analysis = weather_dict  # Store as dict for now
            
        except FileNotFoundError:
            self.logger.warning("⚠️ Weather analysis file not found")
            self.weather_analysis = None
        
        self.logger.info(f"📂 Complete model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        summary = {
            'status': 'trained',
            'model_metadata': self.model_metadata,
            'building_profile': asdict(self.building_profile) if self.building_profile else None,
            'performance_metrics': {
                'training': self.training_metrics,
                'validation': self.validation_metrics
            },
            'weather_analysis_summary': {},
            'device_count': len(self.energy_disaggregator.device_models),
            'device_names': list(self.energy_disaggregator.device_models.keys())
        }
        
        if self.weather_analysis:
            summary['weather_analysis_summary'] = {
                'heating_signature_found': self.weather_analysis.heating_signature is not None,
                'cooling_signature_found': self.weather_analysis.cooling_signature is not None,
                'weather_correlation': self.weather_analysis.correlation_analysis.get('linear_correlation', 0),
                'weather_independent_load': self.weather_analysis.weather_independent_load
            }
        
        return summary