#!/usr/bin/env python3
"""
Energy Load Profile Generator - Energy Balance Main Script
=========================================================

Main script for the energy disaggregation and forecasting system.
This system can train on historical building energy data, disaggregate
total energy into device-level profiles, and forecast future consumption.

Features:
- Train-Generate-Forecast workflow
- Energy balance constraints (sum of devices = total energy)
- Building-agnostic modeling
- Weather-based energy relationships
- GPU-accelerated training and generation
- Multi-scenario forecasting

Workflow:
1. Train: Learn energy patterns from historical data
2. Generate: Disaggregate total energy into device profiles
3. Forecast: Predict future energy consumption
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

# Import project modules
from config_manager import ConfigManager
from weather_fetcher import MultiSourceWeatherFetcher
from weather_database import WeatherDatabase
from analysis_export import LoadProfileAnalyzer

# Import new energy disaggregation components
from building_model import BuildingEnergyModel, BuildingProfile, ModelTrainingConfig, ForecastConfig
from energy_disaggregator import EnergyDisaggregator
from weather_energy_analyzer import WeatherEnergyAnalyzer
from forecast_engine import EnergyForecastEngine


def setup_logging(config: Dict) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})

    # Create logs directory
    log_file = log_config.get('file', 'energy_load_profile.log')
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_devices_config(config_manager: ConfigManager) -> Dict:
    """Load devices configuration."""
    devices_path = 'devices.json'
    if os.path.exists(devices_path):
        with open(devices_path, 'r') as f:
            devices_data = json.load(f)
        
        # Extract devices from the JSON structure
        if 'devices' in devices_data:
            return devices_data['devices']
        else:
            return devices_data
    else:
        # Fall back to config.yaml
        config = config_manager.load_config()
        return config.get('devices', {})


def create_building_profile(building_info: Dict) -> BuildingProfile:
    """Create building profile from configuration."""
    return BuildingProfile(
        building_type=building_info.get('type', 'generic'),
        total_floor_area=building_info.get('total_area_sqm', 10000),
        occupancy_type=building_info.get('occupancy_type', 'commercial'),
        construction_year=building_info.get('construction_year'),
        climate_zone=building_info.get('climate_zone'),
        heating_system_type=building_info.get('heating_system_type'),
        cooling_system_type=building_info.get('cooling_system_type'),
        insulation_rating=building_info.get('insulation_rating'),
        equipment_density=building_info.get('equipment_density')
    )


def train_model(args, config: Dict, logger: logging.Logger) -> BuildingEnergyModel:
    """Train the energy disaggregation model."""
    logger.info("🎯 Starting model training workflow")
    
    # Load training data
    if not os.path.exists(args.training_data):
        raise FileNotFoundError(f"Training data file not found: {args.training_data}")
    
    logger.info(f"📊 Loading training data from {args.training_data}")
    energy_data = pd.read_excel(args.training_data, sheet_name=None)
    
    # Combine all sheets or use specific years
    if isinstance(energy_data, dict):
        # Combine all sheets into single DataFrame
        combined_data = []
        for sheet_name, sheet_data in energy_data.items():
            if 'timestamp' in sheet_data.columns or 'Timestamp' in sheet_data.columns:
                combined_data.append(sheet_data)
        
        if combined_data:
            training_energy_data = pd.concat(combined_data, ignore_index=True)
        else:
            # Use first sheet
            training_energy_data = list(energy_data.values())[0]
    else:
        training_energy_data = energy_data
    
    logger.info(f"📈 Loaded {len(training_energy_data)} energy data points")
    
    # Load weather data
    config_manager = ConfigManager(args.config)
    weather_db = WeatherDatabase()
    weather_fetcher = MultiSourceWeatherFetcher(config_manager.load_config())
    
    # Extract date range from training data
    timestamp_col = 'Timestamp' if 'Timestamp' in training_energy_data.columns else 'timestamp'
    training_energy_data[timestamp_col] = pd.to_datetime(training_energy_data[timestamp_col])
    
    start_date = training_energy_data[timestamp_col].min()
    end_date = training_energy_data[timestamp_col].max()
    
    logger.info(f"🌡️ Fetching weather data for {start_date.date()} to {end_date.date()}")
    
    # Get weather data for training period
    weather_data = weather_fetcher.get_weather_data(
        args.location, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    )
    
    # Load devices configuration
    devices_config = load_devices_config(config_manager)
    logger.info(f"🔧 Loaded configuration for {len(devices_config)} devices")
    
    # Create building profile
    building_info = {}
    if os.path.exists('devices.json'):
        with open('devices.json', 'r') as f:
            devices_data = json.load(f)
            building_info = devices_data.get('building', {})
    
    building_profile = create_building_profile(building_info)
    
    # Initialize building energy model
    building_model = BuildingEnergyModel(
        building_profile=building_profile,
        config_manager=config_manager,
        logger=logger
    )
    
    # Configure training
    training_years = getattr(args, 'training_years', None)
    if training_years is None:
        # Extract years from data
        years = sorted(training_energy_data[timestamp_col].dt.year.unique())
        training_years = years[:-1] if len(years) > 1 else years  # Use all but last year for training
        validation_years = [years[-1]] if len(years) > 1 else []
    else:
        validation_years = getattr(args, 'validation_years', [])
    
    training_config = ModelTrainingConfig(
        training_years=training_years,
        validation_years=validation_years,
        energy_balance_tolerance=getattr(args, 'energy_balance_tolerance', 1.0),
        min_correlation_threshold=0.3,
        cross_validation_folds=getattr(args, 'cv_folds', 3),
        feature_engineering=True,
        device_learning_method='signature_based'
    )
    
    logger.info(f"📚 Training on years {training_years}, validating on {validation_years}")
    
    # Train the model
    training_results = building_model.train(
        training_energy_data, weather_data, devices_config, training_config
    )
    
    # Save trained model
    model_path = getattr(args, 'model_output', 'trained_model.json')
    building_model.save_model(model_path)
    logger.info(f"💾 Trained model saved to {model_path}")
    
    # Log training summary
    logger.info("✅ Model training completed successfully")
    logger.info(f"   📊 Training R²: {training_results['training_metrics']['r2_score']:.3f}")
    logger.info(f"   ⚡ Energy balance error: {training_results['training_metrics']['energy_balance_error']:.3f}%")
    
    if validation_years:
        validation_error = training_results['validation_results'].get('energy_balance_error', 0)
        logger.info(f"   ✅ Validation error: {validation_error:.3f}%")
    
    return building_model


def generate_profiles(args, config: Dict, logger: logging.Logger) -> Dict[str, Any]:
    """Generate comprehensive device energy profiles with visualizations for specified period."""
    logger.info("⚡ Starting enhanced energy profile generation")
    
    # Load trained model
    model_path = getattr(args, 'model_path', 'trained_model.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    
    # Initialize building model
    config_manager = ConfigManager(args.config)
    
    building_model = BuildingEnergyModel(
        config_manager=config_manager,
        logger=logger
    )
    
    # Load the trained model
    building_model.load_model(model_path)
    logger.info(f"📂 Loaded trained model from {model_path}")
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    logger.info(f"📅 Generating profiles for {start_date.date()} to {end_date.date()}")
    
    # Get weather data for generation period
    weather_fetcher = MultiSourceWeatherFetcher(config.get('weather', {}))
    weather_data = weather_fetcher.get_weather_data(args.location, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Determine energy data source
    total_energy_data = None
    time_data = None
    
    # Check if we have actual energy data for validation
    if hasattr(args, 'validation_data') and args.validation_data:
        if os.path.exists(args.validation_data):
            logger.info(f"📊 Loading validation data from {args.validation_data}")
            validation_data = pd.read_excel(args.validation_data, sheet_name=None)
            
            # Find the appropriate sheet for the generation year
            generation_year = start_date.year
            validation_energy = None
            
            if isinstance(validation_data, dict):
                for sheet_name, sheet_data in validation_data.items():
                    if str(generation_year) in sheet_name or len(validation_data) == 1:
                        validation_energy = sheet_data
                        break
            else:
                validation_energy = validation_data
            
            if validation_energy is not None:
                logger.info("🔬 Using actual energy data for generation")
                total_energy_data = validation_energy
            else:
                logger.warning("⚠️ No validation data found for generation year, using synthetic data")
        else:
            logger.warning(f"⚠️ Validation data file not found: {args.validation_data}, using synthetic data")
    
    # Generate synthetic data if no actual data available
    if total_energy_data is None:
        logger.info("🔮 Generating synthetic energy profiles using forecast")
        
        forecast_config = ForecastConfig(
            forecast_years=[start_date.year],
            weather_scenario='historical',
            uncertainty_estimation=False,
            seasonal_adjustment=True,
            trend_extrapolation=False
        )
        
        forecast_result = building_model.forecast_energy(weather_data, forecast_config)
        
        # Use forecasted total energy for disaggregation
        total_energy = forecast_result['total_energy_forecast']
        
        # Create time data matching weather data range
        # Use weather data timestamps to ensure exact match
        if 'Timestamp' in weather_data.columns:
            time_data = pd.DataFrame({'timestamp': weather_data['Timestamp'].copy()})
        elif weather_data.index.name == 'datetime' or hasattr(weather_data.index, 'dt'):
            time_data = pd.DataFrame({'timestamp': weather_data.index.copy()})
        else:
            # Fallback to generating time data
            time_data = pd.DataFrame({
                'timestamp': pd.date_range(start=start_date, end=end_date, freq='15min')
            })
        
        # Ensure length consistency
        min_length = min(len(total_energy), len(time_data))
        total_energy = total_energy[:min_length]
        time_data = time_data[:min_length]
        weather_data = weather_data[:min_length]
        
        total_energy_data = total_energy
    
    # Generate comprehensive profiles with visualizations
    generation_result = building_model.generate_energy_profiles(
        total_energy=total_energy_data,
        weather_data=weather_data,
        location=args.location,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        time_data=time_data,
        create_visualizations=True,
        export_data=True
    )
    
    if generation_result is None:
        raise RuntimeError("Failed to generate energy profiles")
    
    result = generation_result['disaggregation_result']
    logger.info(f"✅ Profile generation completed with {result.energy_balance_error:.6f}% energy balance error")
    
    # Log comprehensive results summary
    logger.info("📋 Generation Results Summary:")
    logger.info(f"   📍 Location: {generation_result['location']}")
    logger.info(f"   📅 Period: {generation_result['start_date']} to {generation_result['end_date']}")
    logger.info(f"   🔧 Device Count: {len(result.device_profiles)}")
    logger.info(f"   ⚡ Energy Balance Error: {result.energy_balance_error:.6f}%")
    
    if 'output_directory' in generation_result:
        logger.info(f"   📁 Output Directory: {generation_result['output_directory']}")
        logger.info(f"   📊 Visualizations: {'✅ Created' if generation_result.get('visualizations_created', False) else '❌ Not created'}")
        logger.info(f"   📋 Data Export: {'✅ Completed' if generation_result.get('data_exported', False) or generation_result.get('visualizations_created', False) else '❌ Failed'}")
    
    # Add device allocation summary to logs
    logger.info("🔧 Device Allocation Summary (Top 10):")
    top_devices = sorted(result.allocation_summary.items(), key=lambda x: x[1], reverse=True)[:10]
    for device, allocation in top_devices:
        logger.info(f"   - {device.replace('_', ' ').title()}: {allocation:.2f}%")
    
    return generation_result


def forecast_energy(args, config: Dict, logger: logging.Logger) -> Dict[str, Any]:
    """Forecast energy consumption for future periods."""
    logger.info("🔮 Starting energy forecasting workflow")
    
    # Load trained model
    model_path = getattr(args, 'model_path', 'trained_model.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    
    # Initialize components
    config_manager = ConfigManager(args.config)
    
    building_model = BuildingEnergyModel(
        config_manager=config_manager,
        logger=logger
    )
    
    building_model.load_model(model_path)
    logger.info(f"📂 Loaded trained model from {model_path}")
    
    # Initialize forecast engine
    forecast_engine = EnergyForecastEngine(building_model, logger)
    
    # Parse forecast parameters
    forecast_years = getattr(args, 'forecast_years', [2025])
    scenarios = getattr(args, 'scenarios', ['baseline', 'warm_climate', 'cold_climate'])
    
    logger.info(f"🎯 Forecasting for years {forecast_years} with scenarios {scenarios}")
    
    # Load historical weather for scenario generation
    if hasattr(args, 'historical_weather') and os.path.exists(args.historical_weather):
        historical_weather = pd.read_excel(args.historical_weather)
    else:
        # Use weather from training period as baseline
        start_date = datetime(min(forecast_years) - 5, 1, 1)  # 5 years of history
        end_date = datetime(min(forecast_years) - 1, 12, 31)
        
        weather_fetcher = MultiSourceWeatherFetcher(config.get('weather', {}))
        historical_weather = weather_fetcher.get_weather_data(args.location, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Generate forecasts
    forecast_results = forecast_engine.forecast_multiple_scenarios(
        historical_weather, forecast_years, scenarios, include_uncertainty=True
    )
    
    logger.info(f"✅ Generated forecasts for {len(forecast_results)} scenarios")
    
    # Export forecast results
    output_dir = getattr(args, 'output_dir', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    location_str = args.location.replace(' ', '_').replace(',', '').lower()
    
    # Export to JSON
    forecast_filename = f"energy_forecast_{location_str}_{'-'.join(map(str, forecast_years))}_{timestamp}.json"
    forecast_path = os.path.join(output_dir, forecast_filename)
    
    forecast_engine.export_forecast_results(forecast_results, forecast_path)
    
    # Create summary plots
    plot_filename = f"forecast_scenarios_{location_str}_{'-'.join(map(str, forecast_years))}_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    forecast_engine.plot_forecast_scenarios(forecast_results, plot_path)
    
    # Get forecast summary
    forecast_summary = forecast_engine.get_forecast_summary(forecast_results)
    logger.info("📊 Forecast Summary:")
    for scenario_name, scenario_data in forecast_summary['scenario_comparison'].items():
        total_energy = scenario_data['total_energy'] / 1000  # Convert to MWh
        logger.info(f"   {scenario_name}: {total_energy:.1f} MWh total, {scenario_data['peak_demand']:.1f} kW peak")
    
    return {
        'forecast_results': forecast_results,
        'forecast_summary': forecast_summary,
        'output_path': forecast_path,
        'plot_path': plot_path
    }


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Energy Load Profile Generator - Energy Balance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Train model on historical data
  python main.py train --training-data load_profiles.xlsx --location "Bottrop, Germany"
  
  # Generate device profiles for 2024 with validation
  python main.py generate --start-date 2024-01-01 --end-date 2024-12-31 \\
                          --location "Bottrop, Germany" --validation-data load_profiles.xlsx
  
  # Forecast energy for 2025
  python main.py forecast --forecast-years 2025 --location "Bottrop, Germany"
  
  # Forecast multiple scenarios
  python main.py forecast --forecast-years 2025 2026 --location "Bottrop, Germany" \\
                          --scenarios baseline warm_climate extreme_heat
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train energy disaggregation model')
    train_parser.add_argument('--training-data', required=True, help='Historical energy data (Excel file)')
    train_parser.add_argument('--location', required=True, help='Location for weather data')
    train_parser.add_argument('--training-years', nargs='+', type=int, help='Years to use for training')
    train_parser.add_argument('--validation-years', nargs='+', type=int, help='Years to use for validation')
    train_parser.add_argument('--model-output', default='trained_model.json', help='Output path for trained model')
    train_parser.add_argument('--energy-balance-tolerance', type=float, default=1.0, help='Energy balance tolerance (%)')
    train_parser.add_argument('--cv-folds', type=int, default=3, help='Cross-validation folds')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate device energy profiles')
    generate_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    generate_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    generate_parser.add_argument('--location', required=True, help='Location for weather data')
    generate_parser.add_argument('--model-path', default='trained_model.json', help='Path to trained model')
    generate_parser.add_argument('--validation-data', help='Actual energy data for validation (Excel file)')
    generate_parser.add_argument('--output-dir', default='output', help='Output directory')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Forecast future energy consumption')
    forecast_parser.add_argument('--forecast-years', nargs='+', type=int, required=True, help='Years to forecast')
    forecast_parser.add_argument('--location', required=True, help='Location for weather scenarios')
    forecast_parser.add_argument('--model-path', default='trained_model.json', help='Path to trained model')
    forecast_parser.add_argument('--scenarios', nargs='+', default=['baseline'], 
                                help='Forecast scenarios (baseline, warm_climate, cold_climate, extreme_heat)')
    forecast_parser.add_argument('--historical-weather', help='Historical weather data file (optional)')
    forecast_parser.add_argument('--output-dir', default='output', help='Output directory')
    
    # Common arguments
    for subparser in [train_parser, generate_parser, forecast_parser]:
        subparser.add_argument('--config', default='config.yaml', help='Configuration file')
        subparser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser


def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.load_config()
    
    # Setup logging
    if args.verbose:
        config['logging'] = config.get('logging', {})
        config['logging']['level'] = 'DEBUG'
    
    logger = setup_logging(config)
    
    try:
        logger.info(f"🚀 Starting Energy Load Profile Generator - {args.command} mode")
        
        if args.command == 'train':
            result = train_model(args, config, logger)
            logger.info("✅ Training completed successfully")
            
        elif args.command == 'generate':
            result = generate_profiles(args, config, logger)
            logger.info(f"✅ Profile generation completed: {result.get('output_directory', 'No output directory created')}")
            
        elif args.command == 'forecast':
            result = forecast_energy(args, config, logger)
            logger.info(f"✅ Forecasting completed: {result['output_path']}")
            
        else:
            logger.error(f"❌ Unknown command: {args.command}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ Operation cancelled by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)