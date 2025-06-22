#!/usr/bin/env python3
"""
Energy Load Profile Generator - Realistic-First Main Script
===========================================================

This is the main script completely redesigned for realism-first approach.
All mathematical approaches have been replaced with physics-based,
realistic device behavior models.

Features:
- Automatic pattern enhancement for realism
- Physics-based device calculations
- AI-driven adaptation to device changes
- Thermal inertia and natural variations
- Completely removes mathematical-only approaches
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# Import project modules
from config_manager import ConfigManager
from weather_fetcher import MultiSourceWeatherFetcher
from weather_database import WeatherDatabase
from device_calculator import DeviceLoadCalculator  # Now intelligent & realistic
from analysis_export import LoadProfileAnalyzer
from pattern_optimizer import IntelligentPatternOptimizer

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

    # Setup file handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=log_config.get('max_file_size_mb', 10) * 1024 * 1024,
        backupCount=log_config.get('backup_count', 5)
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)

    if log_config.get('console', True):
        root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate REALISTIC energy load profiles with physics-based device behavior',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Main arguments (required unless using info-only commands)
    parser.add_argument(
        '--location', '-l',
        type=str,
        help='Location for weather data (e.g., "Berlin, Germany")'
    )

    parser.add_argument(
        '--start-date', '-s',
        type=str,
        help='Start date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--end-date', '-e',
        type=str,
        help='End date in YYYY-MM-DD format'
    )

    # Optional arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Configuration file path'
    )

    parser.add_argument(
        '--devices', '-d',
        type=str,
        nargs='+',
        help='Specific devices to include (overrides config)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output',
        help='Output directory for results'
    )

    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'xlsx', 'both'],
        help='Output format (overrides config)'
    )

    parser.add_argument(
        '--weather-source', '-w',
        choices=['open_meteo', 'weatherapi', 'dwd'],
        help='Preferred weather data source (optional)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )

    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh weather data (ignore cache)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress console output'
    )

    parser.add_argument(
        '--use-optimized',
        action='store_true',
        help='Use optimized patterns from optimized_config.yaml'
    )
    
    parser.add_argument(
        '--training-data',
        type=str,
        help='Historical load data file for AI training (Excel/CSV)'
    )
    
    parser.add_argument(
        '--enable-learning',
        action='store_true',
        help='Enable AI learning from historical data'
    )
    
    parser.add_argument(
        '--discover-devices',
        action='store_true',
        help='Auto-discover devices from load profile'
    )
    
    parser.add_argument(
        '--building-size',
        type=float,
        help='Building size in square meters (overrides config)'
    )
    
    parser.add_argument(
        '--occupancy-type',
        choices=['single', 'family', 'elderly', 'student'],
        help='Occupancy type for lifestyle modeling'
    )
    
    parser.add_argument(
        '--iterative-training',
        action='store_true',
        help='Run iterative 2024->2025 training workflow'
    )

    # Intelligent behavior controls
    parser.add_argument(
        '--disable-adaptation',
        action='store_true',
        help='Disable automatic AI adaptation (still realistic, but static)'
    )
    
    parser.add_argument(
        '--disable-learning',
        action='store_true',
        help='Disable all AI learning features'
    )

    parser.add_argument(
        '--realism-level',
        type=float,
        default=1.0,
        help='Realism level factor (0.5-2.0, default=1.0)'
    )

    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available devices and exit'
    )

    parser.add_argument(
        '--list-locations',
        action='store_true',
        help='List priority locations and exit'
    )

    parser.add_argument(
        '--list-weather-sources',
        action='store_true',
        help='List available weather sources and exit'
    )

    parser.add_argument(
        '--db-stats',
        action='store_true',
        help='Show database statistics and exit'
    )

    return parser.parse_args()

def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range."""
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_dt >= end_dt:
            print("Error: Start date must be before end date")
            return False

        if end_dt > datetime.now():
            print("Warning: End date is in the future")

        days_diff = (end_dt - start_dt).days
        if days_diff > 365:
            print(f"Warning: Date range is {days_diff} days. Large ranges may take longer to process.")

        return True

    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD. {e}")
        return False

def list_devices(config_manager: ConfigManager):
    """List available devices with intelligent features."""
    devices = config_manager.get_all_devices()
    enabled_devices = config_manager.get_enabled_devices()

    print("\n=== Available Devices (Intelligent Realistic System) ===")
    print(f"{'Device Name':<20} {'Peak Power (W)':<15} {'Status':<10} {'AI Features'}")
    print("-" * 80)

    ai_features = {
        'heater': 'Smart learning, thermal physics, occupancy adaptation',
        'air_conditioner': 'Efficiency learning, weather correlation, thermal mass',
        'refrigeration': 'Cycle optimization, temperature learning',
        'general_load': 'Pattern discovery, lifestyle adaptation',
        'lighting': 'Daylight learning, occupancy detection',
        'water_heater': 'Usage pattern learning, thermal optimization'
    }

    for device_name, device_config in devices.items():
        status = "Enabled" if device_name in enabled_devices else "Disabled"
        peak_power = device_config.get('peak_power', 0)
        features = ai_features.get(device_name, 'Intelligent learning + physics')
        
        # Show if device was discovered
        if device_config.get('discovered', False):
            status += " (AI-discovered)"

        print(f"{device_name:<20} {peak_power:<15} {status:<10} {features}")

    print(f"\n🧠 Intelligent learning with physics-based realistic behavior")
    print(f"🔍 Automatic device discovery from load patterns")
    print(f"🎯 Multi-scale learning (15min to yearly patterns)")
    print(f"🏠 Building-aware efficiency modeling")

def list_locations(config_manager: ConfigManager):
    """List priority locations."""
    locations = config_manager.get_value('locations', 'priority_cities') or []

    print("\n=== Priority German Cities ===")
    for i, city in enumerate(locations, 1):
        print(f"{i:2d}. {city}")

def list_weather_sources(config_manager: ConfigManager):
    """List available weather sources."""
    sources = config_manager.get_value('weather_sources') or []

    print("\n=== Available Weather Sources ===")
    print(f"{'Source':<15} {'Enabled':<8} {'Priority':<8} {'Description'}")
    print("-" * 80)

    for source in sources:
        name = source.get('name', 'Unknown')
        enabled = "Yes" if source.get('enabled', False) else "No"
        priority = source.get('priority', 'N/A')
        description = source.get('description', 'No description')

        print(f"{name:<15} {enabled:<8} {priority:<8} {description}")

    print("\nUsage: --weather-source <source_name>")
    print("Example: --weather-source dwd")

def show_database_stats(config_manager: ConfigManager):
    """Show database statistics."""
    db_path = config_manager.get_value('database', 'path')
    db = WeatherDatabase(db_path)
    stats = db.get_database_stats()

    print("\n=== Database Statistics ===")
    print(f"Database file: {db_path}")
    print(f"File size: {stats.get('database_size_mb', 0):.2f} MB")
    print(f"Total weather records: {stats.get('total_weather_records', 0):,}")
    print(f"Total locations: {stats.get('total_locations', 0)}")
    print(f"Date range: {stats.get('earliest_date', 'N/A')} to {stats.get('latest_date', 'N/A')}")

    print("\nRecords by source:")
    for source, count in stats.get('by_source', {}).items():
        print(f"  {source}: {count:,}")

def generate_output_filename(config: Dict, args, format_type: str) -> str:
    """Generate output filename with realistic mode indicator."""
    output_config = config.get('load_profile', {}).get('output', {})
    prefix = output_config.get('filename_prefix', 'energy_load_profile')

    # Always add realistic indicator since this is now the only mode
    prefix += '_realistic'

    # Clean location name for filename
    location_clean = args.location.replace(', ', '_').replace(' ', '_').replace(',', '')

    filename = f"{prefix}_{location_clean}_{args.start_date}_to_{args.end_date}"

    # Add weather source if specified
    if args.weather_source:
        filename += f"_{args.weather_source}"

    if output_config.get('add_timestamp', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename += f"_{timestamp}"

    return f"{filename}.{format_type}"

def run_iterative_training(args, config_manager: ConfigManager, logger):
    """Run iterative 2024->2025 training workflow."""
    logger.info("🎯 Starting iterative training workflow: 2024 -> 2025")
    
    if not args.training_data:
        logger.error("Training data file required for iterative training (--training-data)")
        return False
    
    if not os.path.exists(args.training_data):
        logger.error(f"Training data file not found: {args.training_data}")
        return False
    
    # Initialize optimizer with intelligent learning
    optimizer = IntelligentPatternOptimizer(args.config)
    
    # Get training configuration
    training_config = config_manager.get_training_configuration()
    logger.info(f"Training: {training_config['train_year']} -> {training_config['target_year']}")
    
    try:
        # Run optimization with intelligence
        results = optimizer.optimize_with_intelligence(
            target_data_path=args.training_data,
            location=args.location,
            learning_enabled=True
        )
        
        logger.info(f"✅ Training completed with {results.get('iterations', 0)} iterations")
        logger.info(f"🎯 Final error: {results.get('final_error', 0):.3f}")
        logger.info(f"📈 Convergence: {results.get('converged', False)}")
        
        # Save learned patterns
        learned_devices = results.get('learned_devices', {})
        for device_name, device_params in learned_devices.items():
            config_manager.update_device_from_learning(device_name, device_params)
            logger.info(f"📚 Learned parameters for {device_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def discover_devices_from_data(args, config_manager: ConfigManager, logger):
    """Discover devices from historical load data."""
    if not args.training_data:
        logger.warning("No training data provided for device discovery")
        return
    
    logger.info(f"🔍 Discovering devices from {args.training_data}")
    
    try:
        # Load historical data
        if args.training_data.endswith('.xlsx'):
            data = pd.read_excel(args.training_data)
        else:
            data = pd.read_csv(args.training_data)
        
        # Initialize device calculator for discovery
        device_calculator = DeviceLoadCalculator(config_manager.get_all_devices())
        
        # Learn from historical data (includes device discovery)
        learning_results = device_calculator.learn_from_historical_data(data)
        
        # Add discovered devices to configuration
        discovered_devices = learning_results.get('discovered_devices', {})
        for device_name, characteristics in discovered_devices.items():
            config_manager.add_discovered_device(device_name, characteristics)
            logger.info(f"🆕 Discovered device: {device_name} (confidence: {characteristics.get('confidence', 0):.2f})")
        
        if discovered_devices:
            logger.info(f"✅ Discovered {len(discovered_devices)} new devices")
        else:
            logger.info("No new devices discovered (existing configuration sufficient)")
            
    except Exception as e:
        logger.error(f"Device discovery failed: {e}")

def main():
    """Main execution function - Intelligent Realistic approach."""
    args = parse_arguments()

    if args.use_optimized:
        optimized_config_path = 'optimized_config.yaml'
        if os.path.exists(optimized_config_path):
            args.config = optimized_config_path
            logging.getLogger(__name__).info("Using optimized realistic configuration")
        else:
            logging.getLogger(__name__).warning("Optimized config not found, using default realistic config")

    # Load configuration - prioritize optimized_config.yaml if available
    try:
        if args.config == 'config.yaml' and os.path.exists('optimized_config.yaml'):
            args.config = 'optimized_config.yaml'
            logging.getLogger(__name__).info("🎯 Using optimized configuration for best settings")
        config_manager = ConfigManager(args.config)
        config = config_manager.load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override building parameters if specified
    if args.building_size:
        building_updates = {'building': {'total_area_sqm': args.building_size}}
        config_manager.update_devices_config(building_updates)
        
    if args.occupancy_type:
        building_updates = {'building': {'occupancy_schedule': args.occupancy_type}}
        config_manager.update_devices_config(building_updates)
        
    # Override learning settings
    if args.disable_learning:
        config['learning']['enabled'] = False
        
    if args.enable_learning and args.training_data:
        config['learning']['enabled'] = True
    
    # Override logging if verbose/quiet specified
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    elif args.quiet:
        config['logging']['console'] = False

    # Setup logging
    logger = setup_logging(config)
    logger.info("🚀 Starting INTELLIGENT Energy Load Profile Generator")
    logger.info("🧠 AI-enhanced physics-based device behavior with learning")
    
    # Show building and learning configuration
    building_config = config_manager.get_building_config()
    learning_config = config.get('learning', {})
    logger.info(f"🏠 Building: {building_config.get('total_area_sqm', 15000)}m², {building_config.get('occupancy_schedule', 'academic_calendar')} schedule")
    logger.info(f"🎓 Learning: {'ENABLED' if learning_config.get('enabled', True) else 'DISABLED'}")

    # Handle info-only commands
    if args.list_devices:
        list_devices(config_manager)
        return

    if args.list_locations:
        list_locations(config_manager)
        return

    if args.list_weather_sources:
        list_weather_sources(config_manager)
        return

    if args.db_stats:
        show_database_stats(config_manager)
        return

    # Validate configuration
    if not config_manager.validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)

    # Validate date range
    if not validate_date_range(args.start_date, args.end_date):
        sys.exit(1)

    # Validate weather source if specified
    if args.weather_source:
        available_sources = [s['name'] for s in config.get('weather_sources', [])]
        if args.weather_source not in available_sources:
            logger.error(f"Invalid weather source '{args.weather_source}'. Available: {', '.join(available_sources)}")
            sys.exit(1)
        logger.info(f"Using preferred weather source: {args.weather_source}")

    # Handle iterative training workflow
    if args.iterative_training:
        success = run_iterative_training(args, config_manager, logger)
        if not success:
            sys.exit(1)
        logger.info("📚 Iterative training completed - patterns optimized for realism")
    
    # Handle device discovery
    if args.discover_devices and args.training_data:
        discover_devices_from_data(args, config_manager, logger)
    
    # Create timestamped output directory
    location_clean = args.location.lower().replace(', ', '_').replace(' ', '_').replace(',', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_output_dir = os.path.join(args.output_dir, f"{location_clean}_{timestamp}")
    os.makedirs(timestamped_output_dir, exist_ok=True)
    logger.info(f"📁 Output directory: {timestamped_output_dir}")
    
    # Update args.output_dir to use the timestamped directory
    args.output_dir = timestamped_output_dir

    try:
        # Initialize components - All realistic now
        logger.info("Initializing realistic components...")

        db_path = config.get('database', {}).get('path', 'energy_weather.db')
        weather_fetcher = MultiSourceWeatherFetcher(config, db_path)
        
        # Device calculator is now always realistic-first
        logger.info("🧠 Initializing Realistic Device Calculator with auto-enhancement")
        device_calculator = DeviceLoadCalculator(config_manager.get_all_devices())
        
        # Disable adaptation if requested
        if args.disable_adaptation:
            device_calculator.adaptation_enabled = False
            logger.info("⚙️  Auto-adaptation disabled (static realistic behavior)")
        
        analyzer = LoadProfileAnalyzer(config, config_manager)

        # Determine devices to use
        if args.devices:
            devices = args.devices
            logger.info(f"Using specified devices: {devices}")
        else:
            devices = config_manager.get_enabled_devices()
            logger.info(f"Using enabled devices from config: {devices}")

        device_quantities = config_manager.get_device_quantities()

        # Apply realism level adjustment
        if args.realism_level != 1.0:
            logger.info(f"🎚️  Applying realism level factor: {args.realism_level}")
            for device_name in devices:
                if device_name in config_manager.get_all_devices():
                    # Adjust noise levels and variation factors
                    device_calculator._device_states[device_name]['realism_factor'] = args.realism_level

        # Step 1: Fetch weather data
        logger.info(f"🌦️  Fetching weather data for {args.location} from {args.start_date} to {args.end_date}")
        if args.weather_source:
            logger.info(f"Preferred weather source: {args.weather_source}")

        weather_data = weather_fetcher.get_weather_data(
            args.location,
            args.start_date,
            args.end_date,
            force_refresh=args.force_refresh,
            preferred_source=args.weather_source
        )

        if weather_data.empty:
            logger.error("No weather data available for the specified location and date range")
            sys.exit(1)

        logger.info(f"Retrieved {len(weather_data):,} weather records")

        # Step 2: Calculate realistic load profile
        logger.info("🏠 Calculating physics-based realistic energy load profile...")
        logger.info("🔧 Features: Thermal inertia, device cycles, AI adaptation, natural variations")
        
        load_data = device_calculator.calculate_total_load(
            weather_data,
            devices,
            device_quantities
        )

        if load_data.empty:
            logger.error("Failed to generate load profile")
            sys.exit(1)

        # Step 3: Perform analysis with realism metrics
        logger.info("📊 Performing enhanced load profile analysis...")
        analysis_results = analyzer.analyze_load_profile(load_data)
        
        # Add realism statistics
        realism_stats = {}
        for device in devices:
            device_stats = device_calculator.get_device_statistics(load_data, device)
            if device_stats:
                realism_stats[device] = {
                    'realism_score': device_stats.get('realism_score', 0),
                    'smooth_transitions_pct': device_stats.get('smooth_transitions_pct', 0),
                    'max_transition_w': device_stats.get('max_transition_w', 0),
                    'efficiency_factor': device_stats.get('efficiency_factor', 1.0),
                    'adaptation_factor': device_stats.get('adaptation_factor', 1.0)
                }
        
        analysis_results['realism_statistics'] = realism_stats

        # Step 4: Generate plots
        if not args.no_plots and config.get('analysis', {}).get('generate_plots', True):
            plots_dir = os.path.join(args.output_dir, 'plots')
            logger.info("📈 Generating realistic behavior visualization plots...")
            analyzer.generate_plots(load_data, plots_dir)

        # Step 5: Export data
        output_format = args.format or config.get('load_profile', {}).get('output', {}).get('format', 'both')

        if output_format in ['csv', 'both']:
            csv_filename = os.path.join(args.output_dir, generate_output_filename(config, args, 'csv'))
            logger.info(f"💾 Exporting to CSV: {csv_filename}")
            analyzer.export_to_csv(load_data, analysis_results, csv_filename)

        if output_format in ['xlsx', 'both']:
            xlsx_filename = os.path.join(args.output_dir, generate_output_filename(config, args, 'xlsx'))
            logger.info(f"💾 Exporting to Excel: {xlsx_filename}")
            analyzer.export_to_excel(load_data, analysis_results, xlsx_filename)

        # Step 6: Display enhanced summary with realism metrics
        basic_stats = analysis_results.get('basic_statistics', {})
        
        logger.info("=== 🚀 REALISTIC Generation Complete ===")
        logger.info(f"🏢 Location: {args.location}")
        logger.info(f"📅 Period: {args.start_date} to {args.end_date}")
        if args.weather_source:
            logger.info(f"🌦️  Weather source: {args.weather_source}")
        logger.info(f"⚡ Total energy: {basic_stats.get('total_energy_kwh', 0):.2f} kWh")
        logger.info(f"📊 Average power: {basic_stats.get('average_power_w', 0):.0f} W")
        logger.info(f"🔝 Peak power: {basic_stats.get('max_power_w', 0):.0f} W")
        logger.info(f"⚖️  Load factor: {basic_stats.get('load_factor', 0):.3f}")
        logger.info(f"📈 Data points: {basic_stats.get('data_points', 0):,}")
        logger.info(f"📁 Output directory: {args.output_dir}")

        # Enhanced device breakdown with realism scores
        device_breakdown = analysis_results.get('device_breakdown', {})
        if device_breakdown:
            logger.info("\n=== 🏠 Device Energy Consumption & Realism ===")
            for device, stats in device_breakdown.items():
                percentage = stats.get('percentage', 0)
                total_kwh = stats.get('total_kwh', 0)
                realism_score = realism_stats.get(device, {}).get('realism_score', 0)
                smooth_pct = realism_stats.get(device, {}).get('smooth_transitions_pct', 0)
                
                logger.info(f"{device}: {total_kwh:.2f} kWh ({percentage:.1f}%) - "
                           f"Realism: {realism_score:.0f}/100, Smooth: {smooth_pct:.0f}%")

        # Overall realism assessment
        avg_realism = np.mean([stats.get('realism_score', 0) for stats in realism_stats.values()])
        logger.info(f"\n🎯 Overall Realism Score: {avg_realism:.0f}/100")
        
        if avg_realism >= 85:
            logger.info("✅ EXCELLENT: Highly realistic behavior achieved")
        elif avg_realism >= 70:
            logger.info("✅ GOOD: Realistic behavior with minor mathematical artifacts")
        elif avg_realism >= 50:
            logger.info("⚠️  ACCEPTABLE: Some unrealistic patterns detected")
        else:
            logger.info("❌ POOR: Significant unrealistic patterns - check configuration")

        print(f"\n🚀 Realistic energy load profile generated successfully!")
        print(f"🧠 Physics-based behavior: ENABLED")
        print(f"🌊 Thermal inertia & smooth transitions: APPLIED")
        print(f"🔧 Auto-enhancement: COMPLETED")
        print(f"📁 Output files saved to: {args.output_dir}")
        if args.weather_source:
            print(f"🌦️  Weather data source: {args.weather_source}")
        print(f"🎯 Realism Score: {avg_realism:.0f}/100")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()