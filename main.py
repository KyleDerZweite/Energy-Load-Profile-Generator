#!/usr/bin/env python3
"""
Energy Load Profile Generator - Main Script
==========================================
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

# Import project modules
from config_manager import ConfigManager
from weather_fetcher import MultiSourceWeatherFetcher
from weather_database import WeatherDatabase
from device_calculator import DeviceLoadCalculator
from analysis_export import LoadProfileAnalyzer

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
        description='Generate energy load profiles from weather data and device patterns',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
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

    parser.add_argument(
        '--use-optimized',
        action='store_true',
        help='Use optimized patterns from optimized_config.yaml'
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
    """List available devices."""
    devices = config_manager.get_section('devices')
    enabled_devices = config_manager.get_enabled_devices()

    print("\n=== Available Devices ===")
    print(f"{'Device Name':<20} {'Peak Power (W)':<15} {'Status':<10} {'Description'}")
    print("-" * 70)

    for device_name, device_config in devices.items():
        status = "Enabled" if device_name in enabled_devices else "Disabled"
        peak_power = device_config.get('peak_power', 0)
        comfort_temp = device_config.get('comfort_temp', 20)

        print(f"{device_name:<20} {peak_power:<15} {status:<10} "
              f"Comfort temp: {comfort_temp}°C")

    print(f"\nDefault devices: {', '.join(config_manager.get_value('load_profile', 'default_devices') or [])}")

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
    """Generate output filename."""
    output_config = config.get('load_profile', {}).get('output', {})
    prefix = output_config.get('filename_prefix', 'energy_load_profile')

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

def main():
    """Main execution function."""
    args = parse_arguments()

    if args.use_optimized:
        optimized_config_path = 'optimized_config.yaml'
        if os.path.exists(optimized_config_path):
            args.config = optimized_config_path
            logging.getLogger(__name__).info("Using optimized configuration")
        else:
            logging.getLogger(__name__).warning("Optimized config not found, using default config")

    # Load configuration
    try:
        config_manager = ConfigManager(args.config)
        config = config_manager.load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override logging if verbose/quiet specified
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    elif args.quiet:
        config['logging']['console'] = False

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting Energy Load Profile Generator")

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Initialize components
        logger.info("Initializing components...")

        db_path = config.get('database', {}).get('path', 'energy_weather.db')
        weather_fetcher = MultiSourceWeatherFetcher(config, db_path)
        device_calculator = DeviceLoadCalculator(config.get('devices', {}))
        analyzer = LoadProfileAnalyzer(config)

        # Determine devices to use
        if args.devices:
            devices = args.devices
            logger.info(f"Using specified devices: {devices}")
        else:
            devices = config_manager.get_enabled_devices()
            logger.info(f"Using enabled devices from config: {devices}")

        device_quantities = config_manager.get_device_quantities()

        # Step 1: Fetch weather data
        logger.info(f"Fetching weather data for {args.location} from {args.start_date} to {args.end_date}")
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

        # Step 2: Calculate load profile
        logger.info("Calculating energy load profile...")
        load_data = device_calculator.calculate_total_load(
            weather_data,
            devices,
            device_quantities
        )

        if load_data.empty:
            logger.error("Failed to generate load profile")
            sys.exit(1)

        # Step 3: Perform analysis
        logger.info("Performing load profile analysis...")
        analysis_results = analyzer.analyze_load_profile(load_data)

        # Step 4: Generate plots
        if not args.no_plots and config.get('analysis', {}).get('generate_plots', True):
            plots_dir = os.path.join(args.output_dir, 'plots')
            logger.info("Generating visualization plots...")
            analyzer.generate_plots(load_data, plots_dir)

        # Step 5: Export data
        output_format = args.format or config.get('load_profile', {}).get('output', {}).get('format', 'both')

        if output_format in ['csv', 'both']:
            csv_filename = os.path.join(args.output_dir, generate_output_filename(config, args, 'csv'))
            logger.info(f"Exporting to CSV: {csv_filename}")
            analyzer.export_to_csv(load_data, analysis_results, csv_filename)

        if output_format in ['xlsx', 'both']:
            xlsx_filename = os.path.join(args.output_dir, generate_output_filename(config, args, 'xlsx'))
            logger.info(f"Exporting to Excel: {xlsx_filename}")
            analyzer.export_to_excel(load_data, analysis_results, xlsx_filename)

        # Step 6: Display summary
        basic_stats = analysis_results.get('basic_statistics', {})
        logger.info("=== Generation Complete ===")
        logger.info(f"Location: {args.location}")
        logger.info(f"Period: {args.start_date} to {args.end_date}")
        if args.weather_source:
            logger.info(f"Weather source: {args.weather_source}")
        logger.info(f"Total energy: {basic_stats.get('total_energy_kwh', 0):.2f} kWh")
        logger.info(f"Average power: {basic_stats.get('average_power_w', 0):.0f} W")
        logger.info(f"Peak power: {basic_stats.get('max_power_w', 0):.0f} W")
        logger.info(f"Load factor: {basic_stats.get('load_factor', 0):.3f}")
        logger.info(f"Data points: {basic_stats.get('data_points', 0):,}")
        logger.info(f"Output directory: {args.output_dir}")

        # Device breakdown summary
        device_breakdown = analysis_results.get('device_breakdown', {})
        if device_breakdown:
            logger.info("\n=== Device Energy Consumption ===")
            for device, stats in device_breakdown.items():
                percentage = stats.get('percentage', 0)
                total_kwh = stats.get('total_kwh', 0)
                logger.info(f"{device}: {total_kwh:.2f} kWh ({percentage:.1f}%)")

        print(f"\n✅ Energy load profile generated successfully!")
        print(f"📁 Output files saved to: {args.output_dir}")
        if args.weather_source:
            print(f"🌦️  Weather data source: {args.weather_source}")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()