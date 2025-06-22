import yaml
import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import copy
import numpy as np
import pandas as pd

class ConfigManager:
    """Enhanced multi-file configuration manager for university energy profile system."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = None
        self.devices_config = None
        self.optimization_config = None
        self.logger = logging.getLogger(__name__)
        
        # Enhanced configuration tracking
        self.dynamic_devices = {}
        self.learned_patterns = {}
        self.building_parameters = {}
        self.optimization_history = []

    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration from multiple files."""
        try:
            # Load main configuration
            if not os.path.exists(self.config_path):
                self.logger.warning(f"Config file {self.config_path} not found, creating default config")
                self.create_default_config()

            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)

            # Load devices configuration
            devices_path = self.config.get('config_files', {}).get('devices', 'devices.json')
            self._load_devices_config(devices_path)

            # Load optimization configuration
            optimization_path = self.config.get('config_files', {}).get('optimization', 'optimization_config.yaml')
            self._load_optimization_config(optimization_path)

            # Validate and set defaults
            self.config = self._validate_and_set_defaults(self.config)
            
            # Initialize enhanced features
            self._initialize_building_parameters()
            self._initialize_intelligent_features()

            self.logger.info(f"Multi-file configuration loaded successfully")
            return self.config

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            self.logger.error(f"Error parsing configuration files: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def _load_devices_config(self, devices_path: str):
        """Load devices configuration from JSON file."""
        try:
            if os.path.exists(devices_path):
                with open(devices_path, 'r', encoding='utf-8') as file:
                    self.devices_config = json.load(file)
                self.logger.info(f"Devices configuration loaded from {devices_path}")
            else:
                self.logger.warning(f"Devices config file {devices_path} not found, using defaults")
                self.devices_config = self._get_default_devices_config()
        except Exception as e:
            self.logger.error(f"Error loading devices config: {e}")
            self.devices_config = self._get_default_devices_config()

    def _load_optimization_config(self, optimization_path: str):
        """Load optimization configuration from YAML file."""
        try:
            if os.path.exists(optimization_path):
                with open(optimization_path, 'r', encoding='utf-8') as file:
                    self.optimization_config = yaml.safe_load(file)
                self.logger.info(f"Optimization configuration loaded from {optimization_path}")
            else:
                self.logger.warning(f"Optimization config file {optimization_path} not found, using defaults")
                self.optimization_config = self._get_default_optimization_config()
        except Exception as e:
            self.logger.error(f"Error loading optimization config: {e}")
            self.optimization_config = self._get_default_optimization_config()

    def _validate_and_set_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and set default values."""
        validated_config = copy.deepcopy(config)

        # API Keys section
        if 'api_keys' not in validated_config:
            validated_config['api_keys'] = {}

        # Database configuration
        db_defaults = {
            'path': 'energy_weather.db',
            'auto_backup': True,
            'backup_interval_days': 7
        }
        validated_config['database'] = {**db_defaults, **validated_config.get('database', {})}

        # Weather sources
        if 'weather_sources' not in validated_config:
            validated_config['weather_sources'] = self._get_default_weather_sources()

        # Load profile configuration
        load_profile_defaults = {
            'output': {
                'format': 'xlsx',
                'include_weather_data': True,
                'include_device_breakdown': True,
                'include_statistics': True,
                'filename_prefix': 'energy_load_profile',
                'add_timestamp': True
            }
        }
        validated_config['load_profile'] = {**load_profile_defaults, **validated_config.get('load_profile', {})}

        # Analysis configuration
        analysis_defaults = {
            'include_hourly_patterns': True,
            'include_monthly_patterns': True,
            'include_seasonal_analysis': True,
            'include_temperature_correlation': True,
            'include_peak_analysis': True,
            'generate_plots': True,
            'plot_days': 14,
            'plot_format': 'png',
            'plot_dpi': 300
        }
        validated_config['analysis'] = {**analysis_defaults, **validated_config.get('analysis', {})}

        # Locations
        if 'locations' not in validated_config:
            validated_config['locations'] = self._get_default_locations()

        # Logging configuration
        logging_defaults = {
            'level': 'INFO',
            'file': 'energy_load_profile.log',
            'console': True,
            'max_file_size_mb': 10,
            'backup_count': 5
        }
        validated_config['logging'] = {**logging_defaults, **validated_config.get('logging', {})}

        return validated_config

    def _get_default_weather_sources(self) -> List[Dict[str, Any]]:
        """Get default weather source configurations."""
        return [
            {
                'name': 'open_meteo',
                'enabled': True,
                'priority': 1,
                'description': 'Open-Meteo ERA5 (1940-present, FREE)',
                'date_range': ['1940-01-01', '2024-12-31']
            },
            {
                'name': 'weatherapi',
                'enabled': True,
                'priority': 2,
                'description': 'WeatherAPI.com (recent data, 1 year limit)',
                'date_range': ['2023-01-01', '2024-12-31']
            }
        ]

    def _get_default_devices_config(self) -> Dict[str, Any]:
        """Get minimal default devices configuration."""
        return {
            "building": {
                "type": "university_campus",
                "total_area_sqm": 15000,
                "occupancy_schedule": "academic_calendar"
            },
            "devices": {},
            "device_quantities": {}
        }

    def _get_default_optimization_config(self) -> Dict[str, Any]:
        """Get default optimization configuration."""
        return {
            "learning_rate": 0.01,
            "max_episodes": 1000,
            "optimization": {
                "algorithm": "genetic",
                "population_size": 40,
                "generations": 100
            }
        }

    def _get_default_locations(self) -> Dict[str, List[str]]:
        """Get default German cities."""
        return {
            'priority_cities': [
                'Berlin, Germany',
                'München, Germany',
                'Hamburg, Germany',
                'Köln, Germany',
                'Frankfurt am Main, Germany',
                'Stuttgart, Germany',
                'Düsseldorf, Germany',
                'Dortmund, Germany',
                'Essen, Germany',
                'Leipzig, Germany'
            ]
        }

    def create_default_config(self):
        """Create a default main configuration file."""
        default_config = {
            'api_keys': {
                'weatherapi_key': 'your_weatherapi_key_here'
            },
            'database': {
                'path': 'energy_weather.db',
                'auto_backup': True,
                'backup_interval_days': 7
            },
            'weather_sources': self._get_default_weather_sources(),
            'load_profile': {
                'output': {
                    'format': 'xlsx',
                    'include_weather_data': True,
                    'include_device_breakdown': True,
                    'include_statistics': True,
                    'filename_prefix': 'energy_load_profile',
                    'add_timestamp': True
                }
            },
            'analysis': {
                'include_hourly_patterns': True,
                'include_monthly_patterns': True,
                'include_seasonal_analysis': True,
                'include_temperature_correlation': True,
                'include_peak_analysis': True,
                'generate_plots': True,
                'plot_days': 14,
                'plot_format': 'png',
                'plot_dpi': 300
            },
            'locations': self._get_default_locations(),
            'logging': {
                'level': 'INFO',
                'file': 'energy_load_profile.log',
                'console': True,
                'max_file_size_mb': 10,
                'backup_count': 5
            },
            'config_files': {
                'devices': 'devices.json',
                'optimization': 'optimization_config.yaml'
            }
        }

        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(default_config, file, default_flow_style=False, indent=2, sort_keys=False, allow_unicode=True)
            self.logger.info(f"Default configuration created at {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to create default config: {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        """Get the loaded main configuration."""
        if self.config is None:
            self.load_config()
        return self.config

    def get_devices_config(self) -> Dict[str, Any]:
        """Get the loaded devices configuration."""
        if self.devices_config is None:
            self.load_config()
        return self.devices_config

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get the loaded optimization configuration."""
        if self.optimization_config is None:
            self.load_config()
        return self.optimization_config

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get a specific configuration section."""
        config = self.get_config()
        return config.get(section_name, {})

    def get_devices_section(self, section_name: str) -> Dict[str, Any]:
        """Get a specific devices configuration section."""
        devices_config = self.get_devices_config()
        return devices_config.get(section_name, {})

    def get_value(self, *keys) -> Any:
        """Get a nested configuration value from main config."""
        config = self.get_config()
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def get_devices_value(self, *keys) -> Any:
        """Get a nested configuration value from devices config."""
        config = self.get_devices_config()
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def get_building_config(self) -> Dict[str, Any]:
        """Get building configuration from devices config."""
        return self.get_devices_section('building')

    def get_all_devices(self) -> Dict[str, Any]:
        """Get all device configurations."""
        return self.get_devices_section('devices')

    def get_device_quantities(self) -> Dict[str, int]:
        """Get device quantities from devices configuration."""
        return self.get_devices_section('device_quantities')

    def get_enabled_devices(self) -> List[str]:
        """Get list of enabled device names."""
        devices = self.get_all_devices()
        return [name for name, config in devices.items() if config.get('enabled', True)]

    def get_devices_by_category(self, category: str) -> Dict[str, Any]:
        """Get devices filtered by category."""
        devices = self.get_all_devices()
        return {name: config for name, config in devices.items() 
                if config.get('category') == category}

    def get_devices_by_power_class(self, power_class: str) -> Dict[str, Any]:
        """Get devices filtered by power class."""
        devices = self.get_all_devices()
        return {name: config for name, config in devices.items() 
                if config.get('power_class') == power_class}

    def get_critical_devices(self) -> List[str]:
        """Get list of critical priority devices."""
        devices = self.get_all_devices()
        return [name for name, config in devices.items() 
                if config.get('priority') == 'critical']

    def validate_device_config(self, device_name: str, device_config: Dict[str, Any]) -> bool:
        """Validate device configuration."""
        required_fields = ['peak_power', 'temp_coefficient', 'comfort_temp', 'daily_pattern']

        for field in required_fields:
            if field not in device_config:
                self.logger.error(f"Device '{device_name}' missing required field: {field}")
                return False

        # Validate daily pattern
        daily_pattern = device_config['daily_pattern']
        if not isinstance(daily_pattern, list) or len(daily_pattern) != 96:
            self.logger.error(f"Device '{device_name}' has invalid daily_pattern (must be list of 96 values for 15-min intervals), got {len(daily_pattern)} values")
            return False

        # Validate daily pattern values are between 0.0 and 1.0
        for i, value in enumerate(daily_pattern):
            if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
                self.logger.error(f"Device '{device_name}' daily_pattern[{i}] = {value} must be between 0.0 and 1.0")
                return False

        # Validate numeric values
        numeric_fields = ['peak_power', 'temp_coefficient', 'comfort_temp']
        for field in numeric_fields:
            if not isinstance(device_config[field], (int, float)):
                self.logger.error(f"Device '{device_name}' field '{field}' must be numeric")
                return False

        # Validate peak_power is positive
        if device_config['peak_power'] <= 0:
            self.logger.error(f"Device '{device_name}' peak_power must be positive")
            return False

        return True

    def validate_config(self) -> bool:
        """Validate the entire configuration."""
        # Validate main config
        config = self.get_config()
        
        # Validate devices
        devices = self.get_all_devices()
        for device_name, device_config in devices.items():
            if not self.validate_device_config(device_name, device_config):
                return False

        # Validate weather sources
        weather_sources = config.get('weather_sources', [])
        if not weather_sources:
            self.logger.warning("No weather sources configured")

        # Validate database path
        db_path = config.get('database', {}).get('path')
        if not db_path:
            self.logger.error("Database path not configured")
            return False

        # Validate building config
        building_config = self.get_building_config()
        if not building_config:
            self.logger.warning("No building configuration found")

        self.logger.info("Configuration validation passed")
        return True

    def _initialize_building_parameters(self):
        """Initialize building-specific parameters for realistic modeling."""
        building_config = self.get_building_config()
        
        # Calculate derived parameters
        square_meters = building_config.get('total_area_sqm', 15000)
        ceiling_height = building_config.get('ceiling_height', 3.2)
        insulation_rating = building_config.get('insulation_rating', 0.8)
        
        self.building_parameters = {
            'volume': square_meters * ceiling_height,
            'surface_area': square_meters + (2 * ceiling_height * np.sqrt(square_meters)),
            'heat_capacity': square_meters * 250 * insulation_rating,  # kJ/K
            'heat_loss_coefficient': (1.0 - insulation_rating) * 0.5,  # W/K/m²
            'thermal_time_constant': square_meters * insulation_rating * 2.0,  # hours
            'base_infiltration': building_config.get('air_changes_per_hour', 2.5)
        }
        
        self.logger.info(f"Building parameters initialized: {square_meters}m², insulation {insulation_rating:.1f}")

    def _initialize_intelligent_features(self):
        """Initialize AI learning and adaptation features."""
        # For university buildings, we focus on occupancy patterns and equipment scheduling
        self.intelligent_features = {
            'device_discovery_enabled': True,
            'pattern_learning_enabled': True,
            'occupancy_detection_enabled': True,
            'weather_adaptation_enabled': True,
            'equipment_scheduling_enabled': True,
            'peak_demand_management': True
        }
        self.logger.info("University intelligent features initialized")

    def get_building_parameters(self) -> Dict[str, float]:
        """Get calculated building parameters for device modeling."""
        return self.building_parameters.copy()

    def get_thermal_properties(self) -> Dict[str, float]:
        """Get thermal properties for realistic heating/cooling calculations."""
        building_config = self.get_building_config()
        
        return {
            'thermal_mass': building_config.get('thermal_mass', 0.7),
            'insulation_rating': building_config.get('insulation_rating', 0.8),
            'air_changes_per_hour': building_config.get('air_changes_per_hour', 2.5),
            'window_ratio': building_config.get('window_ratio', 0.25),
            'heat_capacity': self.building_parameters.get('heat_capacity', 300000),
            'time_constant': self.building_parameters.get('thermal_time_constant', 150)
        }

    def get_power_management_config(self) -> Dict[str, Any]:
        """Get power management configuration."""
        return self.get_devices_section('power_management')

    def update_config(self, updates: Dict[str, Any]):
        """Update main configuration values and save to file."""
        config = self.get_config()

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(config, updates)

        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, indent=2, sort_keys=False, allow_unicode=True)
            self.logger.info(f"Main configuration updated and saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save updated config: {e}")
            raise

    def update_devices_config(self, updates: Dict[str, Any]):
        """Update devices configuration and save to file."""
        devices_config = self.get_devices_config()

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(devices_config, updates)

        devices_path = self.config.get('config_files', {}).get('devices', 'devices.json')
        try:
            with open(devices_path, 'w', encoding='utf-8') as file:
                json.dump(devices_config, file, indent=2, ensure_ascii=False)
            self.logger.info(f"Devices configuration updated and saved to {devices_path}")
        except Exception as e:
            self.logger.error(f"Failed to save updated devices config: {e}")
            raise

    def add_device(self, device_name: str, device_config: Dict[str, Any], quantity: int = 1):
        """Add a new device to the configuration."""
        if self.validate_device_config(device_name, device_config):
            # Add to devices
            devices_updates = {
                'devices': {device_name: device_config},
                'device_quantities': {device_name: quantity}
            }
            self.update_devices_config(devices_updates)
            self.logger.info(f"Added device '{device_name}' with quantity {quantity}")
            return True
        return False

    def remove_device(self, device_name: str):
        """Remove a device from the configuration."""
        devices_config = self.get_devices_config()
        
        if device_name in devices_config.get('devices', {}):
            del devices_config['devices'][device_name]
        
        if device_name in devices_config.get('device_quantities', {}):
            del devices_config['device_quantities'][device_name]
        
        # Save updated config
        devices_path = self.config.get('config_files', {}).get('devices', 'devices.json')
        try:
            with open(devices_path, 'w', encoding='utf-8') as file:
                json.dump(devices_config, file, indent=2, ensure_ascii=False)
            self.logger.info(f"Removed device '{device_name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config after removing device: {e}")
            return False

    def list_devices_by_properties(self, **filters) -> List[str]:
        """List devices matching specific properties."""
        devices = self.get_all_devices()
        matching_devices = []
        
        for device_name, device_config in devices.items():
            matches = True
            for prop, value in filters.items():
                if device_config.get(prop) != value:
                    matches = False
                    break
            if matches:
                matching_devices.append(device_name)
        
        return matching_devices

    def get_total_peak_power(self) -> float:
        """Calculate total peak power consumption of all enabled devices."""
        devices = self.get_all_devices()
        quantities = self.get_device_quantities()
        total_power = 0
        
        for device_name, device_config in devices.items():
            if device_config.get('enabled', True):
                peak_power = device_config.get('peak_power', 0)
                quantity = quantities.get(device_name, 1)
                total_power += peak_power * quantity
        
        return total_power

    def get_device_statistics(self) -> Dict[str, Any]:
        """Get comprehensive device statistics."""
        devices = self.get_all_devices()
        quantities = self.get_device_quantities()
        
        stats = {
            'total_devices': len(devices),
            'enabled_devices': len(self.get_enabled_devices()),
            'total_peak_power': self.get_total_peak_power(),
            'categories': {},
            'power_classes': {},
            'priorities': {}
        }
        
        for device_name, device_config in devices.items():
            # Count by category
            category = device_config.get('category', 'unknown')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Count by power class
            power_class = device_config.get('power_class', 'unknown')
            stats['power_classes'][power_class] = stats['power_classes'].get(power_class, 0) + 1
            
            # Count by priority
            priority = device_config.get('priority', 'unknown')
            stats['priorities'][priority] = stats['priorities'].get(priority, 0) + 1
        
        return stats