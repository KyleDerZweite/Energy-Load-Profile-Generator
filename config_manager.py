import yaml
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
import copy

class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = None
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                self.logger.warning(f"Config file {self.config_path} not found, creating default config")
                self.create_default_config()

            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)

            # Validate and set defaults
            self.config = self._validate_and_set_defaults(self.config)

            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.config

        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML config: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise

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
            'default_devices': ['heater', 'air_conditioner', 'general_load', 'refrigeration'],
            'device_quantities': {
                'heater': 1,
                'air_conditioner': 1,
                'general_load': 2,
                'refrigeration': 1,
                'lighting': 3,
                'water_heater': 1
            },
            'output': {
                'format': 'both',
                'include_weather_data': True,
                'include_device_breakdown': True,
                'include_statistics': True,
                'filename_prefix': 'energy_load_profile',
                'add_timestamp': True
            }
        }
        validated_config['load_profile'] = {**load_profile_defaults, **validated_config.get('load_profile', {})}

        # Device configurations
        if 'devices' not in validated_config:
            validated_config['devices'] = self._get_default_device_configs()

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

    def _get_default_device_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default device configurations."""
        return {
            'heater': {
                'base_power': 2000,
                'temp_coefficient': -50,
                'comfort_temp': 20,
                'seasonal_factor': 1.2,
                'daily_pattern': [0.8, 0.6, 0.5, 0.4, 0.4, 0.6, 1.0, 0.9, 0.7, 0.6, 0.6, 0.7,
                                  0.8, 0.8, 0.9, 1.0, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.8],
                'enabled': True
            },
            'air_conditioner': {
                'base_power': 1500,
                'temp_coefficient': 80,
                'comfort_temp': 22,
                'seasonal_factor': 1.0,
                'daily_pattern': [0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3,
                                  1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                'enabled': True
            },
            'refrigeration': {
                'base_power': 800,
                'temp_coefficient': 15,
                'comfort_temp': 15,
                'seasonal_factor': 1.0,
                'daily_pattern': [0.9, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1,
                                  1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.9, 0.9, 0.9],
                'enabled': True
            },
            'general_load': {
                'base_power': 1000,
                'temp_coefficient': 0,
                'comfort_temp': 20,
                'seasonal_factor': 1.0,
                'daily_pattern': [0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.8, 1.0, 0.9, 0.8, 0.8, 0.9,
                                  1.0, 0.9, 0.8, 0.9, 1.2, 1.4, 1.3, 1.1, 0.9, 0.7, 0.6, 0.5],
                'enabled': True
            },
            'lighting': {
                'base_power': 300,
                'temp_coefficient': 0,
                'comfort_temp': 20,
                'seasonal_factor': 1.1,
                'daily_pattern': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.7, 0.5, 0.4, 0.4,
                                  0.4, 0.4, 0.5, 0.7, 1.0, 1.2, 1.3, 1.2, 1.0, 0.8, 0.6, 0.3],
                'enabled': True
            },
            'water_heater': {
                'base_power': 2500,
                'temp_coefficient': -30,
                'comfort_temp': 18,
                'seasonal_factor': 1.0,
                'daily_pattern': [0.3, 0.2, 0.2, 0.2, 0.3, 0.6, 1.2, 1.0, 0.7, 0.5, 0.4, 0.5,
                                  0.6, 0.5, 0.4, 0.5, 0.7, 1.0, 1.2, 1.1, 0.9, 0.7, 0.5, 0.4],
                'enabled': True
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
        """Create a default configuration file."""
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
                'default_devices': ['heater', 'air_conditioner', 'general_load', 'refrigeration'],
                'device_quantities': {
                    'heater': 1,
                    'air_conditioner': 1,
                    'general_load': 2,
                    'refrigeration': 1,
                    'lighting': 3,
                    'water_heater': 1
                },
                'output': {
                    'format': 'both',
                    'include_weather_data': True,
                    'include_device_breakdown': True,
                    'include_statistics': True,
                    'filename_prefix': 'energy_load_profile',
                    'add_timestamp': True
                }
            },
            'devices': self._get_default_device_configs(),
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
            }
        }

        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(default_config, file, default_flow_style=False, indent=2, sort_keys=False)
            self.logger.info(f"Default configuration created at {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to create default config: {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        """Get the loaded configuration."""
        if self.config is None:
            self.load_config()
        return self.config

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get a specific configuration section."""
        config = self.get_config()
        return config.get(section_name, {})

    def get_value(self, *keys) -> Any:
        """Get a nested configuration value."""
        config = self.get_config()
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def validate_device_config(self, device_name: str, device_config: Dict[str, Any]) -> bool:
        """Validate device configuration."""
        required_fields = ['base_power', 'temp_coefficient', 'comfort_temp', 'daily_pattern']

        for field in required_fields:
            if field not in device_config:
                self.logger.error(f"Device '{device_name}' missing required field: {field}")
                return False

        # Validate daily pattern
        daily_pattern = device_config['daily_pattern']
        if not isinstance(daily_pattern, list) or len(daily_pattern) != 24:
            self.logger.error(f"Device '{device_name}' has invalid daily_pattern (must be list of 24 values)")
            return False

        # Validate numeric values
        numeric_fields = ['base_power', 'temp_coefficient', 'comfort_temp']
        for field in numeric_fields:
            if not isinstance(device_config[field], (int, float)):
                self.logger.error(f"Device '{device_name}' field '{field}' must be numeric")
                return False

        return True

    def validate_config(self) -> bool:
        """Validate the entire configuration."""
        config = self.get_config()

        # Validate devices
        devices = config.get('devices', {})
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

        self.logger.info("Configuration validation passed")
        return True

    def get_enabled_devices(self) -> List[str]:
        """Get list of enabled device names."""
        devices = self.get_section('devices')
        return [name for name, config in devices.items() if config.get('enabled', True)]

    def get_device_quantities(self) -> Dict[str, int]:
        """Get device quantities from configuration."""
        return self.get_value('load_profile', 'device_quantities') or {}

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration values and save to file."""
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
                yaml.dump(config, file, default_flow_style=False, indent=2, sort_keys=False)
            self.logger.info(f"Configuration updated and saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save updated config: {e}")
            raise