# Energy Load Profile Generator

A comprehensive Python project to generate detailed energy load profiles by correlating weather data and device usage patterns. This tool leverages multi-source weather APIs, local caching, and sophisticated device modeling to create accurate energy consumption profiles for residential and commercial applications.

## Features

- **Multi-Source Weather Data:** Fetch historical and real-time weather data from multiple APIs, including Open-Meteo (free, 1940-present) and WeatherAPI (1-year limit)
- **Intelligent Caching:** Store weather data locally in SQLite for faster access, reduced API calls, and offline operation
- **Advanced Device Modeling:** Simulate energy consumption for various devices based on weather conditions, daily patterns, and seasonal variations
- **Comprehensive Analysis:** Analyze temporal patterns, device breakdowns, weather correlations, and peak consumption events
- **Flexible Configuration:** Configure device settings, weather sources, and output formats via `config.yaml`
- **Multiple Export Formats:** Save results in CSV and XLSX formats with detailed analysis summaries and statistics
- **Visualization:** Generate plots for load profiles, daily patterns, temperature correlations, and device breakdowns
- **German Cities Support:** Built-in coordinates for 50+ German cities with intelligent location resolution

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KyleDerZweite/Energy-Load-Profile-Generator.git
   cd Energy-Load-Profile-Generator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the project:**
   The first run will create a default `config.yaml` file. Update it with your API keys and preferences:
   ```bash
   python main.py --list-devices  # This will create config.yaml if it doesn't exist
   ```

## Quick Start

Generate a basic energy load profile for Berlin for January 2024:

```bash
python main.py --location "Berlin, Germany" --start-date 2024-01-01 --end-date 2024-01-31
```

## Usage

### Command Line Interface

The main script provides a comprehensive command-line interface with the following options:

```bash
python main.py [OPTIONS]
```

### Required Arguments

- `--location, -l`: Location for weather data (e.g., "Berlin, Germany")
- `--start-date, -s`: Start date in YYYY-MM-DD format
- `--end-date, -e`: End date in YYYY-MM-DD format

### Optional Arguments

- `--config, -c`: Configuration file path (default: config.yaml)
- `--devices, -d`: Specific devices to include (overrides config)
- `--output-dir, -o`: Output directory for results (default: output)
- `--format, -f`: Output format - csv, xlsx, or both (overrides config)
- `--no-plots`: Skip plot generation
- `--force-refresh`: Force refresh weather data (ignore cache)
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress console output

### Information Commands

- `--list-devices`: List available devices and their configurations
- `--list-locations`: List priority German cities
- `--db-stats`: Show weather database statistics

### Usage Examples

#### Basic Examples

```bash
# Generate load profile for Berlin for one month
python main.py --location "Berlin, Germany" --start-date 2024-01-01 --end-date 2024-01-31

# Generate for Munich with specific output directory
python main.py -l "München, Germany" -s 2024-06-01 -e 2024-06-30 -o ./munich_results

# Generate for Hamburg with CSV output only
python main.py -l "Hamburg, Germany" -s 2024-03-01 -e 2024-03-31 --format csv

# Generate for Frankfurt without plots (faster processing)
python main.py -l "Frankfurt, Germany" -s 2024-01-01 -e 2024-03-31 --no-plots
```

#### Advanced Usage

```bash
# Generate with specific devices only
python main.py -l "Stuttgart, Germany" -s 2024-01-01 -e 2024-01-31 \
  --devices heater air_conditioner lighting

# Force refresh weather data (ignore cache)
python main.py -l "Köln, Germany" -s 2023-01-01 -e 2023-12-31 --force-refresh

# Generate full year with verbose logging
python main.py -l "Leipzig, Germany" -s 2024-01-01 -e 2024-12-31 \
  --verbose --output-dir ./annual_analysis

# Generate in quiet mode (no console output)
python main.py -l "Düsseldorf, Germany" -s 2024-01-01 -e 2024-01-07 --quiet
```

#### Information Commands

```bash
# List all available devices and their configurations
python main.py --list-devices

# Show priority German cities
python main.py --list-locations

# Display weather database statistics
python main.py --db-stats
```

#### Pattern Optimization

```bash
# Run pattern optimization
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Berlin, Germany"

# Use optimized patterns in main system
python main.py --location "Berlin, Germany" --start-date 2024-01-01 --end-date 2024-01-31 --use-optimized

# Run optimization with custom config
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Berlin, Germany" --optimization-config my_optimization.yaml

# Verbose optimization
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Berlin, Germany" --verbose
```


### Output Structure

The generator creates the following output structure:

```
output/
├── energy_load_profile_Berlin_Germany_2024-01-01_to_2024-01-31_20250603_084216.csv
├── energy_load_profile_Berlin_Germany_2024-01-01_to_2024-01-31_20250603_084216.xlsx
├── energy_load_profile_Berlin_Germany_2024-01-01_to_2024-01-31_20250603_084216_summary.csv
└── plots/
    ├── load_profile.png
    ├── daily_patterns.png
    ├── temperature_correlation.png
    ├── device_breakdown.png
    └── monthly_patterns.png
```

### Excel Export Sheets

When exporting to Excel format, the following sheets are created:

- **Load_Profile**: Complete time-series data with weather and device power consumption
- **Statistics**: Basic statistics including energy totals, averages, and percentiles
- **Hourly_Patterns**: Average consumption patterns by hour of day
- **Monthly_Patterns**: Monthly consumption summaries
- **Device_Breakdown**: Individual device statistics and contribution analysis
- **Weather_Correlation**: Temperature and humidity correlation analysis

## Configuration

### Device Configuration

The `config.yaml` file allows you to configure devices with the following parameters:

```yaml
devices:
  heater:
    base_power: 2000          # Base power consumption (W)
    temp_coefficient: -50     # Power change per degree difference
    comfort_temp: 20          # Target temperature (°C)
    seasonal_factor: 1.2      # Seasonal adjustment factor
    daily_pattern: [...]      # 24-hour usage pattern (0-2.0 multipliers)
    enabled: true             # Enable/disable device
```

### Weather Sources

Configure multiple weather data sources with priority ordering:

```yaml
weather_sources:
  - name: "open_meteo"
    enabled: true
    priority: 1               # Lower numbers = higher priority
    description: "Open-Meteo ERA5 (1940-present, FREE)"
    date_range: ["1940-01-01", "2024-12-31"]
  
  - name: "weatherapi"
    enabled: true
    priority: 2
    description: "WeatherAPI.com (recent data, 1 year limit)"
    date_range: ["2023-01-01", "2024-12-31"]
```

### API Keys

Add your API keys to the configuration:

```yaml
api_keys:
  weatherapi_key: "your_actual_api_key_here"
```

**Note:** Open-Meteo is free and requires no API key, while WeatherAPI requires registration.

## Supported Devices

The generator includes models for the following device types:

- **Heater**: Temperature-dependent heating with seasonal patterns
- **Air Conditioner**: Cooling load based on temperature and humidity
- **Refrigeration**: Constant load with temperature sensitivity
- **General Load**: Base electrical load with daily patterns
- **Lighting**: Time-dependent lighting with seasonal variation
- **Water Heater**: Hot water heating with usage patterns

Each device can be configured with custom power ratings, temperature coefficients, and daily usage patterns.

## Supported Locations

The generator includes built-in coordinates for 50+ German cities including:

- Major cities: Berlin, München, Hamburg, Köln, Frankfurt am Main
- Regional centers: Stuttgart, Düsseldorf, Leipzig, Dresden, Hannover
- Smaller cities: Heidelberg, Regensburg, Würzburg, Göttingen, and more

Custom locations can be specified and will be automatically geocoded.

## Data Sources

### Weather Data

- **Open-Meteo ERA5**: Free historical weather data from 1940 to present
- **WeatherAPI.com**: Recent weather data with 1-year historical limit
- **Local Caching**: SQLite database for efficient data storage and retrieval

### Device Models

Based on realistic residential and commercial device characteristics with:
- Temperature-dependent power consumption
- Daily usage patterns (24-hour profiles)
- Seasonal variations
- Random variations for realistic modeling

## Performance

- **Processing Speed**: ~10,000 records per second for load calculation
- **Memory Usage**: Optimized for large datasets with streaming processing
- **Storage**: Efficient SQLite caching with data compression
- **API Limits**: Intelligent source selection to minimize API calls

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- requests >= 2.28.0
- PyYAML >= 6.0
- xlsxwriter >= 3.0.0
- sqlite3 (built-in)

## Troubleshooting

### Common Issues

1. **No weather data available**: Check your API keys and internet connection
2. **Invalid date format**: Use YYYY-MM-DD format for dates
3. **Missing device configuration**: Run `--list-devices` to see available devices
4. **Large date ranges**: Expect longer processing times for ranges > 1 year

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python main.py --verbose [other options]
```

### Database Issues

Check database statistics and clean up if needed:

```bash
python main.py --db-stats
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions, suggestions, or contributions:
- GitHub Issues: [Create an issue](https://github.com/KyleDerZweite/Energy-Load-Profile-Generator/issues)

## Roadmap

Future enhancements planned:
- Additional weather data sources (DWD, NOAA)
- More device types (heat pumps, solar panels, EV chargers)
- Machine learning-based load forecasting
- Web interface for easier configuration
- Integration with smart meter data
- Export to additional formats (JSON, Parquet)