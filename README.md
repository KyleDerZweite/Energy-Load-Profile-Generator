# Energy Load Profile Generator

A comprehensive Python project to generate detailed energy load profiles by correlating weather data and device usage patterns. This tool leverages multi-source weather APIs, local caching, sophisticated device modeling, and **AI-powered pattern optimization** to create accurate energy consumption profiles for residential and commercial applications.

## 🚀 New Features

- **🧠 AI Pattern Optimization**: Use reinforcement learning to optimize device patterns against real load data
- **📊 Live Monitoring Dashboard**: Real-time web interface to monitor optimization progress
- **🎯 Progressive Evaluation**: Smart training strategy that balances speed and accuracy
- **📈 Pattern Evolution Visualization**: Watch patterns evolve in real-time during optimization

## Features

- **Multi-Source Weather Data**: Fetch historical and real-time weather data from multiple APIs, including Open-Meteo (free, 1940-present), WeatherAPI (1-year limit), and DWD (German Weather Service)
- **Intelligent Caching**: Store weather data locally in SQLite for faster access, reduced API calls, and offline operation
- **Advanced Device Modeling**: Simulate energy consumption for various devices based on weather conditions, daily patterns, and seasonal variations
- **🆕 AI Pattern Optimization**: Optimize device usage patterns using genetic algorithms and real load profile data
- **🆕 Live Training Dashboard**: Monitor optimization progress with real-time charts and statistics
- **Comprehensive Analysis**: Analyze temporal patterns, device breakdowns, weather correlations, and peak consumption events
- **Flexible Configuration**: Configure device settings, weather sources, and output formats via YAML files
- **Multiple Export Formats**: Save results in CSV and XLSX formats with detailed analysis summaries and statistics
- **Visualization**: Generate plots for load profiles, daily patterns, temperature correlations, and device breakdowns
- **Extended Location Support**: Built-in coordinates for 50+ German cities including Bottrop, with intelligent geocoding for any location worldwide

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

3. **Install additional dependencies for pattern optimization:**
   ```bash
   pip install flask plotly scikit-learn scipy
   ```

4. **Configure the project:**
   The first run will create a default `config.yaml` file. Update it with your API keys and preferences:
   ```bash
   python main.py --list-devices  # This will create config.yaml if it doesn't exist
   ```

## Quick Start

### Basic Load Profile Generation

Generate a basic energy load profile for Berlin for January 2024:

```bash
python main.py --location "Berlin, Germany" --start-date 2024-01-01 --end-date 2024-01-31
```

### 🆕 Pattern Optimization Quick Start

Optimize device patterns using real load data:

```bash
# Run pattern optimization with live monitoring
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Bottrop, Germany"

# Use optimized patterns in main system
python main.py --location "Bottrop, Germany" --start-date 2024-01-01 --end-date 2024-01-31 --use-optimized
```

## Usage

### Main Load Profile Generator

#### Command Line Interface

```bash
python main.py [OPTIONS]
```

#### Required Arguments

- `--location, -l`: Location for weather data (e.g., "Berlin, Germany")
- `--start-date, -s`: Start date in YYYY-MM-DD format
- `--end-date, -e`: End date in YYYY-MM-DD format

#### Optional Arguments

- `--config, -c`: Configuration file path (default: config.yaml)
- `--devices, -d`: Specific devices to include (overrides config)
- `--output-dir, -o`: Output directory for results (default: output)
- `--format, -f`: Output format - csv, xlsx, or both (overrides config)
- `--weather-source`: Preferred weather source (open_meteo, weatherapi, dwd)
- `--use-optimized`: Use optimized patterns from optimized_config.yaml
- `--no-plots`: Skip plot generation
- `--force-refresh`: Force refresh weather data (ignore cache)
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress console output

#### Information Commands

- `--list-devices`: List available devices and their configurations
- `--list-locations`: List priority German cities
- `--db-stats`: Show weather database statistics

### 🆕 Pattern Optimizer

#### Command Line Interface

```bash
python pattern_optimizer.py [OPTIONS]
```

#### Required Arguments

- `--training-data`: Path to Excel file with real load data
- `--location`: Location for weather data

#### Optional Arguments

- `--config`: Main configuration file (default: config.yaml)
- `--optimization-config`: Optimization configuration file (default: optimization_config.yaml)
- `--output`: Output path for optimized config (default: optimized_config.yaml)
- `--plots-dir`: Directory for output plots (default: optimization_plots)
- `--full-dataset`: Use full dataset instead of progressive evaluation
- `--no-live-monitor`: Disable live monitoring web interface
- `--verbose`: Enable verbose logging

## 🔧 Pattern Optimization Guide

### Training Data Format

Your Excel file should contain multiple sheets (one per year) with this structure:

| Timestamp | Value |
|-----------|--------|
| 2024-01-01 00:15:00 | 55.2 |
| 2024-01-01 00:30:00 | 54.96 |
| 2024-01-01 00:45:00 | 51.48 |

- **Sheets**: Named by year (e.g., "2018", "2019", "2020", "2023", "2024")
- **Timestamp**: Column A - datetime in any pandas-readable format
- **Value**: Column B - power consumption in kW or W
- **Interval**: 15-minute intervals (system will resample if different)
- **Timezone**: Configurable (UTC or local)

### Optimization Strategies

#### 🏃‍♂️ Progressive Evaluation (Recommended)

**Best for**: Balancing speed and accuracy

**Performance Comparison:**

| Strategy | Records Used | Time per Gen | Total Time | Accuracy | Memory Usage |
|----------|-------------|-------------|------------|----------|-------------|
| **Fast** (14 days) | ~56k | ~30s | ~50min | ⭐⭐⭐ | Low |
| **Balanced** (30 days) | ~120k | ~60s | ~100min | ⭐⭐⭐⭐ | Medium |
| **Comprehensive** (60 days) | ~240k | ~120s | ~200min | ⭐⭐⭐⭐⭐ | High |
| **Full Dataset** | 245k+ | ~150s | ~250min | ⭐⭐⭐⭐⭐ | High |

#### Configuration Examples

**Balanced Approach (Recommended):**
```yaml
evaluation:
  use_full_dataset: false
  progressive_evaluation: true
  
  sample_periods:
    - start_month: 12; days: 30  # December
    - start_month: 1; days: 30   # January  
    - start_month: 2; days: 28   # February
    - start_month: 4; days: 30   # April
    - start_month: 7; days: 30   # July
    - start_month: 10; days: 30  # October
  
  # Progressive stages
  # Stage 1 (Gen 1-20): 2 periods (~60k records)
  # Stage 2 (Gen 21-40): 4 periods (~120k records)
  # Stage 3 (Gen 41-60): 6 periods (~180k records)
  # Stage 4 (Gen 61-80): 8 periods (~240k records)
  # Stage 5 (Gen 81-100): All 10 periods (~300k records)
  generations_per_stage: [20, 20, 20, 20, 20]
```

**Fast Approach (Quick Testing):**
```yaml
evaluation:
  use_full_dataset: false
  progressive_evaluation: true
  
  sample_periods:
    - start_month: 1; days: 14   # Winter
    - start_month: 4; days: 14   # Spring
    - start_month: 7; days: 14   # Summer
    - start_month: 10; days: 14  # Autumn
  
  generations_per_stage: [25, 25, 25, 25]
```

**Comprehensive Approach (Maximum Accuracy):**
```yaml
evaluation:
  use_full_dataset: false
  progressive_evaluation: true
  
  sample_periods:
    - start_month: 12; days: 60  # Dec-Jan (winter)
    - start_month: 3; days: 60   # Mar-Apr (spring)
    - start_month: 6; days: 60   # Jun-Jul (summer)
    - start_month: 9; days: 60   # Sep-Oct (autumn)
  
  generations_per_stage: [30, 30, 40]
```

### 📊 Live Monitoring Dashboard

The optimization includes a real-time web dashboard that automatically opens at `http://localhost:5000`:

**Features:**
- 📈 **Real-time Progress**: Current generation, best score, average score, progress percentage
- 📊 **Live Charts**: Score evolution, population distribution, pattern evolution
- 🎯 **Pattern Comparison**: Original vs optimized patterns for each device
- 📋 **Statistics**: Records used, convergence tracking, stage information

**Dashboard Elements:**
- **Score Progress**: Watch the optimization score improve over generations
- **Population Distribution**: See the spread of fitness scores in the current population
- **Pattern Evolution**: Real-time visualization of how device patterns change
- **Progress Tracking**: Visual progress bar and completion percentage

### Usage Examples

#### Basic Examples

```bash
# Generate load profile for Berlin for one month
python main.py --location "Berlin, Germany" --start-date 2024-01-01 --end-date 2024-01-31

# Generate for Munich with specific weather source
python main.py -l "München, Germany" -s 2024-06-01 -e 2024-06-30 --weather-source dwd

# Generate for Bottrop (newly supported city)
python main.py -l "Bottrop, Germany" -s 2024-01-01 -e 2024-01-31 -o ./bottrop_results

# Use optimized patterns
python main.py -l "Hamburg, Germany" -s 2024-03-01 -e 2024-03-31 --use-optimized
```

#### 🆕 Pattern Optimization Examples

```bash
# Basic optimization with live monitoring
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Bottrop, Germany"

# Fast optimization for testing
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Berlin, Germany" \
  --optimization-config fast_optimization.yaml

# Comprehensive optimization with full dataset
python pattern_optimizer.py --training-data load_profiles.xlsx --location "München, Germany" \
  --full-dataset --verbose

# Optimization without live monitoring (headless)
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Stuttgart, Germany" \
  --no-live-monitor

# Custom output paths
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Köln, Germany" \
  --output my_optimized_config.yaml --plots-dir my_optimization_plots
```

#### Advanced Usage

```bash
# Generate with specific devices only
python main.py -l "Stuttgart, Germany" -s 2024-01-01 -e 2024-01-31 \
  --devices heater air_conditioner lighting

# Force refresh weather data and use specific source
python main.py -l "Köln, Germany" -s 2023-01-01 -e 2023-12-31 \
  --force-refresh --weather-source open_meteo

# Generate full year with verbose logging
python main.py -l "Leipzig, Germany" -s 2024-01-01 -e 2024-12-31 \
  --verbose --output-dir ./annual_analysis

# Optimization workflow
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Dresden, Germany"
python main.py -l "Dresden, Germany" -s 2024-01-01 -e 2024-01-31 --use-optimized
```

### Output Structure

#### Main Generator Output

```
output/
├── energy_load_profile_Berlin_Germany_2024-01-01_to_2024-01-31_dwd_20250603_215201.csv
├── energy_load_profile_Berlin_Germany_2024-01-01_to_2024-01-31_dwd_20250603_215201.xlsx
├── energy_load_profile_Berlin_Germany_2024-01-01_to_2024-01-31_dwd_20250603_215201_summary.csv
└── plots/
    ├── load_profile.png
    ├── daily_patterns.png
    ├── temperature_correlation.png
    ├── device_breakdown.png
    └── monthly_patterns.png
```

#### 🆕 Optimization Output

```
optimization_plots/
├── pattern_comparison.png       # Original vs optimized patterns
├── optimization_progress.png    # Score evolution with stage markers
└── pattern_evolution.png       # Pattern changes over time

optimized_config.yaml           # Optimized configuration file
optimization_config.yaml        # Optimization settings
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

### Main Configuration (`config.yaml`)

#### Device Configuration

```yaml
devices:
  heater:
    peak_power: 2000             # Peak power consumption (W) - auto-scaled based on real data
    temp_coefficient: -50        # Power change per degree difference
    comfort_temp: 20             # Target temperature (°C)
    seasonal_factor: 1.2         # Seasonal adjustment factor
    daily_pattern: [...]         # 96 values (15-min intervals) for 24-hour pattern
    fixed_pattern: false         # Set to true to prevent optimization
    enabled: true                # Enable/disable device
```

#### Weather Sources

```yaml
weather_sources:
  - name: "dwd"
    enabled: true
    priority: 1                  # Highest priority for German locations
    description: "Deutscher Wetterdienst (German Weather Service)"
    
  - name: "open_meteo"
    enabled: true
    priority: 2
    description: "Open-Meteo ERA5 (1940-present, FREE)"
    
  - name: "weatherapi"
    enabled: true
    priority: 3
    description: "WeatherAPI.com (recent data, 1 year limit)"
```

### 🆕 Optimization Configuration (`optimization_config.yaml`)

#### Training Data Settings

```yaml
training_data:
  input_file: "load_profiles.xlsx"
  timestamp_column: "Timestamp"    # Column A in your Excel
  value_column: "Value"           # Column B in your Excel
  value_unit: "kW"               # "kW" or "W"
  timezone: "UTC"                # "UTC" or "local"
  years_to_use: [2018, 2019, 2020, 2023, 2024]
```

#### Evaluation Strategy

```yaml
evaluation:
  use_full_dataset: false        # true = use all data, false = progressive
  progressive_evaluation: true   # Smart staged evaluation
  
  sample_periods:               # Define seasonal sample periods
    - start_month: 1
      days: 30
    - start_month: 7
      days: 30
  
  generations_per_stage: [25, 25, 25, 25]  # Generations per evaluation stage
```

#### Device Constraints

```yaml
device_constraints:
  air_conditioner:
    seasonal_percentages:        # Percentage of total load by season
      winter: 0.05              # 5% in winter
      summer: 0.50              # 50% in summer (your requirement)
    allow_optimization: true     # Allow AI to optimize this device
    
  refrigeration:
    year_round_percentage: 0.10
    allow_optimization: false    # Keep this device pattern fixed
```

#### Genetic Algorithm Settings

```yaml
optimization:
  algorithm: "genetic"
  population_size: 30           # Number of pattern combinations
  generations: 100              # Number of optimization iterations
  mutation_rate: 0.1           # How much patterns change
  crossover_rate: 0.8          # How often good patterns combine
```

### API Keys

Add your API keys to the configuration:

```yaml
api_keys:
  weatherapi_key: "your_actual_api_key_here"
```

**Note:** Open-Meteo and DWD are free and require no API keys.

## Supported Devices

The generator includes models for the following device types:

- **Heater**: Temperature-dependent heating with seasonal patterns
- **Air Conditioner**: Cooling load based on temperature and humidity
- **Refrigeration**: Constant load with temperature sensitivity
- **General Load**: Base electrical load with daily patterns
- **Lighting**: Time-dependent lighting with seasonal variation
- **Water Heater**: Hot water heating with usage patterns

Each device can be configured with custom power ratings, temperature coefficients, daily usage patterns, and optimization constraints.

## Supported Locations

### German Cities (Built-in Coordinates)

The generator includes built-in coordinates and weather stations for 50+ German cities including:

- **Major cities**: Berlin, München, Hamburg, Köln, Frankfurt am Main
- **Regional centers**: Stuttgart, Düsseldorf, Leipzig, Dresden, Hannover
- **Industrial cities**: Dortmund, Essen, Bochum, Duisburg, **Bottrop**
- **Smaller cities**: Heidelberg, Regensburg, Würzburg, Göttingen, and more

### Global Support

- **Automatic Geocoding**: Any city worldwide can be specified and will be automatically geocoded
- **Multiple Weather Sources**: Different sources provide different geographic coverage
- **Intelligent Fallback**: System automatically finds the best weather source for each location

## Data Sources

### Weather Data

- **DWD (Deutscher Wetterdienst)**: German national weather service (free, German locations)
- **Open-Meteo ERA5**: Free historical weather data from 1940 to present (global)
- **WeatherAPI.com**: Recent weather data with 1-year historical limit (global)
- **Local Caching**: SQLite database for efficient data storage and retrieval

### 🆕 Load Profile Data

- **Excel Import**: Multi-sheet Excel files with yearly data
- **Flexible Formats**: Supports various timestamp and unit formats
- **Automatic Scaling**: Device peak powers automatically scaled to match real data magnitude
- **Quality Control**: Automatic data validation and cleaning

### Device Models

Based on realistic residential and commercial device characteristics with:
- Temperature-dependent power consumption
- Daily usage patterns (96 x 15-minute intervals)
- Seasonal variations
- 🆕 **AI-optimizable patterns** trained on real load data
- Random variations for realistic modeling

## Performance

### Main Generator
- **Processing Speed**: ~10,000 records per second for load calculation
- **Memory Usage**: Optimized for large datasets with streaming processing
- **Storage**: Efficient SQLite caching with data compression
- **API Limits**: Intelligent source selection to minimize API calls

### 🆕 Pattern Optimization
- **Progressive Training**: Starts fast, becomes more accurate over time
- **Memory Efficient**: Caches weather data to avoid repeated API calls
- **Live Monitoring**: Real-time web dashboard with minimal performance impact
- **Scalable**: Handles datasets from thousands to millions of records

## Dependencies

### Core Dependencies
```
pandas >= 1.5.0
numpy >= 1.21.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
requests >= 2.28.0
PyYAML >= 6.0
xlsxwriter >= 3.0.0
sqlite3 (built-in)
```

### 🆕 Optimization Dependencies
```
flask >= 2.0.0
plotly >= 5.0.0
scikit-learn >= 1.0.0
scipy >= 1.7.0
webbrowser (built-in)
threading (built-in)
```

## Troubleshooting

### Common Issues

1. **No weather data available**: Check your API keys and internet connection
2. **Invalid date format**: Use YYYY-MM-DD format for dates
3. **Missing device configuration**: Run `--list-devices` to see available devices
4. **Large date ranges**: Expect longer processing times for ranges > 1 year
5. **🆕 Infinite optimization scores**: Check timezone alignment between training data and weather data
6. **🆕 Live monitoring not opening**: Check if port 5000 is available, or try a different port

### 🆕 Optimization Troubleshooting

**Training Data Issues:**
```bash
# Check your Excel file structure
python -c "import pandas as pd; print(pd.ExcelFile('load_profiles.xlsx').sheet_names)"

# Verify data format
python -c "import pandas as pd; df = pd.read_excel('load_profiles.xlsx', sheet_name='2024'); print(df.head())"
```

**Performance Issues:**
```bash
# Use smaller sample sizes for testing
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Berlin, Germany" \
  --optimization-config fast_optimization.yaml

# Run without live monitoring to save resources
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Berlin, Germany" \
  --no-live-monitor
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Main generator
python main.py --verbose [other options]

# Pattern optimizer
python pattern_optimizer.py --verbose [other options]
```

**Note**: Verbose mode now filters out excessive matplotlib/PIL debug messages for cleaner output.

### Database Issues

Check database statistics and clean up if needed:

```bash
python main.py --db-stats
```

## 🆕 Optimization Workflow

### 1. Prepare Training Data
- Export your real load data to Excel format
- Organize by year in separate sheets
- Ensure 15-minute intervals and consistent units

### 2. Run Pattern Optimization
```bash
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Your City, Germany"
```

### 3. Monitor Progress
- Web dashboard opens automatically at `http://localhost:5000`
- Watch real-time optimization progress
- Observe pattern evolution

### 4. Use Optimized Patterns
```bash
python main.py --location "Your City, Germany" --start-date 2024-01-01 --end-date 2024-01-31 --use-optimized
```

### 5. Compare Results
- Original vs optimized load profiles
- Pattern comparison plots
- Performance metrics

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### 🆕 Areas for Contribution
- Additional weather data sources
- New device models
- Optimization algorithm improvements
- Web interface enhancements
- Performance optimizations

## Contact

For questions, suggestions, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/KyleDerZweite/Energy-Load-Profile-Generator/issues)
- **Developer**: KyleDerZweite

## Roadmap

### Upcoming Features
- 🔄 **Real-time Data Integration**: Live data feeds from smart meters
- 🌐 **Enhanced Web Interface**: Full web-based configuration and monitoring
- 🤖 **Advanced AI Models**: Neural networks and deep learning approaches
- 📱 **Mobile Dashboard**: Mobile-responsive monitoring interface
- 🔌 **IoT Integration**: Direct integration with smart home systems

### Future Enhancements
- Additional weather data sources (NOAA, regional services)
- More device types (heat pumps, solar panels, EV chargers, battery storage)
- Machine learning-based load forecasting
- Advanced optimization algorithms (PSO, differential evolution)
- Export to additional formats (JSON, Parquet, InfluxDB)
- Multi-location optimization
- Seasonal pattern learning
- Integration with energy market data

---

**Last Updated**: 2025-06-03 21:52:01 UTC by KyleDerZweite