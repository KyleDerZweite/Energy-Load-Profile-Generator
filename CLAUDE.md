# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Energy Load Profile Generator is a comprehensive Python system for generating realistic energy consumption profiles by combining weather data with physics-based device modeling. The system uses AI-powered pattern optimization to align generated profiles with real measured load data, focusing on realism over mathematical precision.

## Architecture

### Core Components

- **Main Entry Points**: 
  - `main.py` - Primary load profile generation with realistic-first approach
  - `pattern_optimizer.py` - AI-powered pattern optimization using genetic algorithms
  - `enhanced_main.py` - Enhanced version with additional realistic features

- **Core Modules**:
  - `device_calculator.py` - Physics-based device load calculations with thermal inertia
  - `weather_fetcher.py` - Multi-source weather data fetching (Open-Meteo, WeatherAPI, DWD)
  - `weather_database.py` - SQLite-based weather data caching
  - `config_manager.py` - YAML configuration management
  - `analysis_export.py` - Data analysis and export (CSV/XLSX)

- **Specialized Components**:
  - `realistic_device_calculator.py` - Enhanced realistic behavior modeling
  - `pattern_smoother.py` - Pattern smoothing and transition optimization
  - `pattern_analysis.py` - Load pattern analysis tools

### Key Design Principles

1. **Realism-First**: All device models use physics-based calculations rather than pure mathematical approaches
2. **Multi-Source Weather**: Intelligent fallback between weather APIs with local caching
3. **AI Optimization**: Genetic algorithms optimize device patterns against real load data
4. **Modular Architecture**: Each component handles a specific responsibility with clear interfaces

## Common Development Commands

### Running the System

```bash
# Basic energy profile generation
python main.py --location "Berlin, Germany" --start-date 2024-01-01 --end-date 2024-01-31

# Pattern optimization (requires training data)
python pattern_optimizer.py --training-data load_profiles.xlsx --location "Berlin, Germany"

# Use optimized patterns
python main.py --location "Berlin, Germany" --start-date 2024-01-01 --end-date 2024-01-31 --use-optimized
```

### Development and Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run realistic pattern testing
python test_realistic_patterns.py

# List available devices and configurations
python main.py --list-devices

# Check database statistics
python main.py --db-stats

# Enable verbose logging for debugging
python main.py --verbose [other options]
```

### Configuration Management

- **Main Config**: `config.yaml` - Device settings, weather sources, analysis parameters
- **Optimization Config**: `optimization_config.yaml` - Genetic algorithm parameters, training data settings
- **Device Patterns**: 96-element arrays representing 15-minute intervals over 24 hours

## Configuration Structure

### Device Configuration
Each device requires:
- `peak_power`: Maximum power consumption (W)
- `temp_coefficient`: Power change per temperature degree
- `comfort_temp`: Target temperature for thermal devices
- `daily_pattern`: 96-element array (15-min intervals)
- `enabled`: Boolean to include/exclude device

### Weather Sources
Priority-ordered list with automatic fallback:
1. Open-Meteo (free, 1940-present)
2. WeatherAPI (requires key, 1-year limit)  
3. DWD (German weather service)

### Optimization Features
- Progressive evaluation: Starts fast, becomes more accurate
- Live web monitoring dashboard at `http://localhost:5000`
- Real-time pattern evolution visualization
- Genetic algorithm with configurable population and mutation rates

## Data Flow

1. **Weather Data**: Fetch from multiple sources → Cache in SQLite → Weather DataFrame
2. **Device Modeling**: Physics-based calculations with thermal inertia → Device load profiles
3. **Load Aggregation**: Combine all device loads → Total load profile
4. **Analysis**: Statistical analysis, correlations, pattern detection
5. **Export**: CSV/XLSX with multiple sheets and visualization plots

## File Naming Conventions

- Output files include location, date range, and timestamp
- Realistic mode indicated in filename: `energy_load_profile_realistic_*`
- Weather source appended when explicitly specified
- Plot files saved in `output/plots/` subdirectory

## Key Patterns

### Error Handling
- Graceful fallback between weather sources
- Comprehensive logging with rotating file handlers
- Input validation for dates, locations, and configurations

### Performance Optimizations
- Weather data caching to minimize API calls
- Vectorized calculations using NumPy/Pandas
- Progressive evaluation for large datasets in optimization

### Realism Features
- Thermal inertia modeling for heating/cooling devices
- Smooth transitions between power states
- Natural variations and noise injection
- Automatic pattern enhancement for realistic behavior

## Testing

The system includes specialized testing for realistic patterns:
- `test_realistic_patterns.py` - Validates physics-based behavior
- Pattern smoothness verification
- Thermal inertia validation
- Transition realism scoring

## Dependencies

Core: pandas, numpy, matplotlib, seaborn, requests, PyYAML, xlsxwriter
Optimization: flask, plotly, scikit-learn, scipy
All dependencies listed in `requirements.txt`

## Future Enhancement Ideas

### AI-Powered Pattern Validation (Future Consideration)

**Concept**: Integrate Ollama-based AI model for intelligent pattern validation and optimization guidance.

**Potential Benefits**:
- **Domain Knowledge Validation**: AI could catch unrealistic patterns (like heating peaking in summer)
- **Semantic Understanding**: Evaluate if device interactions make sense (e.g., HVAC coordination)
- **Pattern Reasonableness**: Flag patterns that are mathematically optimal but physically impossible
- **Multi-objective Optimization**: Balance mathematical fitness with real-world feasibility
- **Automated Quality Assurance**: Catch edge cases human reviewers might miss

**Implementation Considerations**:
- **Additional Infrastructure**: Ollama setup, model management, API integration
- **Reliability Questions**: How much to trust AI vs physics-based validation?
- **Performance Impact**: Additional evaluation step could slow optimization
- **Model Choice**: Which model? How to prompt effectively for energy domain?

**Recommended Approach**:
1. **Current State**: Physics-based validation ✅, Realistic pattern smoothing ✅, Multi-criteria scoring ✅
2. **Phase 1**: Complete current optimization with devices.json archiving
3. **Phase 2**: Add AI as optional validator layer
4. **Phase 3**: Use AI for post-optimization review rather than real-time evaluation

**Integration Points**:
- `pattern_optimizer.py` - Add AI validation step in genetic algorithm evaluation
- `device_calculator.py` - AI sanity checks for device interaction patterns
- `analysis_export.py` - AI-powered anomaly detection in results

**Status**: Concept documented for future implementation. Focus on current physics-based approach first.