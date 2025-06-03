# Energy Load Profile Generator

A Python project to generate detailed energy load profiles by correlating weather data and device usage patterns. This tool leverages multi-source weather APIs and local caching for accurate and fast data processing.

## Features

- **Multi-Source Weather Data:** Fetch historical and real-time weather data from multiple APIs, including Open-Meteo and WeatherAPI.
- **Local Caching:** Store weather data locally in SQLite for faster access and reduced API calls.
- **Device Load Simulation:** Simulate energy consumption for various devices based on weather conditions and usage patterns.
- **Comprehensive Analysis:** Analyze temporal patterns, device breakdowns, and weather correlations.
- **Customizable:** Configure device settings, weather sources, and output formats via `config.yaml`.
- **Export Formats:** Save results in CSV and XLSX formats, including analysis summaries.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/USERNAME/energy-load-profile-generator.git
   cd energy-load-profile-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the project:
   Update `config.yaml` with your API keys and preferences.

## Usage

Run the main script to generate load profiles:
```bash
python main.py
```

## Configuration

Customize `config.yaml` to:
- Add API keys (e.g., WeatherAPI).
- Enable/disable weather sources.
- Configure devices and quantities.
- Set output preferences (CSV/XLSX).

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or contributions, feel free to open an issue or contact me via GitHub.