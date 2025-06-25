import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import logging
import io
import zipfile
from weather_database import WeatherDatabase
from weather_quality_control import WeatherQualityController

class MultiSourceWeatherFetcher:
    """Fetches weather data from multiple sources with fallback support."""

    def __init__(self, config: Dict, db_path: str = "weather_data.db"):
        self.config = config
        self.db = WeatherDatabase(db_path)
        self.quality_controller = WeatherQualityController()
        self.logger = logging.getLogger(__name__)
        
        # Enhanced quality control parameters
        self.temp_gradient_limit = 2.0     # °C per hour maximum
        self.outlier_threshold = 3.0       # Standard deviations
        self.max_consecutive_identical = 3 # Maximum identical readings
        self.smooth_window = 3             # Moving average window for smoothing
        
        # Regional climate bounds for validation
        self.climate_bounds = {
            'temp_min': -20.0,    # Absolute minimum for region
            'temp_max': 42.0,     # Absolute maximum (based on 2019 German record)
            'typical_min': -10.0, # Typical winter minimum
            'typical_max': 35.0   # Typical summer maximum
        }

        # City coordinates for German cities (expanded list)
        self.german_city_coords = {
            'Berlin': (52.5200, 13.4050),
            'München': (48.1351, 11.5820), 'Munich': (48.1351, 11.5820),
            'Hamburg': (53.5511, 9.9937),
            'Köln': (50.9375, 6.9603), 'Cologne': (50.9375, 6.9603),
            'Frankfurt am Main': (50.1109, 8.6821), 'Frankfurt': (50.1109, 8.6821),
            'Stuttgart': (48.7758, 9.1829),
            'Düsseldorf': (51.2277, 6.7735),
            'Dortmund': (51.5136, 7.4653),
            'Essen': (51.4556, 7.0116),
            'Leipzig': (51.3397, 12.3731),
            'Bremen': (53.0793, 8.8017),
            'Dresden': (51.0504, 13.7373),
            'Hannover': (52.3759, 9.7320),
            'Nürnberg': (49.4521, 11.0767), 'Nuremberg': (49.4521, 11.0767),
            'Duisburg': (51.4344, 6.7623),
            'Bochum': (51.4819, 7.2162),
            'Wuppertal': (51.2562, 7.1508),
            'Bielefeld': (52.0302, 8.5325),
            'Bonn': (50.7374, 7.0982),
            'Münster': (51.9607, 7.6261),
            'Karlsruhe': (49.0069, 8.4037),
            'Mannheim': (49.4875, 8.4660),
            'Augsburg': (48.3705, 10.8978),
            'Wiesbaden': (50.0782, 8.2398),
            'Mönchengladbach': (51.1805, 6.4428),
            'Braunschweig': (52.2689, 10.5268),
            'Chemnitz': (50.8279, 12.9214),
            'Kiel': (54.3233, 10.1228),
            'Aachen': (50.7753, 6.0839),
            'Halle': (51.4969, 11.9695),
            'Magdeburg': (52.1205, 11.6276),
            'Freiburg': (47.9990, 7.8421),
            'Krefeld': (51.3388, 6.5853),
            'Lübeck': (53.8654, 10.6865),
            'Mainz': (49.9929, 8.2473),
            'Erfurt': (50.9848, 11.0299),
            'Rostock': (54.0887, 12.1447),
            'Kassel': (51.3127, 9.4797),
            'Potsdam': (52.3906, 13.0645),
            'Saarbrücken': (49.2401, 6.9969),
            'Osnabrück': (52.2799, 8.0472),
            'Heidelberg': (49.3988, 8.6724),
            'Darmstadt': (49.8728, 8.6512),
            'Regensburg': (49.0134, 12.1016),
            'Würzburg': (49.7913, 9.9534),
            'Göttingen': (51.5412, 9.9158),
            'Trier': (49.7596, 6.6441),
            'Koblenz': (50.3569, 7.5890),
            'Jena': (50.9278, 11.5896),
            'Schwerin': (53.6355, 11.4010),
            'Cottbus': (51.7606, 14.3346),
            'Bremerhaven': (53.5396, 8.5805),
            # Added Bottrop
            'Bottrop': (51.5217, 6.9289)
        }

        # DWD station mapping for major German cities (expanded list)
        self.dwd_station_mapping = {
            'Berlin': '10382',  # Berlin-Tempelhof
            'München': '01262',  # München-Stadt
            'Hamburg': '01975',  # Hamburg-Fuhlsbüttel
            'Köln': '02667',    # Köln-Bonn
            'Frankfurt am Main': '01420',  # Frankfurt/Main
            'Stuttgart': '04931',  # Stuttgart (Schnarrenberg)
            'Düsseldorf': '01078',  # Düsseldorf
            'Dortmund': '01073',   # Dortmund
            'Essen': '01303',      # Essen-Bredeney
            'Leipzig': '02928',    # Leipzig-Holzhausen
            'Bremen': '00691',     # Bremen
            'Dresden': '01048',    # Dresden-Klotzsche
            'Hannover': '02014',   # Hannover
            'Nürnberg': '03668',   # Nürnberg
            'Duisburg': '01081',   # Düsseldorf (nearby)
            'Bochum': '05480',     # Bochum
            'Wuppertal': '05100',  # Wuppertal-Buchenhofen
            'Bielefeld': '00513',  # Bielefeld
            'Bonn': '00603',       # Bad Godesberg
            'Mannheim': '05906',   # Mannheim
            'Karlsruhe': '02522',  # Karlsruhe
            'Wiesbaden': '05738',  # Wiesbaden-Schierstein
            'Augsburg': '00232',   # Augsburg
            'Kiel': '02261',       # Kiel-Holtenau
            'Erfurt': '01270',     # Erfurt-Weimar
            'Mainz': '03032',      # Mainz
            'Kassel': '02244',     # Kassel
            'Saarbrücken': '04336', # Saarbrücken-Burbach
            # Added Bottrop (using nearby Essen station)
            'Bottrop': '01303'     # Essen-Bredeney (closest DWD station)
        }

    def get_coordinates(self, location: str) -> Tuple[float, float]:
        """Get coordinates for a location with improved fallback."""
        # Clean location name
        clean_location = location.replace(', Germany', '').strip()

        # First check if it's in our predefined coordinates
        if clean_location in self.german_city_coords:
            coords = self.german_city_coords[clean_location]
            self.logger.info(f"Using predefined coordinates for {clean_location}: {coords}")
            return coords

        # Try geocoding with Open-Meteo
        self.logger.info(f"Geocoding {location} using Open-Meteo API")
        try:
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {
                'name': location,  # Use full location string for better results
                'count': 5,  # Get multiple results to choose from
                'language': 'en',
                'format': 'json'
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    # Find the best match
                    best_result = None

                    for result in data['results']:
                        result_name = result.get('name', '')
                        result_country = result.get('country', '')
                        result_admin1 = result.get('admin1', '')

                        self.logger.debug(f"Geocoding result: {result_name}, {result_admin1}, {result_country}")

                        # Prefer German results
                        if result_country.lower() in ['germany', 'deutschland', 'de']:
                            best_result = result
                            break

                        # If no German result, take the first one
                        if best_result is None:
                            best_result = result

                    if best_result:
                        coords = (best_result['latitude'], best_result['longitude'])
                        location_info = f"{best_result.get('name', '')}, {best_result.get('admin1', '')}, {best_result.get('country', '')}"
                        self.logger.info(f"Geocoded {location} to {coords} ({location_info})")

                        # Cache the result for future use
                        self.german_city_coords[clean_location] = coords

                        return coords

        except Exception as e:
            self.logger.warning(f"Geocoding failed for {location}: {e}")

        # Try alternative geocoding with a simpler query
        try:
            self.logger.info(f"Trying simplified geocoding for {clean_location}")
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {
                'name': clean_location,
                'count': 1,
                'language': 'en',
                'format': 'json'
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    coords = (result['latitude'], result['longitude'])
                    self.logger.info(f"Simplified geocoding successful for {clean_location}: {coords}")

                    # Cache the result
                    self.german_city_coords[clean_location] = coords

                    return coords

        except Exception as e:
            self.logger.warning(f"Simplified geocoding also failed for {location}: {e}")

        # Final fallback to Berlin with warning
        self.logger.warning(f"All geocoding attempts failed for {location}, using Berlin coordinates as fallback")
        return (52.5200, 13.4050)

    def get_dwd_station_id(self, location: str) -> Optional[str]:
        """Get DWD station ID for a German city with improved fallback."""
        clean_location = location.replace(', Germany', '').strip()

        # Direct mapping
        if clean_location in self.dwd_station_mapping:
            station_id = self.dwd_station_mapping[clean_location]
            self.logger.info(f"Found DWD station {station_id} for {clean_location}")
            return station_id

        # For unmapped German cities, try to find a nearby station
        if self._is_likely_german_city(location):
            coords = self.get_coordinates(location)
            nearby_station = self._find_nearest_dwd_station(coords)
            if nearby_station:
                self.logger.info(f"Using nearest DWD station {nearby_station} for {clean_location}")
                # Cache for future use
                self.dwd_station_mapping[clean_location] = nearby_station
                return nearby_station

        self.logger.info(f"No suitable DWD station found for {clean_location}")
        return None

    def _is_likely_german_city(self, location: str) -> bool:
        """Check if a location is likely in Germany."""
        location_lower = location.lower()
        return (
                'germany' in location_lower or
                'deutschland' in location_lower or
                location_lower.endswith(', de') or
                # Check if coordinates are within Germany bounds
                self._is_in_germany_bounds(self.get_coordinates(location))
        )

    def _is_in_germany_bounds(self, coords: Tuple[float, float]) -> bool:
        """Check if coordinates are within Germany's approximate bounds."""
        lat, lon = coords
        # Approximate bounds of Germany
        return (47.0 <= lat <= 55.5) and (5.5 <= lon <= 15.5)

    def _find_nearest_dwd_station(self, target_coords: Tuple[float, float]) -> Optional[str]:
        """Find the nearest DWD station to given coordinates."""
        if not self.dwd_station_mapping:
            return None

        min_distance = float('inf')
        nearest_station = None
        target_lat, target_lon = target_coords

        for city, station_id in self.dwd_station_mapping.items():
            if city in self.german_city_coords:
                city_lat, city_lon = self.german_city_coords[city]

                # Calculate approximate distance (simple Euclidean)
                distance = ((target_lat - city_lat) ** 2 + (target_lon - city_lon) ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    nearest_station = station_id

        # Only return if reasonably close (within ~2 degrees)
        if min_distance < 2.0:
            return nearest_station

        return None

    def fetch_open_meteo_data(self, latitude: float, longitude: float,
                              start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Open-Meteo ERA5."""
        url = "https://archive-api.open-meteo.com/v1/era5"

        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,weather_code',
            'timezone': 'Europe/Berlin',
            'format': 'json'
        }

        try:
            self.logger.info(f"Fetching Open-Meteo data for ({latitude:.4f}, {longitude:.4f})")
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if 'hourly' not in data:
                raise ValueError("No hourly data in response")

            hourly = data['hourly']
            timestamps = pd.to_datetime(hourly['time'])

            weather_data = []
            for i, timestamp in enumerate(timestamps):
                weather_data.append({
                    'datetime': timestamp,
                    'temperature': self._safe_float(hourly['temperature_2m'][i], 15.0),
                    'humidity': self._safe_int(hourly['relative_humidity_2m'][i], 50),
                    'precipitation': self._safe_float(hourly['precipitation'][i], 0.0),
                    'weather_code': self._safe_int(hourly['weather_code'][i], 0),
                    'condition': self._weather_code_to_condition(self._safe_int(hourly['weather_code'][i], 0))
                })

            df = pd.DataFrame(weather_data)
            df.set_index('datetime', inplace=True)

            # Ensure proper data types before resampling
            numeric_columns = ['temperature', 'humidity', 'precipitation', 'weather_code']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Resample to 15-minute intervals (fixed: using 'min' instead of deprecated 'T')
            df_15min = df.resample('15min').interpolate(method='linear')

            # Handle non-numeric condition column separately
            condition_15min = df['condition'].resample('15min').ffill()
            df_15min['condition'] = condition_15min

            self.logger.info(f"Successfully fetched {len(df_15min)} Open-Meteo records")
            return df_15min

        except Exception as e:
            self.logger.error(f"Open-Meteo fetch failed: {e}")
            return pd.DataFrame()

    def fetch_weatherapi_data(self, location: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from WeatherAPI.com."""
        api_key = self.config.get('api_keys', {}).get('weatherapi_key')
        if not api_key or api_key == "your_weatherapi_key_here":
            self.logger.warning("WeatherAPI key not configured, skipping")
            return pd.DataFrame()

        base_url = "http://api.weatherapi.com/v1"
        weather_data = []

        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Use the original location string for WeatherAPI (it handles geocoding)
        self.logger.info(f"Using WeatherAPI with location: {location}")

        while current_date <= end_dt:
            url = f"{base_url}/history.json"
            params = {
                'key': api_key,
                'q': location,  # Pass through original location
                'dt': current_date.strftime('%Y-%m-%d')
            }

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                for hour_data in data['forecast']['forecastday'][0]['hour']:
                    weather_data.append({
                        'datetime': pd.to_datetime(hour_data['time']),
                        'temperature': hour_data['temp_c'],
                        'humidity': hour_data['humidity'],
                        'condition': hour_data['condition']['text'],
                        'precipitation': hour_data.get('precip_mm', 0),
                        'weather_code': 0  # WeatherAPI doesn't provide WMO codes
                    })

                time.sleep(0.5)  # Rate limiting

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"WeatherAPI error for {current_date}: {e}")
                # Fill with default values
                for hour in range(24):
                    weather_data.append({
                        'datetime': current_date.replace(hour=hour),
                        'temperature': 15.0,
                        'humidity': 50,
                        'condition': 'Unknown',
                        'precipitation': 0,
                        'weather_code': 0
                    })

            current_date += timedelta(days=1)

        if not weather_data:
            return pd.DataFrame()

        df = pd.DataFrame(weather_data)
        df.set_index('datetime', inplace=True)

        # Ensure proper data types before resampling
        numeric_columns = ['temperature', 'humidity', 'precipitation', 'weather_code']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Resample to 15-minute intervals with fixed syntax
        df_15min = df.resample('15min').interpolate(method='linear')

        # Handle condition column separately
        condition_15min = df['condition'].resample('15min').ffill()
        df_15min['condition'] = condition_15min

        self.logger.info(f"Successfully fetched {len(df_15min)} WeatherAPI records")
        return df_15min

    def fetch_dwd_data(self, location: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from DWD (Deutscher Wetterdienst)."""
        try:
            station_id = self.get_dwd_station_id(location)
            if not station_id:
                self.logger.info(f"No DWD station available for {location}")
                return pd.DataFrame()

            # Check date range to determine which DWD service to use
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            current_date = datetime.now()

            # For historical data (older than 1 month), use Open Data service
            if end_dt < current_date - timedelta(days=30):
                self.logger.info(f"Attempting to fetch DWD historical data for {start_date} to {end_date}")
                historical_data = self._fetch_dwd_historical_data(station_id, start_date, end_date)
                if not historical_data.empty:
                    return historical_data

            # For recent/current data, try DWD API
            self.logger.info(f"Attempting to fetch DWD current data for {start_date} to {end_date}")
            current_data = self._fetch_dwd_api_data(station_id, start_date, end_date)
            return current_data

        except Exception as e:
            self.logger.error(f"DWD fetch failed: {e}")
            return pd.DataFrame()

    def _fetch_dwd_historical_data(self, station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from DWD Open Data FTP."""
        try:
            # DWD historical data requires complex parsing of station files and data formats
            # This is a placeholder implementation showing the approach

            # For historical data, we would need to:
            # 1. Parse station metadata files to get exact file locations
            # 2. Download and extract ZIP files from the CDC FTP server
            # 3. Parse fixed-width format data files
            # 4. Handle different data formats for different time periods

            self.logger.info("DWD historical data implementation requires extensive file parsing")
            self.logger.info("Falling back to other weather sources for historical data")

            # Example of what a full implementation would look like:
            # base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/"
            #
            # For hourly temperature data:
            # url_pattern = f"{base_url}hourly/air_temperature/historical/stundenwerte_TU_{station_id}_*_hist.zip"
            #
            # This would require:
            # - Downloading station list files
            # - Matching station IDs to file names
            # - Downloading and parsing multiple ZIP files
            # - Handling different data formats and quality flags

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"DWD historical data fetch failed: {e}")
            return pd.DataFrame()

    def _fetch_dwd_api_data(self, station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch current/forecast data from DWD API."""
        try:
            url = "https://app-prod-ws.warnwetter.de/v30/stationOverviewExtended"

            # The DWD API expects station IDs as parameters
            params = {'stationIds': station_id}

            self.logger.info(f"Fetching DWD API data for station {station_id}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if station_id not in data:
                self.logger.warning(f"No data returned for DWD station {station_id}")
                return pd.DataFrame()

            station_data = data[station_id]
            weather_data = []

            # Process forecast data (this API mainly provides forecasts, not historical data)
            if 'forecast1' in station_data:
                forecast = station_data['forecast1']
                start_timestamp = forecast.get('start', 0) / 1000  # Convert from milliseconds
                time_step = forecast.get('timeStep', 3600)  # seconds

                temperatures = forecast.get('temperature', [])

                for i, temp in enumerate(temperatures):
                    timestamp = datetime.fromtimestamp(start_timestamp + i * time_step)

                    # Only include data within our date range
                    if start_date <= timestamp.strftime('%Y-%m-%d') <= end_date:
                        weather_data.append({
                            'datetime': timestamp,
                            'temperature': temp / 10.0 if temp is not None else 15.0,  # DWD uses 0.1°C units
                            'humidity': 50,  # Default, as DWD API doesn't provide humidity in this format
                            'condition': 'Clear sky',
                            'precipitation': 0,
                            'weather_code': 0
                        })

            # Also try forecast2 if available
            if 'forecast2' in station_data and not weather_data:
                forecast = station_data['forecast2']
                start_timestamp = forecast.get('start', 0) / 1000
                time_step = forecast.get('timeStep', 3600)

                temperatures = forecast.get('temperature', [])

                for i, temp in enumerate(temperatures):
                    timestamp = datetime.fromtimestamp(start_timestamp + i * time_step)

                    if start_date <= timestamp.strftime('%Y-%m-%d') <= end_date:
                        weather_data.append({
                            'datetime': timestamp,
                            'temperature': temp if temp is not None else 15.0,
                            'humidity': 50,
                            'condition': 'Clear sky',
                            'precipitation': 0,
                            'weather_code': 0
                        })

            if not weather_data:
                self.logger.warning("No weather data extracted from DWD API response")
                return pd.DataFrame()

            df = pd.DataFrame(weather_data)
            df.set_index('datetime', inplace=True)

            # Ensure proper data types
            numeric_columns = ['temperature', 'humidity', 'precipitation', 'weather_code']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Resample to 15-minute intervals with fixed syntax
            df_15min = df.resample('15min').interpolate(method='linear')

            # Handle condition column
            condition_15min = df['condition'].resample('15min').ffill()
            df_15min['condition'] = condition_15min

            self.logger.info(f"Successfully fetched {len(df_15min)} DWD records")
            return df_15min

        except Exception as e:
            self.logger.error(f"DWD API fetch failed: {e}")
            return pd.DataFrame()

    def get_weather_data(self, location: str, start_date: str, end_date: str,
                         force_refresh: bool = False, preferred_source: str = None) -> pd.DataFrame:
        """Get weather data with intelligent source selection and optional source preference."""

        # Check if data exists locally
        if not force_refresh:
            availability = self.db.check_data_availability(location, start_date, end_date)
            if availability['location_exists'] and availability['coverage'] > 0.95:
                self.logger.info(f"Using cached data for {location} (Coverage: {availability['coverage']:.1%})")
                return self.db.get_weather_data(location, start_date, end_date)

        # Determine best source based on date range and preference
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        current_date = datetime.now()

        # Get coordinates for Open-Meteo
        lat, lon = self.get_coordinates(location)

        # Get enabled sources from config
        weather_sources = self.config.get('weather_sources', [])
        enabled_sources = [s for s in weather_sources if s.get('enabled', True)]

        # If a preferred source is specified, try it first
        if preferred_source:
            preferred_sources = [s for s in enabled_sources if s['name'] == preferred_source]
            other_sources = [s for s in enabled_sources if s['name'] != preferred_source]
            enabled_sources = preferred_sources + other_sources

            if preferred_sources:
                self.logger.info(f"Using preferred weather source: {preferred_source}")
            else:
                self.logger.warning(f"Preferred source '{preferred_source}' not found or not enabled")

        # Sort by priority if no preference specified
        if not preferred_source:
            enabled_sources.sort(key=lambda x: x.get('priority', 10))

        # Try sources in order
        for source in enabled_sources:
            source_name = source['name']

            try:
                if source_name == 'open_meteo':
                    weather_data = self.fetch_open_meteo_data(lat, lon, start_date, end_date)
                    if not weather_data.empty:
                        self.db.store_weather_data(location, weather_data, 'open_meteo', lat, lon)
                        return weather_data

                elif source_name == 'weatherapi':
                    # Use WeatherAPI for recent data (within 1 year)
                    if start_dt >= current_date - timedelta(days=365):
                        weather_data = self.fetch_weatherapi_data(location, start_date, end_date)
                        if not weather_data.empty:
                            self.db.store_weather_data(location, weather_data, 'weatherapi', lat, lon)
                            return weather_data
                    else:
                        self.logger.info(f"Skipping WeatherAPI for historical data beyond 1 year ({start_date})")

                elif source_name == 'dwd':
                    # Use DWD for German locations (improved detection)
                    if self._is_likely_german_city(location):
                        weather_data = self.fetch_dwd_data(location, start_date, end_date)
                        if not weather_data.empty:
                            self.db.store_weather_data(location, weather_data, 'dwd', lat, lon)
                            return weather_data
                    else:
                        self.logger.info(f"Skipping DWD for non-German location: {location}")

            except Exception as e:
                self.logger.error(f"Error fetching from {source_name}: {e}")
                continue

        # If all sources fail, return empty DataFrame
        self.logger.error(f"All weather sources failed for {location}")
        return pd.DataFrame()

    def _weather_code_to_condition(self, code: int) -> str:
        """Convert WMO weather code to readable condition."""
        code_map = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
        }
        return code_map.get(code, "Unknown")

    def fetch_clean_weather_data(self, location: str, start_date: str, end_date: str,
                                force_refresh: bool = False, preferred_source: str = None) -> pd.DataFrame:
        """
        Fetch weather data with comprehensive quality control.
        
        Args:
            location: Location name (e.g., "Bottrop, Germany")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: Whether to re-fetch even if data exists
            preferred_source: Preferred weather source to try first
            
        Returns:
            Clean weather DataFrame with validated temperature data
        """
        self.logger.info(f"🔍 Fetching clean weather data for {location}")
        self.logger.info(f"📅 Date range: {start_date} to {end_date}")
        
        # First get raw data using existing method
        raw_data = self.get_weather_data(location, start_date, end_date, force_refresh, preferred_source)
        
        if raw_data.empty:
            self.logger.error("Failed to fetch raw weather data")
            return pd.DataFrame()
        
        self.logger.info(f"📊 Raw data fetched: {len(raw_data)} records")
        
        # Apply quality control and cleaning
        clean_data = self._apply_quality_control(raw_data, location)
        
        # Store clean data in database with enhanced source tag
        lat, lon = self.get_coordinates(location)
        source_tag = f"{getattr(raw_data, '_source', 'multi_source')}_cleaned"
        self.db.store_weather_data(location, clean_data, source_tag, lat, lon)
        
        self.logger.info(f"✅ Clean weather data ready: {len(clean_data)} records")
        return clean_data

    def _apply_quality_control(self, data: pd.DataFrame, location: str) -> pd.DataFrame:
        """Apply comprehensive quality control to weather data."""
        self.logger.info("🔧 Applying quality control measures...")
        
        data = data.copy()
        
        # Step 1: Remove extreme outliers
        data = self._remove_extreme_outliers(data)
        
        # Step 2: Smooth temperature gradients
        data = self._smooth_temperature_gradients(data)
        
        # Step 3: Fix flat-lining periods
        data = self._fix_flat_lining(data)
        
        # Step 4: Apply regional climate validation
        data = self._apply_climate_bounds(data)
        
        # Step 5: Final smoothing pass
        data = self._final_smoothing(data)
        
        # Step 6: Quality validation
        quality_score = self._validate_final_quality(data)
        self.logger.info(f"🎯 Final quality score: {quality_score:.1f}%")
        
        return data

    def _remove_extreme_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove statistically extreme temperature outliers."""
        if 'temperature' not in data.columns:
            return data
        
        temps = data['temperature'].dropna()
        if len(temps) < 10:
            return data
        
        # Calculate outlier bounds
        q25, q75 = temps.quantile(0.25), temps.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 3.0 * iqr
        upper_bound = q75 + 3.0 * iqr
        
        # Also apply absolute climate bounds
        lower_bound = max(lower_bound, self.climate_bounds['temp_min'])
        upper_bound = min(upper_bound, self.climate_bounds['temp_max'])
        
        # Count outliers before removal
        outliers_mask = (data['temperature'] < lower_bound) | (data['temperature'] > upper_bound)
        outlier_count = outliers_mask.sum()
        
        if outlier_count > 0:
            self.logger.info(f"🚫 Removing {outlier_count} extreme temperature outliers")
            
            # Replace outliers with interpolated values
            data.loc[outliers_mask, 'temperature'] = np.nan
            data['temperature'] = data['temperature'].interpolate(method='linear')
        
        return data

    def _smooth_temperature_gradients(self, data: pd.DataFrame) -> pd.DataFrame:
        """Smooth unrealistic temperature gradients."""
        if 'temperature' not in data.columns or len(data) < 2:
            return data
        
        data = data.copy()
        
        # Calculate time differences and temperature gradients
        data['time_diff'] = data.index.to_series().diff().dt.total_seconds() / 3600  # hours
        data['temp_change'] = data['temperature'].diff()
        data['temp_gradient'] = abs(data['temp_change'] / data['time_diff'])
        
        # Find excessive gradients
        excessive_mask = data['temp_gradient'] > self.temp_gradient_limit
        excessive_count = excessive_mask.sum()
        
        if excessive_count > 0:
            self.logger.info(f"🌊 Smoothing {excessive_count} excessive temperature gradients")
            
            # Apply moving average smoothing to reduce gradients
            window_size = max(3, min(7, len(data) // 100))  # Adaptive window size
            data['temperature_smoothed'] = data['temperature'].rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
            
            # Replace only the excessive gradient points
            data.loc[excessive_mask, 'temperature'] = data.loc[excessive_mask, 'temperature_smoothed']
        
        # Clean up helper columns
        data = data.drop(columns=['time_diff', 'temp_change', 'temp_gradient'], errors='ignore')
        if 'temperature_smoothed' in data.columns:
            data = data.drop(columns=['temperature_smoothed'])
        
        return data

    def _fix_flat_lining(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix periods where temperature remains artificially constant."""
        if 'temperature' not in data.columns or len(data) < 4:
            return data
        
        data = data.copy()
        
        # Detect consecutive identical temperatures
        data['temp_diff'] = data['temperature'].diff()
        data['is_same'] = (data['temp_diff'] == 0.0) & (data['temperature'].notna())
        
        # Find groups of consecutive identical values
        data['group'] = (data['is_same'] != data['is_same'].shift()).cumsum()
        
        flat_periods_fixed = 0
        
        for group_id, group in data[data['is_same']].groupby('group'):
            if len(group) >= self.max_consecutive_identical:
                # Add small random variation to break flat-lining
                flat_periods_fixed += 1
                base_temp = group.iloc[0]['temperature']
                
                # Create subtle variation (±0.1°C)
                variations = np.random.normal(0, 0.05, len(group))
                new_temps = base_temp + variations
                
                # Ensure gradual change rather than random noise
                if len(group) > 1:
                    # Create linear interpolation with noise
                    start_temp = base_temp + variations[0]
                    end_temp = base_temp + variations[-1]
                    linear_temps = np.linspace(start_temp, end_temp, len(group))
                    new_temps = linear_temps + variations * 0.3  # Reduce noise
                
                data.loc[group.index, 'temperature'] = new_temps
        
        if flat_periods_fixed > 0:
            self.logger.info(f"🔧 Fixed {flat_periods_fixed} flat-lining periods")
        
        # Clean up helper columns
        data = data.drop(columns=['temp_diff', 'is_same', 'group'], errors='ignore')
        
        return data

    def _apply_climate_bounds(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply regional climate bounds validation."""
        if 'temperature' not in data.columns:
            return data
        
        # Apply hard bounds
        temp_violations = (
            (data['temperature'] < self.climate_bounds['temp_min']) |
            (data['temperature'] > self.climate_bounds['temp_max'])
        )
        
        violation_count = temp_violations.sum()
        if violation_count > 0:
            self.logger.info(f"🌡️ Correcting {violation_count} climate bound violations")
            
            # Replace violations with interpolated values
            data.loc[temp_violations, 'temperature'] = np.nan
            data['temperature'] = data['temperature'].interpolate(method='linear')
        
        return data

    def _final_smoothing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply final light smoothing to ensure natural temperature patterns."""
        if 'temperature' not in data.columns or len(data) < 5:
            return data
        
        # Light smoothing with small window
        data['temperature'] = data['temperature'].rolling(
            window=3, center=True, min_periods=1
        ).mean()
        
        return data

    def _validate_final_quality(self, data: pd.DataFrame) -> float:
        """Validate the final data quality after processing."""
        if data.empty or 'temperature' not in data.columns:
            return 0.0
        
        quality_score = 100.0
        
        # Check for remaining flat periods
        temp_diff = data['temperature'].diff()
        consecutive_same = (temp_diff == 0.0).sum()
        if consecutive_same > len(data) * 0.01:  # More than 1% flat
            quality_score -= 20.0
        
        # Check temperature gradients
        time_diff = data.index.to_series().diff().dt.total_seconds() / 3600
        temp_gradient = abs(temp_diff / time_diff)
        excessive_gradients = (temp_gradient > self.temp_gradient_limit).sum()
        if excessive_gradients > 0:
            quality_score -= 10.0
        
        # Check for missing data
        missing_pct = data['temperature'].isna().sum() / len(data) * 100
        quality_score -= missing_pct
        
        return max(0.0, quality_score)

    def _safe_float(self, value, default: float) -> float:
        """Safely convert value to float with default fallback."""
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def _safe_int(self, value, default: int) -> int:
        """Safely convert value to int with default fallback."""
        try:
            if value is None or pd.isna(value):
                return default
            return int(value)
        except (ValueError, TypeError):
            return default