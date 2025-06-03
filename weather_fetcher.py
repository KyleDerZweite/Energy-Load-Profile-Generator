import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import logging
from weather_database import WeatherDatabase

class MultiSourceWeatherFetcher:
    """Fetches weather data from multiple sources with fallback support."""
    
    def __init__(self, config: Dict, db_path: str = "weather_data.db"):
        self.config = config
        self.db = WeatherDatabase(db_path)
        self.logger = logging.getLogger(__name__)
        
        # City coordinates for German cities
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
            'Bremerhaven': (53.5396, 8.5805)
        }
    
    def get_coordinates(self, location: str) -> Tuple[float, float]:
        """Get coordinates for a location."""
        # Clean location name
        clean_location = location.replace(', Germany', '').strip()
        
        if clean_location in self.german_city_coords:
            return self.german_city_coords[clean_location]
        
        # Try geocoding with Open-Meteo
        try:
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {'name': clean_location, 'count': 1, 'language': 'en', 'format': 'json'}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    coords = (result['latitude'], result['longitude'])
                    self.logger.info(f"Geocoded {location} to {coords}")
                    return coords
        except Exception as e:
            self.logger.warning(f"Geocoding failed for {location}: {e}")
        
        # Default to Berlin
        self.logger.warning(f"Using Berlin coordinates as fallback for {location}")
        return (52.5200, 13.4050)
    
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
            self.logger.info(f"Fetching Open-Meteo data for ({latitude}, {longitude})")
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
                    'temperature': hourly['temperature_2m'][i] if hourly['temperature_2m'][i] is not None else 15.0,
                    'humidity': hourly['relative_humidity_2m'][i] if hourly['relative_humidity_2m'][i] is not None else 50,
                    'precipitation': hourly['precipitation'][i] if hourly['precipitation'][i] is not None else 0,
                    'weather_code': hourly['weather_code'][i] if hourly['weather_code'][i] is not None else 0,
                    'condition': self._weather_code_to_condition(hourly['weather_code'][i] if hourly['weather_code'][i] is not None else 0)
                })
            
            df = pd.DataFrame(weather_data)
            df.set_index('datetime', inplace=True)
            
            # Resample to 15-minute intervals
            df_15min = df.resample('15T').interpolate(method='linear')
            
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
        
        while current_date <= end_dt:
            url = f"{base_url}/history.json"
            params = {
                'key': api_key,
                'q': location,
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
        df_15min = df.resample('15T').interpolate(method='linear')
        
        self.logger.info(f"Successfully fetched {len(df_15min)} WeatherAPI records")
        return df_15min
    
    def get_weather_data(self, location: str, start_date: str, end_date: str, 
                        force_refresh: bool = False) -> pd.DataFrame:
        """Get weather data with intelligent source selection."""
        
        # Check if data exists locally
        if not force_refresh:
            availability = self.db.check_data_availability(location, start_date, end_date)
            if availability['location_exists'] and availability['coverage'] > 0.95:
                self.logger.info(f"Using cached data for {location} (Coverage: {availability['coverage']:.1%})")
                return self.db.get_weather_data(location, start_date, end_date)
        
        # Determine best source based on date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        current_date = datetime.now()
        
        # Get coordinates
        lat, lon = self.get_coordinates(location)
        
        # Try sources in priority order
        weather_sources = self.config.get('weather_sources', [])
        enabled_sources = [s for s in weather_sources if s.get('enabled', True)]
        enabled_sources.sort(key=lambda x: x.get('priority', 10))
        
        for source in enabled_sources:
            source_name = source['name']
            
            try:
                if source_name == 'open_meteo':
                    # Use Open-Meteo for historical data (especially older than 1 year)
                    if start_dt < current_date - timedelta(days=365) or force_refresh:
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