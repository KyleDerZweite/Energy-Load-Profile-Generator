import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import json
import os
import logging

class WeatherDatabase:
    """Enhanced SQLite database for weather data with multi-source support."""
    
    def __init__(self, db_path: str = "weather_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    def init_database(self):
        """Create all necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Main weather data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    temperature REAL,
                    humidity INTEGER,
                    condition TEXT,
                    precipitation REAL DEFAULT 0,
                    weather_code INTEGER DEFAULT 0,
                    data_source TEXT DEFAULT 'unknown',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(location, datetime, data_source)
                )
            ''')
            
            # Indexes for performance
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_location_datetime 
                ON weather_data(location, datetime)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_data_source 
                ON weather_data(data_source)
            ''')
            
            # Locations tracking table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location TEXT UNIQUE NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    first_data_date TEXT,
                    last_data_date TEXT,
                    total_records INTEGER DEFAULT 0,
                    primary_source TEXT,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Data sources table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT UNIQUE NOT NULL,
                    api_url TEXT,
                    description TEXT,
                    date_range_start TEXT,
                    date_range_end TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    priority INTEGER DEFAULT 10,
                    last_used TEXT,
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0
                )
            ''')
            
            # Insert default data sources
            self._init_data_sources(conn)
    
    def _init_data_sources(self, conn):
        """Initialize default data sources."""
        sources = [
            ('open_meteo', 'https://archive-api.open-meteo.com/v1/era5', 
             'Open-Meteo ERA5 Reanalysis (1940-present)', '1940-01-01', '2024-12-31', 1),
            ('weatherapi', 'http://api.weatherapi.com/v1', 
             'WeatherAPI.com (1 year limit)', '2023-01-01', '2024-12-31', 2),
            ('dwd_cdc', 'https://opendata.dwd.de/climate_environment/CDC/', 
             'German Weather Service Climate Data Center', '1881-01-01', '2024-12-31', 3)
        ]
        
        for source in sources:
            conn.execute('''
                INSERT OR IGNORE INTO data_sources 
                (source_name, api_url, description, date_range_start, date_range_end, priority)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', source)
    
    def store_weather_data(self, location: str, weather_df: pd.DataFrame, 
                          data_source: str = 'unknown', latitude: float = None, 
                          longitude: float = None):
        """Store weather data with source tracking."""
        with sqlite3.connect(self.db_path) as conn:
            records = []
            for idx, row in weather_df.iterrows():
                records.append({
                    'location': location,
                    'datetime': idx.strftime('%Y-%m-%d %H:%M:%S'),
                    'temperature': float(row['temperature']) if pd.notna(row['temperature']) else None,
                    'humidity': int(row.get('humidity', 50)) if pd.notna(row.get('humidity', 50)) else 50,
                    'condition': str(row.get('condition', 'Unknown')),
                    'precipitation': float(row.get('precipitation', 0)) if pd.notna(row.get('precipitation', 0)) else 0,
                    'weather_code': int(row.get('weather_code', 0)) if pd.notna(row.get('weather_code', 0)) else 0,
                    'data_source': data_source
                })
            
            # Insert or replace records
            conn.executemany('''
                INSERT OR REPLACE INTO weather_data 
                (location, datetime, temperature, humidity, condition, precipitation, weather_code, data_source)
                VALUES (:location, :datetime, :temperature, :humidity, :condition, :precipitation, :weather_code, :data_source)
            ''', records)
            
            # Update locations table
            first_date = weather_df.index.min().strftime('%Y-%m-%d')
            last_date = weather_df.index.max().strftime('%Y-%m-%d')
            
            conn.execute('''
                INSERT OR REPLACE INTO locations 
                (location, latitude, longitude, first_data_date, last_data_date, total_records, primary_source, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (location, latitude, longitude, first_date, last_date, len(weather_df), 
                  data_source, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        self.logger.info(f"Stored {len(weather_df)} weather records for {location} from {data_source}")
    
    def get_weather_data(self, location: str, start_date: str, end_date: str, 
                        preferred_source: str = None) -> Optional[pd.DataFrame]:
        """Retrieve weather data with optional source preference."""
        with sqlite3.connect(self.db_path) as conn:
            base_query = '''
                SELECT datetime, temperature, humidity, condition, precipitation, weather_code, data_source
                FROM weather_data
                WHERE location = ? AND datetime BETWEEN ? AND ?
            '''
            
            start_datetime = f"{start_date} 00:00:00"
            end_datetime = f"{end_date} 23:59:59"
            params = [location, start_datetime, end_datetime]
            
            if preferred_source:
                base_query += ' AND data_source = ?'
                params.append(preferred_source)
            
            base_query += ' ORDER BY datetime, data_source'
            
            df = pd.read_sql_query(base_query, conn, params=params)
            
            if df.empty:
                return None
            
            # Remove duplicates, preferring better sources
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.drop_duplicates(subset=['datetime'], keep='first')
            df.set_index('datetime', inplace=True)
            
            return df
    
    def check_data_availability(self, location: str, start_date: str, end_date: str) -> Dict:
        """Enhanced availability check with source information."""
        with sqlite3.connect(self.db_path) as conn:
            # Check location info
            location_query = '''
                SELECT latitude, longitude, first_data_date, last_data_date, total_records, primary_source
                FROM locations WHERE location = ?
            '''
            location_result = conn.execute(location_query, (location,)).fetchone()
            
            if not location_result:
                return {
                    'location_exists': False,
                    'coverage': 0.0,
                    'available_range': None,
                    'sources': []
                }
            
            lat, lon, first_available, last_available, total_records, primary_source = location_result
            
            # Check coverage by source
            coverage_query = '''
                SELECT data_source, COUNT(*) as count
                FROM weather_data
                WHERE location = ? AND datetime BETWEEN ? AND ?
                GROUP BY data_source
            '''
            
            start_datetime = f"{start_date} 00:00:00"
            end_datetime = f"{end_date} 23:59:59"
            
            source_data = conn.execute(coverage_query, (location, start_datetime, end_datetime)).fetchall()
            
            # Calculate expected records
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            expected_records = int((end_dt - start_dt).total_seconds() / 900) + 1  # 15-min intervals
            
            total_available = sum(count for _, count in source_data)
            coverage = min(1.0, total_available / expected_records) if expected_records > 0 else 0
            
            sources = [{'source': source, 'records': count} for source, count in source_data]
            
            return {
                'location_exists': True,
                'coverage': coverage,
                'available_records': total_available,
                'expected_records': expected_records,
                'available_range': (first_available, last_available),
                'primary_source': primary_source,
                'sources': sources,
                'coordinates': (lat, lon) if lat and lon else None
            }
    
    def get_database_stats(self) -> Dict:
        """Comprehensive database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Basic counts
            stats['total_weather_records'] = conn.execute(
                'SELECT COUNT(*) FROM weather_data'
            ).fetchone()[0]
            
            stats['total_locations'] = conn.execute(
                'SELECT COUNT(*) FROM locations'
            ).fetchone()[0]
            
            # Source breakdown
            source_stats = conn.execute('''
                SELECT data_source, COUNT(*) as records
                FROM weather_data
                GROUP BY data_source
                ORDER BY records DESC
            ''').fetchall()
            
            stats['by_source'] = dict(source_stats)
            
            # Date range
            date_range = conn.execute('''
                SELECT MIN(datetime) as earliest, MAX(datetime) as latest
                FROM weather_data
            ''').fetchone()
            
            stats['earliest_date'] = date_range[0]
            stats['latest_date'] = date_range[1]
            
            # Database file size
            if os.path.exists(self.db_path):
                stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            else:
                stats['database_size_mb'] = 0
            
            return stats
    
    def cleanup_old_data(self, location: str = None, older_than_days: int = 365):
        """Clean up old weather data to save space."""
        cutoff_date = (datetime.now() - timedelta(days=older_than_days)).strftime('%Y-%m-%d %H:%M:%S')
        
        with sqlite3.connect(self.db_path) as conn:
            if location:
                deleted = conn.execute('''
                    DELETE FROM weather_data 
                    WHERE location = ? AND datetime < ?
                ''', (location, cutoff_date)).rowcount
            else:
                deleted = conn.execute('''
                    DELETE FROM weather_data 
                    WHERE datetime < ?
                ''', (cutoff_date,)).rowcount
            
            # Update location statistics
            conn.execute('''
                UPDATE locations SET 
                    first_data_date = (
                        SELECT MIN(datetime) FROM weather_data WHERE location = locations.location
                    ),
                    total_records = (
                        SELECT COUNT(*) FROM weather_data WHERE location = locations.location
                    )
            ''')
        
        self.logger.info(f"Cleaned up {deleted} old weather records")
        return deleted
    
    def vacuum_database(self):
        """Optimize database performance."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('VACUUM')
        self.logger.info("Database vacuumed and optimized")