"""
Weather Data Quality Control System
==================================

This module implements comprehensive quality control for weather data to detect
and handle unrealistic temperature patterns, interpolation artifacts, and outliers
that can corrupt energy-weather correlation analysis.

Key Features:
- Outlier detection and flagging
- Temperature gradient validation
- Consecutive identical value detection
- Regional climate validation
- Quality scoring and reporting
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QualityIssue:
    """Represents a data quality issue found in weather data."""
    issue_type: str           # 'outlier', 'flat_lining', 'gradient', 'unrealistic'
    severity: str             # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    description: str
    value: float
    expected_range: Tuple[float, float]
    recommendation: str


@dataclass
class QualityReport:
    """Comprehensive quality assessment report for weather data."""
    location: str
    date_range: Tuple[str, str]
    total_records: int
    issues_found: List[QualityIssue]
    quality_score: float      # 0-100, where 100 is perfect
    data_completeness: float  # Percentage of valid data
    temperature_statistics: Dict[str, float]
    recommendations: List[str]


class WeatherQualityController:
    """
    Advanced quality control system for weather data validation and correction.
    
    Detects interpolation artifacts, unrealistic extremes, and other data quality
    issues that can corrupt energy-weather correlation analysis.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Climate validation parameters for German cities
        self.climate_bounds = {
            'bottrop': {
                'temp_min_extreme': -20.0,    # Historical record low for region
                'temp_max_extreme': 42.0,     # Based on 2019 German records
                'temp_min_typical': -10.0,    # Typical winter low
                'temp_max_typical': 35.0,     # Typical summer high
                'daily_temp_range_max': 20.0  # Maximum realistic daily range
            }
        }
        
        # Quality control thresholds
        self.max_temp_gradient = 3.0       # °C per hour maximum change
        self.max_consecutive_identical = 4  # Maximum identical consecutive readings
        self.outlier_sigma_threshold = 3.5  # Standard deviations for outlier detection
        self.min_humidity = 0              # Minimum valid humidity %
        self.max_humidity = 100            # Maximum valid humidity %
        
        self.logger.info("🔍 Weather Quality Controller initialized")
    
    def analyze_data_quality(self, db_path: str, location: str, 
                           start_date: str = None, end_date: str = None) -> QualityReport:
        """
        Perform comprehensive quality analysis on weather data.
        
        Args:
            db_path: Path to SQLite weather database
            location: Location name to analyze
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            QualityReport with detailed analysis and recommendations
        """
        self.logger.info(f"🔍 Starting quality analysis for {location}")
        
        # Load data from database
        weather_data = self._load_weather_data(db_path, location, start_date, end_date)
        
        if weather_data.empty:
            self.logger.warning(f"No weather data found for {location}")
            return self._create_empty_report(location, start_date, end_date)
        
        # Analyze data quality issues
        issues = []
        issues.extend(self._detect_temperature_outliers(weather_data))
        issues.extend(self._detect_flat_lining(weather_data))
        issues.extend(self._detect_unrealistic_gradients(weather_data))
        issues.extend(self._detect_climate_violations(weather_data, location))
        issues.extend(self._detect_humidity_issues(weather_data))
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(weather_data, issues)
        data_completeness = self._calculate_data_completeness(weather_data)
        temp_stats = self._calculate_temperature_statistics(weather_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, weather_data)
        
        # Create report
        date_range = (
            weather_data['datetime'].min().strftime('%Y-%m-%d'),
            weather_data['datetime'].max().strftime('%Y-%m-%d')
        )
        
        report = QualityReport(
            location=location,
            date_range=date_range,
            total_records=len(weather_data),
            issues_found=issues,
            quality_score=quality_score,
            data_completeness=data_completeness,
            temperature_statistics=temp_stats,
            recommendations=recommendations
        )
        
        self.logger.info(f"✅ Quality analysis complete: {quality_score:.1f}% quality score")
        self._log_quality_summary(report)
        
        return report
    
    def _load_weather_data(self, db_path: str, location: str, 
                          start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load weather data from SQLite database."""
        with sqlite3.connect(db_path) as conn:
            query = '''
                SELECT datetime, temperature, humidity, condition, precipitation, weather_code
                FROM weather_data
                WHERE location LIKE ?
            '''
            params = [f'%{location}%']
            
            if start_date and end_date:
                query += ' AND datetime BETWEEN ? AND ?'
                params.extend([f"{start_date} 00:00:00", f"{end_date} 23:59:59"])
            
            query += ' ORDER BY datetime'
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
            
            return df
    
    def _detect_temperature_outliers(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Detect temperature outliers using statistical methods."""
        issues = []
        
        if 'temperature' not in data.columns or data['temperature'].isna().all():
            return issues
        
        temps = data['temperature'].dropna()
        if len(temps) < 10:
            return issues
        
        # Statistical outlier detection
        mean_temp = temps.mean()
        std_temp = temps.std()
        lower_bound = mean_temp - self.outlier_sigma_threshold * std_temp
        upper_bound = mean_temp + self.outlier_sigma_threshold * std_temp
        
        outliers = data[
            (data['temperature'] < lower_bound) | 
            (data['temperature'] > upper_bound)
        ]
        
        for _, row in outliers.iterrows():
            severity = 'high' if abs(row['temperature'] - mean_temp) > 4 * std_temp else 'medium'
            
            issues.append(QualityIssue(
                issue_type='outlier',
                severity=severity,
                timestamp=row['datetime'],
                description=f"Temperature {row['temperature']:.1f}°C is statistical outlier",
                value=row['temperature'],
                expected_range=(lower_bound, upper_bound),
                recommendation='Consider smoothing or flagging for manual review'
            ))
        
        return issues
    
    def _detect_flat_lining(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Detect periods where temperature remains artificially constant."""
        issues = []
        
        if 'temperature' not in data.columns or len(data) < 4:
            return issues
        
        # Look for consecutive identical temperatures
        data = data.copy()
        data['temp_diff'] = data['temperature'].diff()
        data['is_same'] = data['temp_diff'] == 0.0
        
        # Find groups of consecutive identical temperatures
        data['group'] = (data['is_same'] != data['is_same'].shift()).cumsum()
        consecutive_groups = data[data['is_same']].groupby('group')
        
        for group_id, group in consecutive_groups:
            consecutive_count = len(group)
            
            if consecutive_count >= self.max_consecutive_identical:
                severity = 'critical' if consecutive_count >= 8 else 'high'
                
                issues.append(QualityIssue(
                    issue_type='flat_lining',
                    severity=severity,
                    timestamp=group.iloc[0]['datetime'],
                    description=f"Temperature flat-lined at {group.iloc[0]['temperature']:.1f}°C for {consecutive_count} consecutive readings",
                    value=group.iloc[0]['temperature'],
                    expected_range=(0, 0),  # No expected range for this type
                    recommendation='Likely interpolation artifact - consider data source validation'
                ))
        
        return issues
    
    def _detect_unrealistic_gradients(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Detect unrealistic temperature changes between consecutive readings."""
        issues = []
        
        if 'temperature' not in data.columns or len(data) < 2:
            return issues
        
        data = data.copy()
        data['temp_change'] = data['temperature'].diff()
        data['time_diff'] = data['datetime'].diff().dt.total_seconds() / 3600  # Convert to hours
        
        # Calculate temperature gradient (°C per hour)
        data['temp_gradient'] = abs(data['temp_change'] / data['time_diff'])
        
        # Find excessive gradients
        excessive_gradients = data[data['temp_gradient'] > self.max_temp_gradient]
        
        for _, row in excessive_gradients.iterrows():
            if pd.isna(row['temp_gradient']):
                continue
                
            severity = 'critical' if row['temp_gradient'] > 5.0 else 'high'
            
            issues.append(QualityIssue(
                issue_type='gradient',
                severity=severity,
                timestamp=row['datetime'],
                description=f"Excessive temperature gradient: {row['temp_gradient']:.1f}°C/hour",
                value=row['temp_gradient'],
                expected_range=(0, self.max_temp_gradient),
                recommendation='Smooth temperature transitions or check data source'
            ))
        
        return issues
    
    def _detect_climate_violations(self, data: pd.DataFrame, location: str) -> List[QualityIssue]:
        """Detect temperatures that violate regional climate bounds."""
        issues = []
        
        if 'temperature' not in data.columns:
            return issues
        
        # Get climate bounds for location
        location_key = location.lower().replace(' ', '').replace(',', '')
        climate = self.climate_bounds.get(location_key, self.climate_bounds['bottrop'])
        
        # Check extreme violations
        extreme_cold = data[data['temperature'] < climate['temp_min_extreme']]
        extreme_hot = data[data['temperature'] > climate['temp_max_extreme']]
        
        for _, row in extreme_cold.iterrows():
            issues.append(QualityIssue(
                issue_type='unrealistic',
                severity='critical',
                timestamp=row['datetime'],
                description=f"Temperature {row['temperature']:.1f}°C below historical minimum for region",
                value=row['temperature'],
                expected_range=(climate['temp_min_extreme'], climate['temp_max_extreme']),
                recommendation='Verify data source - may be instrument error'
            ))
        
        for _, row in extreme_hot.iterrows():
            issues.append(QualityIssue(
                issue_type='unrealistic',
                severity='critical',
                timestamp=row['datetime'],
                description=f"Temperature {row['temperature']:.1f}°C above historical maximum for region",
                value=row['temperature'],
                expected_range=(climate['temp_min_extreme'], climate['temp_max_extreme']),
                recommendation='Verify against official weather records'
            ))
        
        return issues
    
    def _detect_humidity_issues(self, data: pd.DataFrame) -> List[QualityIssue]:
        """Detect invalid humidity values."""
        issues = []
        
        if 'humidity' not in data.columns:
            return issues
        
        invalid_humidity = data[
            (data['humidity'] < self.min_humidity) | 
            (data['humidity'] > self.max_humidity) |
            data['humidity'].isna()
        ]
        
        for _, row in invalid_humidity.iterrows():
            issues.append(QualityIssue(
                issue_type='unrealistic',
                severity='medium',
                timestamp=row['datetime'],
                description=f"Invalid humidity value: {row['humidity']}%",
                value=row['humidity'] if not pd.isna(row['humidity']) else -999,
                expected_range=(self.min_humidity, self.max_humidity),
                recommendation='Replace with interpolated value'
            ))
        
        return issues
    
    def _calculate_quality_score(self, data: pd.DataFrame, issues: List[QualityIssue]) -> float:
        """Calculate overall quality score (0-100)."""
        if len(data) == 0:
            return 0.0
        
        # Base score
        score = 100.0
        
        # Deduct points based on issue severity and frequency
        for issue in issues:
            if issue.severity == 'critical':
                score -= 10.0
            elif issue.severity == 'high':
                score -= 5.0
            elif issue.severity == 'medium':
                score -= 2.0
            else:  # low
                score -= 0.5
        
        # Additional penalty for high issue density
        issue_density = len(issues) / len(data)
        if issue_density > 0.01:  # More than 1% of records have issues
            score -= 20.0 * issue_density
        
        return max(0.0, min(100.0, score))
    
    def _calculate_data_completeness(self, data: pd.DataFrame) -> float:
        """Calculate percentage of complete data records."""
        if len(data) == 0:
            return 0.0
        
        complete_records = data.dropna().shape[0]
        return (complete_records / len(data)) * 100.0
    
    def _calculate_temperature_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate temperature statistics for the dataset."""
        if 'temperature' not in data.columns or data['temperature'].isna().all():
            return {}
        
        temps = data['temperature'].dropna()
        
        return {
            'count': len(temps),
            'mean': float(temps.mean()),
            'std': float(temps.std()),
            'min': float(temps.min()),
            'max': float(temps.max()),
            'median': float(temps.median()),
            'q25': float(temps.quantile(0.25)),
            'q75': float(temps.quantile(0.75)),
            'range': float(temps.max() - temps.min())
        }
    
    def _generate_recommendations(self, issues: List[QualityIssue], data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on identified issues."""
        recommendations = []
        
        # Count issues by type
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        # Generate specific recommendations
        if issue_counts.get('flat_lining', 0) > 0:
            recommendations.append(
                f"Found {issue_counts['flat_lining']} flat-lining periods. "
                "Consider implementing temporal smoothing or re-fetching data."
            )
        
        if issue_counts.get('outlier', 0) > 0:
            recommendations.append(
                f"Found {issue_counts['outlier']} temperature outliers. "
                "Apply statistical outlier correction or flag for manual review."
            )
        
        if issue_counts.get('gradient', 0) > 0:
            recommendations.append(
                f"Found {issue_counts['gradient']} unrealistic temperature gradients. "
                "Implement gradient smoothing with max 2°C/hour change limit."
            )
        
        if issue_counts.get('unrealistic', 0) > 0:
            recommendations.append(
                f"Found {issue_counts['unrealistic']} climatically unrealistic values. "
                "Cross-validate with official weather station data."
            )
        
        # General recommendations
        critical_issues = [i for i in issues if i.severity == 'critical']
        if critical_issues:
            recommendations.append(
                "Critical data quality issues detected. "
                "Consider re-fetching data from alternative sources."
            )
        
        if not recommendations:
            recommendations.append("Data quality is acceptable for energy analysis.")
        
        return recommendations
    
    def _create_empty_report(self, location: str, start_date: str, end_date: str) -> QualityReport:
        """Create report for cases with no data."""
        return QualityReport(
            location=location,
            date_range=(start_date or 'N/A', end_date or 'N/A'),
            total_records=0,
            issues_found=[],
            quality_score=0.0,
            data_completeness=0.0,
            temperature_statistics={},
            recommendations=['No weather data found for specified location and date range.']
        )
    
    def _log_quality_summary(self, report: QualityReport) -> None:
        """Log a summary of the quality analysis."""
        self.logger.info(f"📊 Quality Report for {report.location}:")
        self.logger.info(f"   📅 Date range: {report.date_range[0]} to {report.date_range[1]}")
        self.logger.info(f"   📊 Total records: {report.total_records:,}")
        self.logger.info(f"   🎯 Quality score: {report.quality_score:.1f}%")
        self.logger.info(f"   ✅ Data completeness: {report.data_completeness:.1f}%")
        self.logger.info(f"   ⚠️ Issues found: {len(report.issues_found)}")
        
        # Log critical issues
        critical_issues = [i for i in report.issues_found if i.severity == 'critical']
        if critical_issues:
            self.logger.warning(f"   🚨 Critical issues: {len(critical_issues)}")
            for issue in critical_issues[:3]:  # Log first 3 critical issues
                self.logger.warning(f"      - {issue.description}")
    
    def export_quality_report(self, report: QualityReport, filepath: str) -> None:
        """Export detailed quality report to file."""
        with open(filepath, 'w') as f:
            f.write(f"# Weather Data Quality Report\\n")
            f.write(f"**Location:** {report.location}\\n")
            f.write(f"**Date Range:** {report.date_range[0]} to {report.date_range[1]}\\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write(f"## Summary\\n")
            f.write(f"- **Total Records:** {report.total_records:,}\\n")
            f.write(f"- **Quality Score:** {report.quality_score:.1f}%\\n")
            f.write(f"- **Data Completeness:** {report.data_completeness:.1f}%\\n")
            f.write(f"- **Issues Found:** {len(report.issues_found)}\\n\\n")
            
            if report.temperature_statistics:
                f.write(f"## Temperature Statistics\\n")
                stats = report.temperature_statistics
                f.write(f"- **Mean:** {stats.get('mean', 0):.1f}°C\\n")
                f.write(f"- **Range:** {stats.get('min', 0):.1f}°C to {stats.get('max', 0):.1f}°C\\n")
                f.write(f"- **Standard Deviation:** {stats.get('std', 0):.1f}°C\\n\\n")
            
            if report.issues_found:
                f.write(f"## Issues Detected\\n")
                for issue in report.issues_found:
                    f.write(f"### {issue.severity.upper()}: {issue.issue_type}\\n")
                    f.write(f"- **Time:** {issue.timestamp}\\n")
                    f.write(f"- **Description:** {issue.description}\\n")
                    f.write(f"- **Recommendation:** {issue.recommendation}\\n\\n")
            
            f.write(f"## Recommendations\\n")
            for rec in report.recommendations:
                f.write(f"- {rec}\\n")
        
        self.logger.info(f"📄 Quality report exported to {filepath}")