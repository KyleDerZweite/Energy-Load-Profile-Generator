#!/usr/bin/env python3
"""
Energy Forecast Evaluation Script
=================================

Comprehensive evaluation of the 2025 energy forecast to assess:
- Forecast quality and realism
- Scenario differentiation
- Pattern consistency  
- Improvement opportunities
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_forecast_data(forecast_file: str) -> Dict[str, Any]:
    """Load forecast data from JSON file."""
    with open(forecast_file, 'r') as f:
        return json.load(f)

def load_historical_data(file_path: str) -> pd.DataFrame:
    """Load historical energy data for comparison."""
    try:
        df = pd.read_excel(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')
        return df
    except Exception as e:
        logger.warning(f"Could not load historical data: {e}")
        return pd.DataFrame()

def analyze_forecast_statistics(forecast_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Calculate statistical metrics for each forecast scenario."""
    stats = {}
    
    for scenario_name, scenario_data in forecast_data['scenarios'].items():
        values = np.array(scenario_data['total_energy_forecast'])
        
        stats[scenario_name] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,  # Coefficient of variation
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25)
        }
    
    return stats

def evaluate_scenario_differentiation(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Evaluate how well scenarios differentiate from each other."""
    scenarios = list(stats.keys())
    
    if len(scenarios) < 2:
        return {'differentiation_score': 0.0}
    
    # Calculate mean differences between scenarios
    mean_diffs = []
    for i in range(len(scenarios)):
        for j in range(i+1, len(scenarios)):
            diff = abs(stats[scenarios[i]]['mean'] - stats[scenarios[j]]['mean'])
            mean_diffs.append(diff)
    
    avg_mean_diff = np.mean(mean_diffs)
    
    # Calculate range differences
    range_diffs = []
    for i in range(len(scenarios)):
        for j in range(i+1, len(scenarios)):
            diff = abs(stats[scenarios[i]]['range'] - stats[scenarios[j]]['range'])
            range_diffs.append(diff)
    
    avg_range_diff = np.mean(range_diffs)
    
    return {
        'mean_difference': avg_mean_diff,
        'range_difference': avg_range_diff,
        'differentiation_score': avg_mean_diff + avg_range_diff
    }

def check_forecast_realism(stats: Dict[str, Dict[str, float]], historical_stats: Dict[str, float] = None) -> Dict[str, Any]:
    """Check if forecast values are realistic."""
    realism_check = {}
    
    for scenario, scenario_stats in stats.items():
        checks = {
            'positive_values': scenario_stats['min'] > 0,
            'reasonable_cv': 0.1 <= scenario_stats['cv'] <= 2.0,  # Reasonable coefficient of variation
            'no_extreme_outliers': scenario_stats['max'] / scenario_stats['mean'] < 10,  # No extreme spikes
            'sufficient_variability': scenario_stats['std'] > 1.0  # Some variability expected
        }
        
        # If historical data available, compare ranges
        if historical_stats:
            checks['within_historical_range'] = (
                scenario_stats['min'] >= historical_stats.get('min', 0) * 0.5 and
                scenario_stats['max'] <= historical_stats.get('max', 1000) * 2.0
            )
            checks['similar_variability'] = (
                0.5 <= scenario_stats['cv'] / historical_stats.get('cv', 1.0) <= 2.0
            )
        
        realism_check[scenario] = {
            'checks': checks,
            'realism_score': sum(checks.values()) / len(checks)
        }
    
    return realism_check

def analyze_temporal_patterns(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze temporal patterns in the forecast."""
    pattern_analysis = {}
    
    for scenario_name, scenario_data in forecast_data['scenarios'].items():
        values = np.array(scenario_data['total_energy_forecast'])
        
        # Look for trends
        x = np.arange(len(values))
        trend_coeff = np.polyfit(x, values, 1)[0]
        
        # Look for seasonality (if enough data points)
        seasonal_pattern = 'unknown'
        if len(values) >= 52:  # At least weekly data
            # Simple seasonal detection - look for periodic patterns
            try:
                fft = np.fft.fft(values)
                freqs = np.fft.fftfreq(len(values))
                dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                dominant_period = 1 / abs(freqs[dominant_freq_idx]) if freqs[dominant_freq_idx] != 0 else 0
                
                if 6 <= dominant_period <= 8:
                    seasonal_pattern = 'weekly'
                elif 350 <= dominant_period <= 370:
                    seasonal_pattern = 'annual'
                else:
                    seasonal_pattern = f'period_{dominant_period:.1f}'
            except:
                seasonal_pattern = 'analysis_failed'
        
        # Calculate autocorrelation at lag 1
        if len(values) > 1:
            try:
                autocorr_lag1 = np.corrcoef(values[:-1], values[1:])[0, 1]
                if np.isnan(autocorr_lag1):
                    autocorr_lag1 = 0
            except:
                autocorr_lag1 = 0
        else:
            autocorr_lag1 = 0
        
        pattern_analysis[scenario_name] = {
            'trend_coefficient': trend_coeff,
            'trend_direction': 'increasing' if trend_coeff > 0.1 else 'decreasing' if trend_coeff < -0.1 else 'stable',
            'seasonal_pattern': seasonal_pattern,
            'autocorrelation_lag1': autocorr_lag1,
            'smoothness': 1 - np.std(np.diff(values)) / np.std(values) if np.std(values) > 0 else 0
        }
    
    return pattern_analysis

def identify_improvement_opportunities(stats: Dict[str, Dict[str, float]], 
                                     realism: Dict[str, Any], 
                                     patterns: Dict[str, Any],
                                     differentiation: Dict[str, float]) -> List[str]:
    """Identify areas for system improvement."""
    improvements = []
    
    # Check scenario differentiation
    if differentiation['differentiation_score'] < 5.0:
        improvements.append("LOW_SCENARIO_DIFFERENTIATION: Scenarios are too similar - improve weather scenario generation")
    
    # Check realism scores
    low_realism_scenarios = [s for s, r in realism.items() if r['realism_score'] < 0.7]
    if low_realism_scenarios:
        improvements.append(f"LOW_REALISM: Scenarios {low_realism_scenarios} have realism issues - review energy models")
    
    # Check for flat patterns
    flat_scenarios = [s for s, p in patterns.items() if abs(p['trend_coefficient']) < 0.01 and p['autocorrelation_lag1'] > 0.9]
    if flat_scenarios:
        improvements.append(f"FLAT_PATTERNS: Scenarios {flat_scenarios} show little variation - add more dynamic behavior")
    
    # Check coefficient of variation
    low_variation_scenarios = [s for s, st in stats.items() if st['cv'] < 0.1]
    if low_variation_scenarios:
        improvements.append(f"LOW_VARIATION: Scenarios {low_variation_scenarios} have insufficient variability")
    
    # Check for unrealistic values
    unrealistic_scenarios = []
    for scenario, scenario_stats in stats.items():
        if scenario_stats['min'] <= 0 or scenario_stats['max'] / scenario_stats['mean'] > 5:
            unrealistic_scenarios.append(scenario)
    
    if unrealistic_scenarios:
        improvements.append(f"UNREALISTIC_VALUES: Scenarios {unrealistic_scenarios} have unrealistic energy values")
    
    return improvements

def create_forecast_plots(forecast_data: Dict[str, Any], stats: Dict[str, Dict[str, float]], output_dir: str):
    """Create visualization plots for forecast analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot 1: Scenario comparison
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Time series
    plt.subplot(2, 2, 1)
    for scenario_name, scenario_data in forecast_data['scenarios'].items():
        values = scenario_data['total_energy_forecast']
        plt.plot(values, label=scenario_name, linewidth=2)
    
    plt.title('Forecast Scenarios Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Distribution comparison
    plt.subplot(2, 2, 2)
    all_values = []
    scenario_labels = []
    
    for scenario_name, scenario_data in forecast_data['scenarios'].items():
        values = scenario_data['total_energy_forecast']
        all_values.extend(values)
        scenario_labels.extend([scenario_name] * len(values))
    
    df_plot = pd.DataFrame({'Energy': all_values, 'Scenario': scenario_labels})
    sns.boxplot(data=df_plot, x='Scenario', y='Energy')
    plt.title('Energy Distribution by Scenario')
    plt.xticks(rotation=45)
    
    # Subplot 3: Statistics comparison
    plt.subplot(2, 2, 3)
    metrics = ['mean', 'std', 'min', 'max']
    scenarios = list(stats.keys())
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, scenario in enumerate(scenarios):
        values = [stats[scenario][metric] for metric in metrics]
        plt.bar(x + i * width, values, width, label=scenario)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Statistical Comparison')
    plt.xticks(x + width, metrics)
    plt.legend()
    
    # Subplot 4: Variability analysis
    plt.subplot(2, 2, 4)
    cvs = [stats[scenario]['cv'] for scenario in scenarios]
    ranges = [stats[scenario]['range'] for scenario in scenarios]
    
    plt.scatter(cvs, ranges, s=100)
    for i, scenario in enumerate(scenarios):
        plt.annotate(scenario, (cvs[i], ranges[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Coefficient of Variation')
    plt.ylabel('Range (Max - Min)')
    plt.title('Variability Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'forecast_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'forecast_analysis.pdf', bbox_inches='tight')
    plt.show()

def main():
    """Main evaluation function."""
    print("🔮 Energy Forecast Evaluation")
    print("=" * 50)
    
    # Load forecast data (use latest forecast file)
    forecast_file = "output/energy_forecast_bottrop_germany_2025_20250625_234519.json"
    if not Path(forecast_file).exists():
        print(f"❌ Forecast file not found: {forecast_file}")
        return
    
    forecast_data = load_forecast_data(forecast_file)
    print(f"📊 Loaded forecast with {len(forecast_data['scenarios'])} scenarios")
    
    # Load historical data for comparison
    historical_data = load_historical_data("load_profiles.xlsx")
    historical_stats = {}
    if not historical_data.empty:
        historical_values = historical_data['Value'].values
        historical_stats = {
            'mean': np.mean(historical_values),
            'std': np.std(historical_values),
            'min': np.min(historical_values),
            'max': np.max(historical_values),
            'cv': np.std(historical_values) / np.mean(historical_values)
        }
        print(f"📈 Historical data: {len(historical_values)} records")
    
    # Analyze forecast statistics
    stats = analyze_forecast_statistics(forecast_data)
    print(f"\n📊 FORECAST STATISTICS:")
    for scenario, scenario_stats in stats.items():
        print(f"\n{scenario.upper()} SCENARIO:")
        print(f"  Mean: {scenario_stats['mean']:.2f} kW")
        print(f"  Std Dev: {scenario_stats['std']:.2f} kW")
        print(f"  Range: {scenario_stats['min']:.2f} - {scenario_stats['max']:.2f} kW")
        print(f"  Coefficient of Variation: {scenario_stats['cv']:.3f}")
    
    # Evaluate scenario differentiation
    differentiation = evaluate_scenario_differentiation(stats)
    print(f"\n🎯 SCENARIO DIFFERENTIATION:")
    print(f"  Mean difference: {differentiation['mean_difference']:.2f} kW")
    print(f"  Range difference: {differentiation['range_difference']:.2f} kW")
    print(f"  Differentiation score: {differentiation['differentiation_score']:.2f}")
    
    # Check forecast realism
    realism = check_forecast_realism(stats, historical_stats)
    print(f"\n✅ REALISM ASSESSMENT:")
    for scenario, scenario_realism in realism.items():
        print(f"\n{scenario.upper()} SCENARIO:")
        print(f"  Realism score: {scenario_realism['realism_score']:.2f}")
        for check, passed in scenario_realism['checks'].items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
    
    # Analyze temporal patterns
    patterns = analyze_temporal_patterns(forecast_data)
    print(f"\n📈 TEMPORAL PATTERNS:")
    for scenario, pattern in patterns.items():
        print(f"\n{scenario.upper()} SCENARIO:")
        print(f"  Trend: {pattern['trend_direction']} ({pattern['trend_coefficient']:.4f})")
        print(f"  Seasonal pattern: {pattern['seasonal_pattern']}")
        if 'autocorrelation_lag1' in pattern:
            print(f"  Autocorrelation (lag 1): {pattern['autocorrelation_lag1']:.3f}")
        else:
            print(f"  Autocorrelation (lag 1): Not available")
        if 'smoothness' in pattern:
            print(f"  Smoothness: {pattern['smoothness']:.3f}")
        else:
            print(f"  Smoothness: Not available")
    
    # Identify improvement opportunities
    improvements = identify_improvement_opportunities(stats, realism, patterns, differentiation)
    print(f"\n🔧 IMPROVEMENT OPPORTUNITIES:")
    if improvements:
        for i, improvement in enumerate(improvements, 1):
            print(f"  {i}. {improvement}")
    else:
        print("  ✅ No major issues identified!")
    
    # Create visualization plots
    create_forecast_plots(forecast_data, stats, "output")
    print(f"\n📊 Visualization plots saved to output/forecast_analysis.png")
    
    # Overall assessment
    avg_realism = np.mean([r['realism_score'] for r in realism.values()])
    overall_score = (avg_realism + min(1.0, differentiation['differentiation_score'] / 10)) / 2
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"  Average realism score: {avg_realism:.2f}")
    print(f"  Overall forecast quality: {overall_score:.2f}")
    
    if overall_score >= 0.8:
        print("  🟢 EXCELLENT: Forecast quality is very good")
    elif overall_score >= 0.6:
        print("  🟡 GOOD: Forecast quality is acceptable with room for improvement")
    else:
        print("  🔴 NEEDS IMPROVEMENT: Forecast quality requires attention")
    
    print(f"\n✅ Forecast evaluation completed!")

if __name__ == "__main__":
    main()