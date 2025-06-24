"""
Energy Balance Validator
========================

This module provides comprehensive validation for energy balance constraints
in the energy disaggregation system. It ensures that device profiles sum to
total energy consumption within acceptable tolerances.

Features:
- Energy conservation validation
- Device allocation constraint checking
- Time-series energy balance verification
- Statistical energy balance analysis
- Performance metrics calculation
- Validation reporting and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class EnergyBalanceValidationResult:
    """Result of energy balance validation."""
    is_valid: bool
    energy_balance_error: float          # % error in total energy conservation
    max_instantaneous_error: float       # Maximum instantaneous error (%)
    mean_absolute_error: float          # Mean absolute error (kW)
    root_mean_square_error: float       # RMSE (kW)
    device_allocation_sum: float        # Sum of device allocations (should be ~1.0)
    violated_constraints: List[str]     # List of constraint violations
    validation_metrics: Dict[str, float] # Additional validation metrics
    time_series_errors: np.ndarray      # Time series of balance errors
    device_contributions: Dict[str, float] # Device contribution percentages


@dataclass
class ValidationConstraints:
    """Constraints for energy balance validation."""
    max_energy_balance_error: float = 1.0    # Maximum acceptable total energy error (%)
    max_instantaneous_error: float = 5.0     # Maximum instantaneous error (%)
    max_device_allocation: float = 0.6        # Maximum single device allocation (60%)
    min_total_allocation: float = 0.95        # Minimum total device allocation (95%)
    max_missing_energy: float = 0.05          # Maximum unaccounted energy (5%)
    energy_conservation_tolerance: float = 0.01  # Energy conservation tolerance (1%)


class EnergyBalanceValidator:
    """
    Comprehensive energy balance validator for energy disaggregation systems.
    
    This validator ensures that the energy disaggregation maintains proper
    energy conservation and realistic device allocations.
    """
    
    def __init__(self, constraints: Optional[ValidationConstraints] = None, logger=None):
        self.constraints = constraints or ValidationConstraints()
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation history
        self.validation_history: List[EnergyBalanceValidationResult] = []
        
        self.logger.info("⚖️ Energy Balance Validator initialized")
    
    def validate_energy_balance(self, total_actual: np.ndarray, total_predicted: np.ndarray,
                               device_profiles: Dict[str, np.ndarray],
                               device_allocations: Optional[Dict[str, float]] = None) -> EnergyBalanceValidationResult:
        """
        Perform comprehensive energy balance validation.
        
        Args:
            total_actual: Actual total energy consumption
            total_predicted: Predicted total energy (sum of devices)
            device_profiles: Dictionary of device energy profiles
            device_allocations: Device allocation percentages (optional)
            
        Returns:
            EnergyBalanceValidationResult with validation outcome
        """
        self.logger.info("⚖️ Validating energy balance constraints")
        
        # Basic energy balance validation
        energy_balance_error = self._calculate_energy_balance_error(total_actual, total_predicted)
        max_instantaneous_error = self._calculate_max_instantaneous_error(total_actual, total_predicted)
        mae = self._calculate_mae(total_actual, total_predicted)
        rmse = self._calculate_rmse(total_actual, total_predicted)
        
        # Time series errors
        time_series_errors = self._calculate_time_series_errors(total_actual, total_predicted)
        
        # Device allocation validation
        device_allocation_sum, device_contributions = self._validate_device_allocations(device_profiles, device_allocations)
        
        # Constraint violation checking
        violated_constraints = self._check_constraint_violations(
            energy_balance_error, max_instantaneous_error, device_allocation_sum, device_contributions
        )
        
        # Additional validation metrics
        validation_metrics = self._calculate_additional_metrics(
            total_actual, total_predicted, device_profiles
        )
        
        # Determine if validation passed
        is_valid = len(violated_constraints) == 0
        
        result = EnergyBalanceValidationResult(
            is_valid=is_valid,
            energy_balance_error=energy_balance_error,
            max_instantaneous_error=max_instantaneous_error,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            device_allocation_sum=device_allocation_sum,
            violated_constraints=violated_constraints,
            validation_metrics=validation_metrics,
            time_series_errors=time_series_errors,
            device_contributions=device_contributions
        )
        
        # Store in history
        self.validation_history.append(result)
        
        # Log validation results
        self._log_validation_results(result)
        
        return result
    
    def validate_disaggregation_result(self, disaggregation_result) -> EnergyBalanceValidationResult:
        """
        Validate an EnergyDisaggregationResult object.
        
        Args:
            disaggregation_result: EnergyDisaggregationResult from disaggregator
            
        Returns:
            EnergyBalanceValidationResult
        """
        return self.validate_energy_balance(
            total_actual=disaggregation_result.total_actual,
            total_predicted=disaggregation_result.total_predicted,
            device_profiles=disaggregation_result.device_profiles,
            device_allocations=disaggregation_result.allocation_summary
        )
    
    def _calculate_energy_balance_error(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate total energy balance error as percentage."""
        total_actual = np.sum(actual)
        total_predicted = np.sum(predicted)
        
        if total_actual == 0:
            return 0.0 if total_predicted == 0 else 100.0
        
        return abs(total_predicted - total_actual) / total_actual * 100.0
    
    def _calculate_max_instantaneous_error(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate maximum instantaneous error as percentage."""
        # Avoid division by zero
        mask = actual > 0
        if not np.any(mask):
            return 0.0
        
        instantaneous_errors = np.abs(predicted[mask] - actual[mask]) / actual[mask] * 100.0
        return float(np.max(instantaneous_errors))
    
    def _calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate mean absolute error."""
        return float(np.mean(np.abs(predicted - actual)))
    
    def _calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate root mean square error."""
        return float(np.sqrt(np.mean((predicted - actual) ** 2)))
    
    def _calculate_time_series_errors(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Calculate time series of instantaneous errors."""
        # Avoid division by zero
        mask = actual > 0
        errors = np.zeros_like(actual)
        errors[mask] = (predicted[mask] - actual[mask]) / actual[mask] * 100.0
        return errors
    
    def _validate_device_allocations(self, device_profiles: Dict[str, np.ndarray],
                                   device_allocations: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        """Validate device allocation constraints."""
        if not device_profiles:
            return 0.0, {}
        
        # Calculate device contributions from profiles
        total_energy = sum(np.sum(profile) for profile in device_profiles.values())
        device_contributions = {}
        
        for device_name, profile in device_profiles.items():
            if total_energy > 0:
                device_contributions[device_name] = np.sum(profile) / total_energy
            else:
                device_contributions[device_name] = 0.0
        
        # Sum of device allocations
        device_allocation_sum = sum(device_contributions.values())
        
        return device_allocation_sum, device_contributions
    
    def _check_constraint_violations(self, energy_balance_error: float,
                                   max_instantaneous_error: float,
                                   device_allocation_sum: float,
                                   device_contributions: Dict[str, float]) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        # Energy balance constraint
        if energy_balance_error > self.constraints.max_energy_balance_error:
            violations.append(f"Energy balance error ({energy_balance_error:.3f}%) exceeds limit ({self.constraints.max_energy_balance_error:.1f}%)")
        
        # Instantaneous error constraint
        if max_instantaneous_error > self.constraints.max_instantaneous_error:
            violations.append(f"Max instantaneous error ({max_instantaneous_error:.3f}%) exceeds limit ({self.constraints.max_instantaneous_error:.1f}%)")
        
        # Device allocation constraints
        for device_name, allocation in device_contributions.items():
            if allocation > self.constraints.max_device_allocation:
                violations.append(f"Device '{device_name}' allocation ({allocation:.3f}) exceeds limit ({self.constraints.max_device_allocation:.1f})")
        
        # Total allocation constraint
        if device_allocation_sum < self.constraints.min_total_allocation:
            violations.append(f"Total device allocation ({device_allocation_sum:.3f}) below minimum ({self.constraints.min_total_allocation:.1f})")
        
        # Missing energy constraint
        missing_energy = abs(1.0 - device_allocation_sum)
        if missing_energy > self.constraints.max_missing_energy:
            violations.append(f"Missing energy ({missing_energy:.3f}) exceeds limit ({self.constraints.max_missing_energy:.1f})")
        
        return violations
    
    def _calculate_additional_metrics(self, actual: np.ndarray, predicted: np.ndarray,
                                    device_profiles: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate additional validation metrics."""
        metrics = {}
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Correlation coefficient
        if len(actual) > 1:
            metrics['correlation'] = np.corrcoef(actual, predicted)[0, 1]
        else:
            metrics['correlation'] = 1.0
        
        # Peak prediction accuracy
        actual_peak = np.max(actual)
        predicted_peak = np.max(predicted)
        if actual_peak > 0:
            metrics['peak_error_percent'] = abs(predicted_peak - actual_peak) / actual_peak * 100.0
        else:
            metrics['peak_error_percent'] = 0.0
        
        # Energy conservation score (0-100, higher is better)
        energy_error = abs(np.sum(predicted) - np.sum(actual)) / np.sum(actual) * 100.0 if np.sum(actual) > 0 else 0.0
        metrics['conservation_score'] = max(0.0, 100.0 - energy_error)
        
        # Device diversity (number of significant devices)
        total_energy = sum(np.sum(profile) for profile in device_profiles.values())
        significant_devices = sum(1 for profile in device_profiles.values() 
                                if total_energy > 0 and np.sum(profile) / total_energy > 0.01)
        metrics['device_diversity'] = float(significant_devices)
        
        # Statistical metrics
        if HAS_SCIPY and len(actual) > 2:
            # Kolmogorov-Smirnov test for distribution similarity
            ks_stat, ks_p_value = stats.ks_2samp(actual, predicted)
            metrics['ks_statistic'] = ks_stat
            metrics['ks_p_value'] = ks_p_value
        
        return metrics
    
    def _log_validation_results(self, result: EnergyBalanceValidationResult) -> None:
        """Log validation results."""
        status = "✅ PASSED" if result.is_valid else "❌ FAILED"
        self.logger.info(f"⚖️ Energy Balance Validation {status}")
        self.logger.info(f"   📊 Energy balance error: {result.energy_balance_error:.3f}%")
        self.logger.info(f"   ⚡ Max instantaneous error: {result.max_instantaneous_error:.3f}%")
        self.logger.info(f"   📈 Device allocation sum: {result.device_allocation_sum:.3f}")
        self.logger.info(f"   🎯 Conservation score: {result.validation_metrics.get('conservation_score', 0):.1f}/100")
        
        if result.violated_constraints:
            self.logger.warning("⚠️ Constraint violations:")
            for violation in result.violated_constraints:
                self.logger.warning(f"   - {violation}")
    
    def generate_validation_report(self, result: EnergyBalanceValidationResult,
                                 save_path: Optional[str] = None) -> str:
        """Generate detailed validation report."""
        report_lines = [
            "=" * 60,
            "ENERGY BALANCE VALIDATION REPORT",
            "=" * 60,
            f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Status: {'PASSED' if result.is_valid else 'FAILED'}",
            "",
            "ENERGY CONSERVATION METRICS:",
            f"  Total Energy Balance Error: {result.energy_balance_error:.3f}%",
            f"  Maximum Instantaneous Error: {result.max_instantaneous_error:.3f}%",
            f"  Mean Absolute Error: {result.mean_absolute_error:.3f} kW",
            f"  Root Mean Square Error: {result.root_mean_square_error:.3f} kW",
            f"  R-squared: {result.validation_metrics.get('r_squared', 0):.4f}",
            f"  Correlation: {result.validation_metrics.get('correlation', 0):.4f}",
            "",
            "DEVICE ALLOCATION METRICS:",
            f"  Total Device Allocation: {result.device_allocation_sum:.3f}",
            f"  Device Diversity: {result.validation_metrics.get('device_diversity', 0):.0f} devices",
            f"  Peak Prediction Error: {result.validation_metrics.get('peak_error_percent', 0):.3f}%",
            "",
            "DEVICE CONTRIBUTIONS:"
        ]
        
        # Add device contributions
        for device, contribution in sorted(result.device_contributions.items(), 
                                         key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {device}: {contribution:.3f} ({contribution*100:.1f}%)")
        
        # Add constraints section
        report_lines.extend([
            "",
            "CONSTRAINT VALIDATION:",
            f"  Max Energy Balance Error: {self.constraints.max_energy_balance_error:.1f}%",
            f"  Max Instantaneous Error: {self.constraints.max_instantaneous_error:.1f}%",
            f"  Max Device Allocation: {self.constraints.max_device_allocation:.1f}",
            f"  Min Total Allocation: {self.constraints.min_total_allocation:.1f}",
        ])
        
        # Add violations if any
        if result.violated_constraints:
            report_lines.extend([
                "",
                "CONSTRAINT VIOLATIONS:",
            ])
            for violation in result.violated_constraints:
                report_lines.append(f"  ❌ {violation}")
        else:
            report_lines.extend([
                "",
                "✅ All constraints satisfied"
            ])
        
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"📄 Validation report saved to {save_path}")
        
        return report
    
    def plot_validation_results(self, result: EnergyBalanceValidationResult,
                              total_actual: np.ndarray, total_predicted: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """Create visualization of validation results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Energy Balance Validation Results', fontsize=16, fontweight='bold')
        
        # Time series of actual vs predicted
        time_indices = np.arange(len(total_actual))
        axes[0, 0].plot(time_indices, total_actual, label='Actual', alpha=0.8)
        axes[0, 0].plot(time_indices, total_predicted, label='Predicted', alpha=0.8)
        axes[0, 0].set_xlabel('Time Index')
        axes[0, 0].set_ylabel('Energy (kW)')
        axes[0, 0].set_title('Actual vs Predicted Energy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot actual vs predicted
        axes[0, 1].scatter(total_actual, total_predicted, alpha=0.6)
        min_val = min(np.min(total_actual), np.min(total_predicted))
        max_val = max(np.max(total_actual), np.max(total_predicted))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual Energy (kW)')
        axes[0, 1].set_ylabel('Predicted Energy (kW)')
        axes[0, 1].set_title(f'Actual vs Predicted (R²={result.validation_metrics.get("r_squared", 0):.3f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series errors
        axes[0, 2].plot(time_indices, result.time_series_errors, alpha=0.8)
        axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 2].axhline(y=self.constraints.max_instantaneous_error, color='red', linestyle='--', alpha=0.8, label='Error Limit')
        axes[0, 2].axhline(y=-self.constraints.max_instantaneous_error, color='red', linestyle='--', alpha=0.8)
        axes[0, 2].set_xlabel('Time Index')
        axes[0, 2].set_ylabel('Error (%)')
        axes[0, 2].set_title('Instantaneous Energy Balance Errors')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Device contribution pie chart
        device_names = list(result.device_contributions.keys())[:10]  # Top 10 devices
        device_values = [result.device_contributions[name] for name in device_names]
        
        if device_values:
            axes[1, 0].pie(device_values, labels=device_names, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Device Energy Contributions')
        
        # Validation metrics bar chart
        metrics_to_plot = ['conservation_score', 'r_squared', 'correlation']
        metric_names = ['Conservation Score', 'R²', 'Correlation']
        metric_values = [result.validation_metrics.get(metric, 0) * 100 if metric != 'conservation_score' 
                        else result.validation_metrics.get(metric, 0) for metric in metrics_to_plot]
        
        bars = axes[1, 1].bar(metric_names, metric_values, color=['green', 'blue', 'orange'])
        axes[1, 1].set_ylabel('Score (%)')
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # Error histogram
        error_values = result.time_series_errors[np.abs(result.time_series_errors) < 50]  # Filter extreme outliers
        axes[1, 2].hist(error_values, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(x=0, color='black', linestyle='-', alpha=0.8, label='Perfect Balance')
        axes[1, 2].axvline(x=self.constraints.max_instantaneous_error, color='red', linestyle='--', alpha=0.8, label='Error Limit')
        axes[1, 2].axvline(x=-self.constraints.max_instantaneous_error, color='red', linestyle='--', alpha=0.8)
        axes[1, 2].set_xlabel('Error (%)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Error Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Validation plots saved to {save_path}")
        
        plt.show()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_history:
            return {'message': 'No validation history available'}
        
        # Calculate summary statistics
        energy_balance_errors = [result.energy_balance_error for result in self.validation_history]
        max_instantaneous_errors = [result.max_instantaneous_error for result in self.validation_history]
        conservation_scores = [result.validation_metrics.get('conservation_score', 0) for result in self.validation_history]
        
        summary = {
            'total_validations': len(self.validation_history),
            'passed_validations': sum(1 for result in self.validation_history if result.is_valid),
            'pass_rate': sum(1 for result in self.validation_history if result.is_valid) / len(self.validation_history) * 100,
            'average_energy_balance_error': np.mean(energy_balance_errors),
            'max_energy_balance_error': np.max(energy_balance_errors),
            'average_conservation_score': np.mean(conservation_scores),
            'latest_validation': {
                'is_valid': self.validation_history[-1].is_valid,
                'energy_balance_error': self.validation_history[-1].energy_balance_error,
                'conservation_score': self.validation_history[-1].validation_metrics.get('conservation_score', 0)
            }
        }
        
        return summary
    
    def export_validation_history(self, filepath: str) -> None:
        """Export validation history to file."""
        import json
        
        export_data = {
            'validation_summary': self.get_validation_summary(),
            'constraints': {
                'max_energy_balance_error': self.constraints.max_energy_balance_error,
                'max_instantaneous_error': self.constraints.max_instantaneous_error,
                'max_device_allocation': self.constraints.max_device_allocation,
                'min_total_allocation': self.constraints.min_total_allocation
            },
            'validation_history': []
        }
        
        for result in self.validation_history:
            export_data['validation_history'].append({
                'is_valid': result.is_valid,
                'energy_balance_error': result.energy_balance_error,
                'max_instantaneous_error': result.max_instantaneous_error,
                'device_allocation_sum': result.device_allocation_sum,
                'violated_constraints': result.violated_constraints,
                'validation_metrics': result.validation_metrics,
                'device_contributions': result.device_contributions
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"📁 Validation history exported to {filepath}")