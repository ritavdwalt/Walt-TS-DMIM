import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')  # Tell Matplotlib it is running on a headless server
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import List

@dataclass
class DiagnosticProfile:
    """Immutable data structure containing the temporal influence metrics for a single parameter."""
    parameter_name: str
    metric_used: str
    peak_lag_steps: int
    peak_delta_unsmoothed: float
    prominence_ratio: float
    duration_t98_steps: int
    cumulative_auc: float
    basin_left: int
    basin_right: int
    confidence: str
    flags: List[str] = field(default_factory=list)

def extract_signal_holistic(param_name: str, curves_dict: dict, metric: str = 'balanced', smooth_window: int = 6, system_warnings: List[str] = None) -> DiagnosticProfile:
    """
    Extracts the true causal lag using time-aware topological prominence.
    Isolates the 'Causal Basin' using localized valley detection for thermodynamic profiling.
    """
    flags = system_warnings.copy() if system_warnings else []
    active_curve = np.nan_to_num(curves_dict[metric])

    # Variance safeguard
    if np.ptp(active_curve) == 0:
        flags.append("Zero variance in curve.")
        return DiagnosticProfile(param_name, metric, 0, 0.0, 0.0, 0, 0.0, 0, 0, 'Zero Variance', flags)

    # 1. Smooth to remove sensor micro-jitter
    smoothed_curve = pd.Series(active_curve).rolling(window=smooth_window, min_periods=1, center=True).mean().to_numpy()

    # 2. THE PADDING TRICK
    padded_curve = np.insert(smoothed_curve, 0, np.min(smoothed_curve))
    peaks, properties = find_peaks(padded_curve, prominence=0)

    if len(peaks) > 0:
        prominences = properties['prominences']
        max_prominence = np.max(prominences)

        # 3. THE RELATIVE CAUSALITY FILTER
        valid_peaks_mask = prominences >= (0.5 * max_prominence)
        first_valid_idx = np.where(valid_peaks_mask)[0][0]
        true_peak_idx = peaks[first_valid_idx] - 1
        
        # 4. THE CAUSAL BASIN EXTRACTION
        dynamic_prom = np.ptp(smoothed_curve) * 0.01
        valleys, _ = find_peaks(-smoothed_curve, prominence=dynamic_prom)
        
        left_valleys = valleys[valleys < true_peak_idx]
        left_base_idx = int(left_valleys[-1]) if len(left_valleys) > 0 else 0
        
        right_valleys = valleys[valleys > true_peak_idx]
        right_base_idx = int(right_valleys[0]) if len(right_valleys) > 0 else len(smoothed_curve) - 1

        # Extract slices
        basin_slice_raw = active_curve[left_base_idx : right_base_idx + 1]
        basin_slice_smoothed = smoothed_curve[left_base_idx : right_base_idx + 1]
        peak_val_raw = active_curve[true_peak_idx]
        
        # Metric 1: Peak Prominence Ratio
        basin_median = np.median(basin_slice_raw)
        prominence_ratio = peak_val_raw / basin_median if basin_median > 1e-5 else peak_val_raw / 1e-5
        
        # Metric 2: Sharpness / Duration (t_delta > 0.98 * peak)
        threshold = 0.98 * peak_val_raw
        duration_t98_steps = int(np.sum(basin_slice_raw >= threshold))
        
        # Metric 3: Cumulative Impact (AUC via Trapezoidal Integration)
        cumulative_auc = float(np.trapezoid(basin_slice_smoothed))

        # Confidence Check
        if prominence_ratio < 1.15:
            confidence = 'Low (Indistinguishable from noise floor)'
            flags.append("LOW CONFIDENCE: Prominence ratio < 1.15. Peak is likely statistical noise.")
        else:
            confidence = 'High (First Causal Peak)'
    else:
        # Absolute fallback
        true_peak_idx = int(np.argmax(smoothed_curve))
        peak_val_raw = active_curve[true_peak_idx]
        prominence_ratio = 1.0
        duration_t98_steps = 1
        cumulative_auc = float(peak_val_raw)
        left_base_idx = 0
        right_base_idx = len(smoothed_curve) - 1
        confidence = 'Low (No peaks detected)'
        flags.append("No topographical peaks detected.")

    return DiagnosticProfile(
        parameter_name=param_name, metric_used=metric, peak_lag_steps=int(true_peak_idx),
        peak_delta_unsmoothed=float(peak_val_raw), prominence_ratio=float(prominence_ratio),
        duration_t98_steps=duration_t98_steps, cumulative_auc=cumulative_auc,
        basin_left=int(left_base_idx), basin_right=int(right_base_idx),
        confidence=confidence, flags=flags
    )

def plot_diagnostic_grid(dmim_curves_dict, acf_curves_dict, profiles_dict, smooth_window=6, show_plot=False,
                         save_path=None, title_prefix=""):
    """Generates the visual grid isolating the Causal Basin and topological metrics."""
    params = list(dmim_curves_dict.keys())
    num_params = len(params)
    cols = 2
    rows = math.ceil(num_params / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = axes.flatten() if num_params > 1 else [axes]
    
    for i, param in enumerate(params):
        ax1 = axes[i]
        profile = profiles_dict[param]
        
        raw_curve = dmim_curves_dict[param][profile.metric_used]
        smoothed_curve = pd.Series(raw_curve).rolling(window=smooth_window, min_periods=1, center=True).mean().to_numpy()
        acf_curve = acf_curves_dict.get(param, np.zeros_like(raw_curve))
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Time Lag (Steps)', fontweight='bold')
        ax1.set_ylabel(f'Sensitivity ({profile.metric_used})', color=color1, fontweight='bold')
        ax1.plot(raw_curve, color=color1, alpha=0.3, label='Raw DMIM')
        ax1.plot(smoothed_curve, color=color1, linewidth=2, label='Smoothed DMIM')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = 'tab:gray'
        ax2.plot(acf_curve, color=color2, linewidth=1.5, linestyle='--', label='ACF Baseline')
        ax2.set_ylabel('ACF', color=color2, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax1.axvspan(profile.basin_left, profile.basin_right, color='tab:blue', alpha=0.1, label='Causal Basin Bounds')
        ax1.axvline(x=profile.peak_lag_steps, color='red', linestyle='-', linewidth=2, label=f'Peak Extracted: {profile.peak_lag_steps}')
        
        # Determine a visual warning symbol based on the engine's confidence rating
        # Using standard, universally supported text/unicode characters
        if "Low" in profile.confidence:
            status_symbol = "⚠ WARNING (Noise)"  # Standard warning sign (U+26A0) or just "[!]"
        else:
            status_symbol = "✓ Valid Signal"     # Standard checkmark (U+2713)

        textstr = '\n'.join((
            f"Status: {status_symbol}",
            f"Peak Lag: {profile.peak_lag_steps} steps",
            f"Peak Delta: {profile.peak_delta_unsmoothed:.4f}",
            f"Prominence Ratio: {profile.prominence_ratio:.4f}",
            f"Cumulative AUC: {profile.cumulative_auc:.4f}"
        ))
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
        ax1.set_title(f"{param.replace('_', ' ').upper()}", color='black', fontsize=14, fontweight='bold')
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=9)
        
        title = f"{title_prefix} {param.replace('_', ' ').upper()}" if title_prefix else f"{param.replace('_', ' ').upper()}"
        ax1.set_title(title, color='black', fontsize=14, fontweight='bold')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot==True: plt.show()
    else:
        plt.show()

    plt.close(fig)