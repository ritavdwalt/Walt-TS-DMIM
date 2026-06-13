import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class DiagnosticProfile:
    """Immutable data structure containing the temporal influence metrics for a single parameter."""
    parameter_name: str
    peak_lag_steps: int
    peak_delta_score: float
    prominence_ratio: float
    cumulative_auc: float
    flags: List[str] = field(default_factory=list)

def compute_parameter_profile(param_name: str, delta_curve: np.ndarray, lag_steps: np.ndarray, system_warnings: List[str] = None) -> DiagnosticProfile:
    """
    Extracts the topological metrics from the raw DMIM sensitivity curve.
    """
    flags = system_warnings.copy() if system_warnings else []
    
    # 1. Acute Metrics
    peak_idx = np.argmax(delta_curve)
    peak_lag = int(lag_steps[peak_idx])
    peak_score = float(delta_curve[peak_idx])
    
    # 2. Systemic Validity (Prominence)
    median_score = np.median(delta_curve)
    prominence = peak_score / median_score if median_score > 0 else 1.0
    
    if prominence < 1.05:
        flags.append("LOW CONFIDENCE: Prominence ratio < 1.05. The peak is likely statistical noise against a flat background.")
        
    # 3. Aggregate Impact (AUC via Trapezoidal Rule)
    auc = float(np.trapz(delta_curve, lag_steps))
    
    return DiagnosticProfile(
        parameter_name=param_name,
        peak_lag_steps=peak_lag,
        peak_delta_score=peak_score,
        prominence_ratio=prominence,
        cumulative_auc=auc,
        flags=flags
    )