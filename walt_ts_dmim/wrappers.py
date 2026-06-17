import os
import contextlib
import numpy as np
from SALib.analyze import delta

def compute_single_lag_dmim(x_col: np.ndarray, y_array: np.ndarray, num_resamples=10, seed=42):
    """
    Computes all DMIM scores (raw, balanced, step) for a single time lag.
    Safely muzzles SALib diagnostics and handles zero-variance edge cases.
    """
    # --- The KDE Variance Collapse Patch ---
    if np.var(x_col) < 1e-5:
        return {'raw': 0.0, 'balanced': 0.0, 'step': 0.0}
    
    problem = {
        'num_vars': 1,
        'names': ['X'],
        'bounds': [[np.min(x_col), np.max(x_col)]]
    }
    
    x_reshaped = x_col.reshape(-1, 1)
    
    # The Fixed Muzzle: Explicitly open and close the null file
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            results = delta.analyze(
                problem, 
                x_reshaped, 
                y_array, 
                num_resamples=num_resamples,
                seed=seed
            )
            
    return {
        'raw': results.get('delta_raw', [0.0])[0],
        'balanced': results.get('delta_balanced', [0.0])[0],
        'step': results.get('delta_step', [0.0])[0]
    }

