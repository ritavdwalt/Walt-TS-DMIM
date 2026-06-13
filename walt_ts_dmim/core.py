import logging
import pandas as pd
import numpy as np

class TemporalAnalyzer:
    """
    Core engine for the Time-Series DMIM framework.
    Handles data validation, synchronization, and memory-efficient lag generation.
    """
    def __init__(self, input_df: pd.DataFrame, target_series: pd.Series, max_lag_steps: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logger()
        
        self.input_df = input_df.copy()
        self.target_series = target_series.copy()
        self.max_lag_steps = max_lag_steps
        
        self._validate_inputs()
        self.logger.info(f"TemporalAnalyzer initialized with {len(self.input_df)} rows. Max lag: {self.max_lag_steps} steps.")

    def _setup_logger(self):
        """Initializes a standard console logger for user feedback."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _validate_inputs(self):
        """Runs automated safety checks to prevent mathematical collapse downstream."""
        # 1. Dimensionality match
        if len(self.input_df) != len(self.target_series):
            self.logger.error("Length mismatch between input dataframe and target series.")
            raise ValueError("input_df and target_series must have the exact same number of rows.")
        
        # 2. Continuous Data Check (NaNs break KDE bandwidth calculations)
        if self.input_df.isna().any().any() or self.target_series.isna().any():
            self.logger.error("NaN values detected. The DMIM KDE requires continuous, uniformly sampled data.")
            raise ValueError("Inputs contain NaNs. Please interpolate or fill missing values before initialization.")
            
        # 3. Intermittent / Variance Collapse Warning
        variances = self.input_df.var()
        zero_var_cols = variances[variances < 1e-5].index.tolist()
        if zero_var_cols:
            self.logger.warning(
                f"Columns with near-zero variance detected: {zero_var_cols}. "
                "Ensure the variance patch in wrappers.py catches this to prevent KDE failure."
            )
            
        # 4. Collinearity Diagnostic (Variance-stealing risk)
        corr_matrix = self.input_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
        if high_corr:
            self.logger.warning(
                f"High collinearity (>0.85) detected involving columns: {high_corr}. "
                "This may distribute sensitivity weights across correlated parameters."
            )

    def get_lagged_data(self, lag_step: int):
        """
        Dynamically generates synchronized X (inputs) and y (target) arrays for a specific lag.
        
        Parameters:
        lag_step (int): The number of rows to shift the input data forward.
        
        Returns:
        tuple: (X_lagged: np.ndarray, y_aligned: np.ndarray) ready for SALib ingestion.
        """
        if lag_step < 0 or lag_step > self.max_lag_steps:
            raise ValueError(f"lag_step must be strictly between 0 and {self.max_lag_steps}")
            
        # Shift inputs forward so past events align with the current target row
        shifted_inputs = self.input_df.shift(lag_step)
        
        # Merge and dynamically drop the top rows where the shift introduced NaNs
        combined = pd.concat([shifted_inputs, self.target_series], axis=1).dropna()
        
        # Strip Pandas overhead and return raw NumPy arrays for maximum processing speed
        X_lagged = combined.iloc[:, :-1].to_numpy()
        y_aligned = combined.iloc[:, -1].to_numpy()
        
        return X_lagged, y_aligned
    

    def export_report(self, profiles: dict, filepath: str = "ts_diagnostic_report.md"):
        """
        Compiles the computed diagnostic profiles and system warnings into a readable Markdown artifact.
        """
        import datetime
        
        with open(filepath, "w") as f:
            # Document Header
            f.write("# Temporal Sensitivity Diagnostics Report\n")
            f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Target Series:** `{self.target_series.name}`\n")
            f.write(f"**Max Lag Steps Checked:** {self.max_lag_steps}\n")
            f.write("---\n\n")
            
            # Summary Table
            f.write("## 1. Systemic Influence Summary\n")
            f.write("| Parameter | Peak Lag (Steps) | Peak $\\delta$ | Prominence | Cumulative AUC |\n")
            f.write("|-----------|:----------------:|:-------------:|:----------:|:--------------:|\n")
            
            for param, profile in profiles.items():
                f.write(f"| **{param}** | {profile.peak_lag_steps} | {profile.peak_delta_score:.4f} | "
                        f"{profile.prominence_ratio:.2f} | {profile.cumulative_auc:.4f} |\n")
            f.write("\n---\n\n")
            
            # Confidence Flags & Warnings Section
            f.write("## 2. Integrity Diagnostics & Warnings\n")
            warnings_found = False
            for param, profile in profiles.items():
                if profile.flags:
                    warnings_found = True
                    f.write(f"### {param}\n")
                    for flag in profile.flags:
                        f.write(f"- ⚠️ **{flag}**\n")
                    f.write("\n")
            
            if not warnings_found:
                f.write("> ✅ **All Parameters Cleared.** No severe collinearity, variance collapse, or low-prominence signals detected.\n")

        self.logger.info(f"Diagnostic report successfully exported to {filepath}")