import logging
import pandas as pd
import numpy as np
from .wrappers import compute_single_lag_dmim
from .diagnostics import extract_signal_holistic

class TemporalAnalyzer:
    def __init__(self, weather_df: pd.DataFrame, indoor_df: pd.DataFrame, target_col: str, max_lag_steps: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logger()
        
        # Weather must be a continuous 1D timeline (index = timestep)
        self.weather_df = weather_df.copy()
        
        # Indoor can contain thousands of duplicates, must have 'timestep' and target_col
        self.indoor_df = indoor_df[['timestep', target_col]].copy()
        self.target_col = target_col
        self.max_lag_steps = max_lag_steps
        
        self._validate_inputs()
        self.logger.info(f"TemporalAnalyzer initialized. Max lag: {self.max_lag_steps} steps.")

    def _setup_logger(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _validate_inputs(self):
        self.system_warnings = {col: [] for col in self.weather_df.columns}
        if self.weather_df.isna().any().any():
            raise ValueError("Input weather features contain NaNs. Must be continuous.")

    def get_lagged_data(self, lag_step: int):
        """
        O(1) Vectorized Relational Shift. 
        Maps a single shifted weather state to multiple duplicate target measurements instantly.
        """
        # Shift the 1D continuous weather timeline forward
        shifted_weather = self.weather_df.shift(lag_step)
        
        # Join the shifted weather onto the multi-dimensional indoor fact table
        combined = self.indoor_df.join(shifted_weather, on='timestep', how='inner')
        
        # Drop rows where the shift introduced NaNs into the weather subset
        combined = combined.dropna(subset=self.weather_df.columns)
        
        X_lagged = combined[self.weather_df.columns].to_numpy()
        y_aligned = combined[self.target_col].to_numpy()
        
        return X_lagged, y_aligned

    def compute_sweeps(self, num_resamples=10, seed=42):
        self.dmim_curves = {param: {'raw': np.zeros(self.max_lag_steps), 
                                    'balanced': np.zeros(self.max_lag_steps), 
                                    'step': np.zeros(self.max_lag_steps)} 
                            for param in self.weather_df.columns}
        
        for k in range(self.max_lag_steps):
            X_lagged, y_aligned = self.get_lagged_data(k)
            for col_idx, param in enumerate(self.weather_df.columns):
                x_col = X_lagged[:, col_idx]
                scores = compute_single_lag_dmim(x_col, y_aligned, num_resamples, seed)
                self.dmim_curves[param]['raw'][k] = scores['raw']
                self.dmim_curves[param]['balanced'][k] = scores['balanced']
                self.dmim_curves[param]['step'][k] = scores['step']
                
        return self.dmim_curves

    def generate_profiles(self, metric_mapping=None, smooth_window=6):
        if metric_mapping is None: metric_mapping = {}
        self.profiles = {}
        for param, curves in self.dmim_curves.items():
            metric = metric_mapping.get(param, 'balanced')
            self.profiles[param] = extract_signal_holistic(param, curves, metric, smooth_window, self.system_warnings.get(param, []))
        return self.profiles