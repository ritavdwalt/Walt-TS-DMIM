import os
import logging
import pandas as pd
import numpy as np
from .wrappers import compute_single_lag_dmim
from .diagnostics import extract_signal_holistic
from SALib.analyze import delta
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from tqdm import tqdm

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

    def compute_sweeps(self, num_resamples: int = 10, seed: int = 42, n_jobs: int = 1):
        """
        Executes DMIM across all temporal lags.
        n_jobs: Number of CPU cores to allocate. Set to -1 to use all cores.
        """
        self.logger.info(f"Initiating Temporal DMIM Sweeps (n_jobs={n_jobs})...")
        self.dmim_curves = {}

        # The isolated function that runs on a single core
        def _compute_single_lag(x_col, y_array, k):
            if np.var(x_col) < 1e-5:
                return k, {'raw': 0.0, 'balanced': 0.0, 'step': 0.0}
            
            problem = {'num_vars': 1, 'names': ['X'], 'bounds': [[np.min(x_col), np.max(x_col)]]}
            x_reshaped = x_col.reshape(-1, 1)
            
            # CRITICAL: Force underlying C-libraries to strictly use 1 thread to prevent thrashing
            with threadpool_limits(limits=1, user_api='blas'):
                results = delta.analyze(problem, x_reshaped, y_array, num_resamples=num_resamples, seed=seed, print_to_console=False)
            
            return k, {
                'raw': results.get('delta_raw', [0.0])[0],
                'balanced': results.get('delta_balanced', [0.0])[0],
                'step': results.get('delta_step', [0.0])[0]
            }

        # Loop over every weather parameter
        for param, matrix in self.matrices.items():
            self.logger.info(f" Sweeping parameter: {param}")
            num_lags = matrix.shape[1]
            
            # Parallelize the lag computations with a progress bar
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_single_lag)(matrix[:, k], self.y_array, k) 
                for k in tqdm(range(num_lags), desc=f"Processing {param}", unit="lag")
            )
            
            # Reconstruct the ordered arrays from the parallel output
            param_results = {'raw': np.zeros(num_lags), 'balanced': np.zeros(num_lags), 'step': np.zeros(num_lags)}
            for k, scores in parallel_results:
                param_results['raw'][k] = scores['raw']
                param_results['balanced'][k] = scores['balanced']
                param_results['step'][k] = scores['step']
                
            self.dmim_curves[param] = param_results
            
        return self.dmim_curves

    def generate_profiles(self, metric_mapping=None, smooth_window=6):
        if metric_mapping is None: metric_mapping = {}
        self.profiles = {}
        for param, curves in self.dmim_curves.items():
            metric = metric_mapping.get(param, 'balanced')
            self.profiles[param] = extract_signal_holistic(param, curves, metric, smooth_window, self.system_warnings.get(param, []))
        return self.profiles
    
    def compute_acf(self):
        """Generates continuous ACF baseline curves using row-based integer steps."""
        self.logger.info("Computing continuous ACF baseline...")
        self.acf_curves = {}
        df_filled = self.weather_df.fillna(0)
        
        for col in df_filled.columns:
            array = df_filled[col].to_numpy()
            array_centered = array - np.mean(array)
            variance = np.sum(array_centered ** 2)
            
            if variance == 0:
                self.acf_curves[col] = np.zeros(self.max_lag_steps)
            else:
                acf_full = np.correlate(array_centered, array_centered, mode='full')
                center_index = len(array) - 1
                limit = min(self.max_lag_steps, len(array))
                curve = acf_full[center_index : center_index + limit] / variance
                
                # Pad if data is shorter than max_lag_steps
                if len(curve) < self.max_lag_steps:
                    curve = np.pad(curve, (0, self.max_lag_steps - len(curve)))
                self.acf_curves[col] = curve
                
        return self.acf_curves
    
    def export_results(self, output_dir: str, file_prefix: str = "run001"):
        """Exports raw curves, ACF, and metrics to CSV for external analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Export Topological Metrics
        if hasattr(self, 'profiles'):
            metrics_list = []
            for param, prof in self.profiles.items():
                prof_dict = vars(prof).copy()
                prof_dict['parameter'] = param
                metrics_list.append(prof_dict)
            pd.DataFrame(metrics_list).to_csv(os.path.join(output_dir, f"{file_prefix}_metrics.csv"), index=False)
            
        # 2. Export Raw DMIM Curves
        if hasattr(self, 'dmim_curves'):
            dmim_df = pd.DataFrame()
            for param, curves in self.dmim_curves.items():
                for metric, array in curves.items():
                    dmim_df[f"{param}_{metric}"] = array
            dmim_df.to_csv(os.path.join(output_dir, f"{file_prefix}_raw_dmim.csv"), index_label="lag_step")
            
        # 3. Export ACF Baseline
        if hasattr(self, 'acf_curves'):
            pd.DataFrame(self.acf_curves).to_csv(os.path.join(output_dir, f"{file_prefix}_acf_baseline.csv"), index_label="lag_step")
            
        self.logger.info(f"✅ Data exported to {output_dir}/ with prefix '{file_prefix}'")