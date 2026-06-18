import pandas as pd
import numpy as np
import os

def generate_synthetic_data():
    print("Generating synthetic testing data...")
    
    # 1. Generate 100 continuous timesteps of weather data
    timesteps = np.arange(100)
    weather_data = {
        'timestep': timesteps,
        'air_temp': 20 + 5 * np.sin(timesteps / 10) + np.random.normal(0, 0.5, 100),
        'relative_humidity': 50 + 20 * np.cos(timesteps / 10) + np.random.normal(0, 2, 100),
        'solar_irradiance': np.clip(800 * np.sin(timesteps / 15), 0, 1000) + np.random.normal(0, 10, 100)
    }
    weather_df = pd.DataFrame(weather_data)
    
    # 2. Generate Indoor Data (With deliberate duplicates and slight phase delay)
    indoor_timesteps = list(timesteps)
    indoor_temps = 22 + 4 * np.sin((np.array(indoor_timesteps) - 5) / 10) + np.random.normal(0, 0.2, 100)
    
    indoor_df = pd.DataFrame({
        'timestep': indoor_timesteps,
        'T': indoor_temps
    })
    
    # Insert a duplicate sensor reading at timestep 50 to prove the O(1) relational mapping works
    duplicate_row = pd.DataFrame({'timestep': [50], 'T': [indoor_df.loc[50, 'T'] + 0.3]})
    indoor_df = pd.concat([indoor_df, duplicate_row]).sort_values('timestep').reset_index(drop=True)
    
    # 3. Export to the examples folder
    os.makedirs('examples', exist_ok=True)
    weather_df.to_csv('examples/synthetic_weather.csv', index=False)
    indoor_df.to_csv('examples/synthetic_indoor.csv', index=False)
    
    print("Successfully generated examples/synthetic_weather.csv and examples/synthetic_indoor.csv")

if __name__ == "__main__":
    generate_synthetic_data()