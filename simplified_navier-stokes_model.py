import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

# --- Step 1: Create CSV Data ---
# Generate synthetic temperature data for 14 days
days = np.arange(1, 15)
actual_temperatures = [22.5, 23.0, 23.2, 23.8, 24.0, 24.5, 24.8, 25.0, 25.3, 25.7, 26.0, 26.3, 26.5, 27.0]

# Create DataFrame and save to CSV
data = pd.DataFrame({"Day": days, "Temperature": actual_temperatures})
data.to_csv("temperature_data.csv", index=False)

# --- Step 2: Load the CSV Data ---
df = pd.read_csv("temperature_data.csv")

# --- Step 3: Forecasting with Simplified Navier-Stokes Equation ---
def forecast_temperature(data, num_days, diffusion_rate=0.1):
    """
    Predict temperatures using a simplified 1D diffusion model based on the Navier-Stokes principle.
    """
    # Convert data to numpy array
    temps = np.array(data)

    # Forecast temperatures for the specified number of days
    forecast = temps.copy()
    for _ in range(num_days):
        # Apply diffusion (averaging neighboring temperatures)
        forecast = convolve1d(forecast, weights=[diffusion_rate, 1 - 2 * diffusion_rate, diffusion_rate],
                              mode='nearest')

    return forecast

# Forecast temperatures for the next 7 days
num_forecast_days = 7
forecasted_temps = forecast_temperature(df["Temperature"], num_forecast_days)

# Append forecasted days
forecast_days = np.arange(1, len(forecasted_temps) + 1)

# --- Step 4: Plot and Compare ---
plt.figure(figsize=(10, 6))
plt.plot(df["Day"], df["Temperature"], label="Actual Temperature", marker='o')
plt.plot(forecast_days, forecasted_temps, label="Forecasted Temperature", marker='x')
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Forecast using Simplified Navier-Stokes Model")
plt.legend()
plt.grid(True)

plt.xlim(1, 7)

plt.show()

# --- Step 5: Calculate Error ---
# Assuming the first 'len(df)' entries are actual data and the remaining are forecasted
errors = np.abs(df["Temperature"] - forecasted_temps[:len(df)])
print("\nMean Absolute Error (MAE):", np.mean(errors))
