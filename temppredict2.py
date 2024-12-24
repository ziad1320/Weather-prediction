import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your temperature data
data = pd.read_csv('temperature_data.csv')  # Ensure this file is present
data['date'] = pd.to_datetime(data['date'])  # Ensure the date is in datetime format
data.set_index('date', inplace=True)  # Set the date as the index

# Number of days you want to use for prediction
N = 5  # Use last 5 days to predict next 5 days

# Prepare the feature and target variables
X = []
y = []

# Loop through the dataset and prepare data for model training
for i in range(N, len(data) - 5):  # Ensure we have 5 days to predict
    X.append(data['temperature'].iloc[i-N:i].values)  # Last N days as input
    y.append(data['temperature'].iloc[i:i+5].values)  # Next 5 days as output

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the next 5 days' temperatures for the first test sample (only one sample)
y_pred = model.predict(X_test)

# Select the first test sample for the plot
predicted = y_pred[0]
actual = y_test[0]

# Evaluate the model (mean squared error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plotting actual vs predicted temperatures for the next 5 days using columns (bar plot)
plt.figure(figsize=(10, 6))

# Set the x-ticks to represent days for easier comparison
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']

# Set positions for bars
bar_width = 0.35
x_positions = np.arange(len(days))

# Plot the predicted and actual temperatures as bar columns for the first test sample
plt.bar(x_positions - bar_width / 2, predicted, width=bar_width, label='Predicted', alpha=0.7)
plt.bar(x_positions + bar_width / 2, actual, width=bar_width, label='Actual', alpha=0.7)

# Labeling the axes and the plot
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.title('Predicted vs Actual Temperatures for the Next 5 Days (Single Test Sample)')
plt.xticks(x_positions, days)
plt.legend()
plt.tight_layout()
plt.show()

# Optionally, you can predict for the next 5 days after the last available data
last_N_days = data['temperature'].iloc[-N:].values.reshape(1, -1)
future_pred = model.predict(last_N_days)

print(f'Predicted temperatures for the next 5 days: {future_pred[0]}')
