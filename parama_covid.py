import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_percentage_error

# Load dataset
file_path = "us.csv"
df = pd.read_csv(file_path)

# Convert 'date' column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Ensure 'cases' column is numeric and drop NaN values
df['cases'] = pd.to_numeric(df['cases'], errors='coerce').dropna()

# Split data into training (80%) and testing (20%) sets
split_index = int(len(df) * 0.8)
train, test = df.iloc[:split_index], df.iloc[split_index:]

# Apply 7-day differencing to remove trend
train_diff = train['cases'].diff(7).dropna()

# Since pmdarima is not available, we'll use a manual grid search for ARIMA order selection

import itertools
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

# Load dataset again
df = pd.read_csv(file_path)

# Convert 'date' column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Ensure 'cases' column is numeric and drop NaN values
df['cases'] = pd.to_numeric(df['cases'], errors='coerce').dropna()

# Split data into training (80%) and testing (20%) sets
split_index = int(len(df) * 0.8)
train, test = df.iloc[:split_index], df.iloc[split_index:]

# Apply 7-day differencing to remove trend
train_diff = train['cases'].diff(7).dropna()

# Define parameter grid for ARIMA (p, d, q)
p = range(0, 4)
d = range(0, 2)
q = range(0, 4)
pdq_combinations = list(itertools.product(p, d, q))

best_aic = float("inf")
best_order = None
best_model = None

# Grid search for the best ARIMA order
for order in pdq_combinations:
    try:
        model = ARIMA(train_diff, order=order).fit()
        if model.aic < best_aic:
            best_aic = model.aic
            best_order = order
            best_model = model
    except:
        continue

# Fit the best ARIMA model
arima_model = ARIMA(train_diff, order=best_order).fit()

# In-sample predictions (train set)
train_pred = arima_model.fittedvalues

# Compute MAPE on training set
mape_train = mean_absolute_percentage_error(train_diff, train_pred)

# Forecasting for test period
forecast_steps = len(test)
forecast_diff = arima_model.forecast(steps=forecast_steps)

# Convert forecasted differenced values back to original scale
forecast_cases = test['cases'].shift(7) + forecast_diff

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['cases'], label='Actual Cases', marker='o')
plt.plot(test.index, forecast_cases, label='Forecasted Cases', linestyle='dashed', marker='x')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.title('Actual vs Forecasted Cases')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Display MAPE of training set
print(mape_train, best_order)


# Align actual and forecasted values by ensuring they have the same valid index
actual_cases_aligned = test['cases'].dropna().iloc[:len(forecast_cases.dropna())]
forecast_cases_aligned = forecast_cases.dropna().iloc[:len(actual_cases_aligned)]

# Compute MAPE on test set
mape_test = mean_absolute_percentage_error(actual_cases_aligned, forecast_cases_aligned)

# Display MAPE of test set
mape_test


# Check sizes of actual and forecasted aligned data
actual_size = len(actual_cases_aligned)
forecast_size = len(forecast_cases_aligned)

# Calculate the number of rows dropped
rows_dropped = len(test['cases'].dropna()) - actual_size

# Display sizes and dropped rows
actual_size, forecast_size, rows_dropped
