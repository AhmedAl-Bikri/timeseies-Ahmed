import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

df = pd.read_excel('/content/qatar-monthly-statistics-visitor-arrivals-by-mode-of-enteryecxel.xlsx')

df.rename(columns={
    ' Air': 'Air Arrivals',
    ' Land': 'Land Arrivals',
    ' Sea': 'Sea Arrivals',
    '  Total Visitor Arrivals': 'Total Visitor Arrivals'
}, inplace=True)


df.head()

df['Year'] = pd.to_datetime(df['Month'], format='%Y-%m').dt.year

yearly_data = df.groupby('Year')['Total Visitor Arrivals'].sum().reset_index()

yearly_data.set_index('Year', inplace=True)

print(yearly_data)





"""Comparison"""

train = yearly_data[:-3]
test = yearly_data[-3:]

model = ARIMA(yearly_data['Total Visitor Arrivals'], order=(1, 1, 2))
arima_model_fit = model.fit()

model = SARIMAX(yearly_data['Total Visitor Arrivals'], order=(1, 1, 2),seasonal_order=(1, 1, 2, 6))
sarimax_model_fit = model.fit()

arima_forecast = arima_model_fit.forecast(steps=3)
sarimax_forecast = sarimax_model_fit.get_forecast(steps=3).predicted_mean

def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, forecast) * 100
    return mae, mse, rmse, mape

arima_metrics = calculate_metrics(test['Total Visitor Arrivals'], arima_forecast)

sarimax_metrics = calculate_metrics(test['Total Visitor Arrivals'], sarimax_forecast)

print("ARIMA Model Metrics:")
print(f"MAE: {arima_metrics[0]}")
print(f"MSE: {arima_metrics[1]}")
print(f"RMSE: {arima_metrics[2]}")
print(f"MAPE: {arima_metrics[3]}%\n")

print("SARIMAX Model Metrics:")
print(f"MAE: {sarimax_metrics[0]}")
print(f"MSE: {sarimax_metrics[1]}")
print(f"RMSE: {sarimax_metrics[2]}")
print(f"MAPE: {sarimax_metrics[3]}%\n")

# Metrics for ARIMA and SARIMAX models
metrics = {
    'Model': ['ARIMA', 'SARIMAX'],
    'MAE': [2240014.13, 782211.0],
    'MSE': [5861307500720.43, 1289832392161.67],
    'RMSE': [2421013.73, 1135707.88],
    'MAPE': [71.05, 28.51]
}

# Convert metrics to a DataFrame for plotting
import pandas as pd

metrics_df = pd.DataFrame(metrics)

# Plotting the comparison of models
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# MAE Bar Chart
ax[0, 0].bar(metrics_df['Model'], metrics_df['MAE'], color=['red', 'blue'])
ax[0, 0].set_title('Mean Absolute Error (MAE)')
ax[0, 0].set_ylabel('Value')

# MSE Bar Chart
ax[0, 1].bar(metrics_df['Model'], metrics_df['MSE'], color=['red', 'blue'])
ax[0, 1].set_title('Mean Squared Error (MSE)')
ax[0, 1].set_ylabel('Value')

# RMSE Bar Chart
ax[1, 0].bar(metrics_df['Model'], metrics_df['RMSE'], color=['red', 'blue'])
ax[1, 0].set_title('Root Mean Squared Error (RMSE)')
ax[1, 0].set_ylabel('Value')

# MAPE Bar Chart
ax[1, 1].bar(metrics_df['Model'], metrics_df['MAPE'], color=['red', 'blue'])
ax[1, 1].set_title('Mean Absolute Percentage Error (MAPE)')
ax[1, 1].set_ylabel('Value')

# Display the plots
plt.tight_layout()
plt.show()



# Forecasting for next 3 years
forecast_steps = 3
arima_forecast = arima_model_fit.forecast(steps=forecast_steps)
sarimax_forecast = sarimax_model_fit.get_forecast(steps=forecast_steps).predicted_mean

# Prepare forecast years (2024, 2025, 2026)
forecast_years = range(2024, 2024 + forecast_steps)

# Calculate error metrics for both models
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, forecast) * 100
    return mae, mse, rmse, mape

# Evaluate ARIMA model
arima_metrics = calculate_metrics(test['Total Visitor Arrivals'], arima_forecast)

# Evaluate SARIMAX model
sarimax_metrics = calculate_metrics(test['Total Visitor Arrivals'], sarimax_forecast)

# Print performance metrics
print("ARIMA Model Metrics:")
print(f"MAE: {arima_metrics[0]}")
print(f"MSE: {arima_metrics[1]}")
print(f"RMSE: {arima_metrics[2]}")
print(f"MAPE: {arima_metrics[3]}%\n")

print("SARIMAX Model Metrics:")
print(f"MAE: {sarimax_metrics[0]}")
print(f"MSE: {sarimax_metrics[1]}")
print(f"RMSE: {sarimax_metrics[2]}")
print(f"MAPE: {sarimax_metrics[3]}%\n")

# Plotting the comparison for the next 3 years (2024, 2025, 2026)
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Total Visitor Arrivals'], label='Training Data')
plt.plot(test.index, test['Total Visitor Arrivals'], label='Actual Data', color='black')
plt.plot(forecast_years, arima_forecast, label='ARIMA Forecast', color='red')
plt.plot(forecast_years, sarimax_forecast, label='SARIMAX Forecast', color='blue')
plt.title('ARIMA vs SARIMAX Forecast Comparison')
plt.xlabel('Year')
plt.ylabel('Total Visitor Arrivals')
plt.xticks(np.append(train.index, forecast_years))  # Ensure X-axis includes forecast years
plt.legend()
plt.show()
