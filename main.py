import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tools import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from stepwise_regression import step_reg
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import chi2
import tensorflow as tf
import warnings

warnings.simplefilter(action='ignore')

#Read dataset from the directory it is saved in
data = pd.read_csv('/Users/tejoharshitha/Desktop/energy_dataset.csv')

# Convert time to datetime object and set it as index
data['time'] = pd.to_datetime(data['time'], utc=True, infer_datetime_format=True)
data = data.set_index('time')
data.info()

#Total number of missing values
print('There are {} missing values or NaNs in df_energy.' .format(data.isnull().values.sum()))
# Find the number of NaNs in each column
data.isnull().sum(axis=0)
df_copy = data.iloc[:, :-1].copy()
print(df_copy.isnull().sum(axis=0))

#Plot of load actual with missing values
plt.figure(figsize=(8, 5))
sns.histplot(df_copy['total load actual'], kde=True, color='blue')
plt.title(f'Density Plot of total load actual')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

#Handle missing values
def seasonal_decompose_and_interpolate(df, column_name):

    # Fill missing values in the time series
    imputed_indices = df[df[column_name].isnull()].index

    # Apply STL decomposition
    stl = STL(df_copy[column_name].interpolate(), seasonal=31)
    res = stl.fit()

    # Extract the seasonal and trend components
    seasonal_component = res.seasonal

    # Create the deseasonalized series
    df_deseasonalized = df[column_name] - seasonal_component

    # Interpolate missing values in the deseasonalized series
    df_deseasonalized_imputed = df_deseasonalized.interpolate(method="linear")

    # Add the seasonal component back to create the final imputed series
    df_imputed = df_deseasonalized_imputed + seasonal_component

    # Update the original dataframe with the imputed values
    df.loc[imputed_indices, column_name] = df_imputed[imputed_indices]

    return df

# Process each column
for column in df_copy.columns:
    df_copy = seasonal_decompose_and_interpolate(df_copy, column)

print(df_copy.isnull().sum(axis=0))

#Show distribution of load actual after seasonal deompose (without missing values)
plt.figure(figsize=(8, 5))
sns.histplot(df_copy['total load actual'], kde=True, color='blue')
plt.title(f'Density Plot of total load actual after imputation')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# Plot time series of Price vs time
plt.figure(figsize=(12, 6))
plt.plot(data['price actual'])
plt.title("Electricity prices in Spain from 2015 to 2018")
plt.xlabel("Time")
plt.ylabel("Price in EUR")
plt.grid(True)
plt.show()

#calculate acf of raw data
acf = cal_autocorr(data['price actual'],50, title='ACF of Original data')

#Stationarity check on raw data
cal_rolling(data['price actual'])
ADF_Cal(data['price actual'])
kpss_test(data['price actual'])
df_copy['Price'] = data['price actual']
plt.figure(figsize=(16, 12))

#Pearson correlation heatmap
sns.heatmap(df_copy.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Heatmap of all features')
plt.show()

# splitting data into test and train
X = df_copy.drop(['Price'], axis=1)
y = df_copy['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print('Length of X_train and y_train:', len(X_train))
print('Length of X_test and y_test:', len(X_test))

#first order differencing
data['Price'] = data['price actual'] - data['price actual'].shift(1)
data = data.drop(data.index[0])
ADF_Cal(data['Price'])
kpss_test(data['Price'])
cal_rolling(data['Price'])
df_copy = df_copy.drop(df_copy.index[0])
ACF_PACF_Plot(data['Price'], 50)

print("--------------The dataset is stationary--------------")

# Create the STL model
stl = STL(df_copy['Price'], seasonal=25)

# Fit the model
res = stl.fit()

# Plot the trend, seasonality, and remainder
plt.figure(figsize=(12, 6))
fig = res.plot()
# Add title, x-label, y-label, and legend
plt.title('STL Decomposition of Price Data')
plt.show()

T = res.trend
S = res.seasonal
R = res.resid
print("\n")
def str_trend_seasonal(T, S, R):
    F = np.maximum(0 ,1- np.var(np.array(R))/np.var(np.array(T+R)))
    print(f'The strength of trend for this data set is {100*F:.3f}%')

    FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
    print(f'The strength of seasonality for this data set is {100*FS:.3f}%')

str_trend_seasonal(T,S,R)

df_copy = df_copy.drop(df_copy.index[0])
ACF_PACF_Plot(y_train, 50)

#Holt-Winter method
Holt_Winter_model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=24).fit()
HWforecast = Holt_Winter_model.forecast(steps=len(y_test))

mse_HW = mean_squared_error(y_test, HWforecast)
rmse_HW = np.sqrt(mse_HW)
mae_HW = mean_absolute_error(y_test, HWforecast)
print('MSE for Winter-Holt method:', np.round(mse_HW, 2))
print('RMSE for Winter-Holt method:', np.round(rmse_HW, 2))
print("MAE for Holt-Winter method:", np.round(mae_HW, 2))

plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(y_train, label='Training Data', color='blue')

# Plot actual test data
plt.plot(y_test.index, y_test, label='Test Data', color='green')

# Plot Holt-Winters forecast
plt.plot(y_test.index, HWforecast, label='Holt-Winters Forecast', color='red')

plt.title('Holt-Winters Forecasting')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
#End of Holt-winter method

#Average forecasting method

# Average forecasting
_, forecast_average = average_forecasting(y_train, y_test)
mse_average = mean_squared_error(y_test, forecast_average)
rmse_average = np.sqrt(mse_average)
mae_average = mean_absolute_error(y_test, forecast_average)
print('MSE for Average forecasting:', np.round(mse_average, 2))
print('RMSE for Average forecasting:', np.round(rmse_average, 2))
print("MAE for Average forecasting:", np.round(mae_average, 2))

plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(y_train, label='Training Data', color='blue')

# Plot actual test data
plt.plot(y_test.index, y_test, label='Test Data', color='green')

# Plot forecast
plt.plot(y_test.index, forecast_average, label='Average Forecast', color='red')

plt.title('Average Forecasting')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
#End of Average Forecasting

#Drift forecast
_, forecast_Drift = drift_forecasting(y_train, y_test)
mse_Drift = mean_squared_error(y_test, forecast_Drift)
rmse_Drift = np.sqrt(mse_Drift)
mae_Drift = mean_absolute_error(y_test, forecast_Drift)
print('MSE for Drift forecasting:', np.round(mse_Drift, 2))
print('RMSE for Drift forecasting:', np.round(rmse_Drift, 2))
print("MAE for Drift forecasting:", np.round(mae_Drift, 2))

plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(y_train, label='Training Data', color='blue')

# Plot actual test data
plt.plot(y_test.index, y_test, label='Test Data', color='green')

# Plot forecast
plt.plot(y_test.index, forecast_Drift, label='Drift Forecast', color='red')

plt.title('Drift Forecasting')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
#End of Drift Forecasting method

#Naive method
_, forecast_Naive = Naive_forecasting(y_train, y_test)
mse_Naive = mean_squared_error(y_test, forecast_Naive)
rmse_Naive = np.sqrt(mse_Naive)
mae_Naive = mean_absolute_error(y_test, forecast_Naive)
print('MSE for Naive forecasting:', np.round(mse_Naive, 2))
print('RMSE for Naive forecasting:', np.round(rmse_Naive, 2))
print("MAE for Naive forecasting:", np.round(mae_Naive, 2))

plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(y_train, label='Training Data', color='blue')

# Plot actual test data
plt.plot(y_test.index, y_test, label='Test Data', color='green')

# Plot forecast
plt.plot(y_test.index, forecast_Naive, label='Naive Forecast', color='red')

plt.title('Naive Forecasting')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
#End of Naive Forecasting method

#Simple-Exponential Smoothing
_, forecast_SES = ses(y_train, y_test, y_train[0], alpha=0.9)
mse_SES = mean_squared_error(y_test, forecast_SES)
rmse_SES = np.sqrt(mse_SES)
mae_SES = mean_absolute_error(y_test, forecast_SES)
print('MSE for SES forecasting:', np.round(mse_SES, 2))
print('RMSE for SES forecasting:', np.round(rmse_SES, 2))
print("MAE for SES forecasting:", np.round(mae_SES, 2))

plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(y_train, label='Training Data', color='blue')

# Plot actual test data
plt.plot(y_test.index, y_test, label='Test Data', color='green')

# Plot forecast
plt.plot(y_test.index, forecast_SES, label='SES Forecast', color='red')

plt.title('Simple Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
#End of SES

# Compute the correlation matrix
correlation_matrix = X_train.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Set plot title
plt.title('Heatmap of Correlation Matrix for X_train')

# Show the plot
plt.show()

# Standardize the data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=19, random_state=6313)
X_pca = pca.fit_transform(X_train_scaled)

# Plot cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()

# Assess collinearity using condition number and singular value decomposition
condition_number = np.linalg.cond(X_train_scaled)
print('Condition Number:', condition_number)

#SVD analysis
X_svd = X_train_scaled.to_numpy()
H = np.matmul(X_svd.T, X_svd)
s, d, v = np.linalg.svd(H)
print('Singular Values:', d)

print("The condition number is high so performing VIF and backward stepwise to exclude columns with high collinearity")

#VIF analysis

vif = pd.DataFrame()
vif["Variable"] = X_train_scaled.columns
vif["VIF"] = [variance_inflation_factor(X_train_scaled.values, i) for i in range(X_train_scaled.shape[1])]
# Identify variables with high VIF and consider removing them
high_vif_variables = vif[vif["VIF"] > 10]
X_train_scaled_reduced = X_train_scaled.drop(high_vif_variables["Variable"].tolist(), axis=1)
X_train_scaled_reduced.index = y_train.index
# Add constant term for OLS model
X_train_scaled_reduced = sm.add_constant(X_train_scaled_reduced)
# Fit OLS model
result_ols_reduced = sm.OLS(y_train, X_train_scaled_reduced).fit()
# Show summary
print(result_ols_reduced.summary())

#backward stepwise regression
backselect = step_reg.backward_regression(X_train, y_train, 0.05, verbose=False)
print('Selected Features:', backselect)

# Fit OLS model with selected features
X_train_ols = X_train[backselect]
X_test_ols = X_test[backselect]
X_train_scaled_ols = pd.DataFrame(scaler.fit_transform(X_train_ols), columns=X_train_ols.columns)
X_test_scaled_ols = pd.DataFrame(scaler.transform(X_test_ols), columns=X_test_ols.columns)

# Add constant term for OLS model
X_train_scaled_ols = sm.add_constant(X_train_scaled_ols)
X_test_scaled_ols = sm.add_constant(X_test_scaled_ols)
X_train_scaled_ols.index = X_train_ols.index
# Fit OLS model
result_ols = sm.OLS(y_train, X_train_scaled_ols).fit()
print(result_ols.summary())

# Evaluate OLS model
pred_ols = result_ols.predict(X_test_scaled_ols)
mse_ols = mean_squared_error(y_test, pred_ols)
rmse_ols = np.sqrt(mse_ols)
mae_ols = mean_absolute_error(y_test, pred_ols)
print('MSE for OLS model:', np.round(mse_ols, 2))
print('RMSE for OLS model:', np.round(rmse_ols, 2))
print('MAE for OLS model:', np.round(mae_ols, 2))
ols_residuals = result_ols.resid
q_ols = cal_Q_value(ols_residuals, 'OLS Residuals', 50)
print('Q Value of OLS residuals:', np.round(q_ols, 2))
print('Mean of residuals for OLS:', np.mean(ols_residuals))
print('Variance of residuals for OLS:', np.var(ols_residuals))

# Visualize the results for the first 100 data points
plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(y_test[:100], label='Training Data', color='blue')

# Plot OLS forecast for the first 100 data points
plt.plot(y_test.index[:100], pred_ols[:100], label='OLS Prediction', color='red')

plt.title('Multiple Linear Regression Model - 1-Step Ahead Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
# Plot training data
plt.plot(y_train, label='Training Data', color='blue')

# Plot actual test data
plt.plot(y_test.index, y_test, label='Test Data', color='green')

# Plot OLS forecast
plt.plot(y_test.index, pred_ols, label='OLS Prediction', color='red')

plt.title('Multiple linear regression model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


cal_gpac(acf, 10, 10)
# na=24
# nb=24
# model = sm.tsa.ARIMA(y_train, order=(na,0,nb)).fit()
# for i in range(na):
#     print("Estimated AR coefficient a{}".format(i), "is:", round(model.params[i], 3))
# for i in range(nb):
#     print("Estimated MA coefficient b{}".format(i), "is:", round(model.params[i+na], 3))
# print(model.summary())
#
# arima_predictions = model.get_prediction(start=0, end=len(y_train)-1, dynamic=False)
# arima_predicted_mean = arima_predictions.predicted_mean
# e = y_train - arima_predicted_mean
# re = cal_autocorr(np.array(e), 100, 'ACF of residuals 2')
# plot_pacf(np.array(e), ax=plt.gca(), lags=48)
# Q = len(y_train) * np.sum(np.square(re[1:]))
# DOF = 100 - na - nb
# alfa = 0.10
# chi_critical = chi2.ppf(1 - alfa, DOF)
# print('Chi critical:', chi_critical)
# print('Q Value:', Q)
# print('Alfa value for 99% accuracy:', alfa)
# if Q < chi_critical:
#     print("The residual is white ")
# else:
#     print("The residual is NOT white ")
#
# cal_gpac(re,7,7)


model = SARIMAX(y_train, order=(1,1,6), seasonal_order = (1,1,6,24))
result = model.fit(low_memory=True)
print(result.summary())

sarima_predictions = result.get_prediction(start=0, end=len(y_train)-1, dynamic=False)
sarima_predicted_mean = sarima_predictions.predicted_mean
e = y_train - sarima_predicted_mean
re = cal_autocorr(np.array(e), 120, 'ACF of residuals 2')
plot_pacf(np.array(e), ax=plt.gca(), lags=120)
Q = len(y_train) * np.sum(np.square(re[1:]))
DOF = 108
alfa = 0.01
chi_critical = chi2.ppf(1 - alfa, DOF)
print('Chi critical:', chi_critical)
print('Q Value:', Q)
print('Alfa value for 99% accuracy:', alfa)
if Q < chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")

residual_variance = np.var(e)
parameter_covariance = result.cov_params()

print("Estimated Variance of the Error:", residual_variance)
print("Estimated Covariance of the Estimated Parameters:")
print(parameter_covariance)

# Perform zero-pole cancellation operation and display the final coefficient confidence interval.
# Note: SARIMAX models do not have poles and zeros like transfer functions, so this step may not be applicable.

# Make multiple step ahead predictions for the duration of the test data set.
sarima_forecast = result.get_forecast(steps=len(y_test))
sarima_forecast_mean = sarima_forecast.predicted_mean

# Check the variance of the residual errors versus the variance of the forecast errors.
forecast_errors = y_train - sarima_forecast_mean
forecast_error_variance = np.var(forecast_errors)

print("Variance of Residual Errors:", residual_variance)
print("Variance of Forecast Errors:", forecast_error_variance)

# Plot the predicted values versus the true values (test set).
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Values', color='blue')
plt.plot(sarima_forecast_mean, label='SARIMA Forecast', color='red')
plt.title('SARIMA Model Forecast vs True Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

#LSTM code starts here
split_ind = int(len(df_copy)*0.8)
train = df_copy[:split_ind]
scalers={}
for i in df_copy.columns:
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s
test = df_copy[split_ind:]
for i in df_copy.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s

def split_series(series, n_past, n_future):
  # n_past ==> no of past observations
  # n_future ==> no of future observations
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)

n_past = 200
n_future = 24
n_features = 20

X_train, y_train = split_series(train.values,n_past, n_future)
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
#y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
X_test, y_test = split_series(test.values,n_past, n_future)
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
#y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))

encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]

decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)

model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)
model_e1d1.summary()

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
history_e1d1=model_e1d1.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test),batch_size=64,verbose=0,callbacks=[reduce_lr])

pred_e1d1=model_e1d1.predict(X_test)

for index,i in enumerate(df_copy.columns):
    scaler = scalers['scaler_'+i]
    pred_e1d1[:,:,index]=scaler.inverse_transform(pred_e1d1[:,:,index])
    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])

plt.figure(figsize=(10, 6))
plt.plot(history_e1d1.history['loss'], label='train')
plt.plot(history_e1d1.history['val_loss'], label='validation')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test[:, 0, 0], label='Actual')
plt.plot(pred_e1d1[:, 0, 0], label='Predicted')
plt.title(f'Actual vs. Predicted for 1-Step')
plt.xlabel('Data Points')
plt.ylabel('Scaled Values')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test[:, 23, 0], label='Actual')
plt.plot(pred_e1d1[:, 23, 0], label='Predicted')
plt.title(f'Actual vs. Predicted for h-Step')
plt.xlabel('Data Points')
plt.ylabel('Scaled Values')
plt.legend()
plt.show()

for index, i in enumerate(df_copy.columns):
    mae = mean_absolute_error(y_test[:, :, index], pred_e1d1[:, :, index])
    mse = mean_squared_error(y_test[:, :, index], pred_e1d1[:, :, index])
    print(f'Metrics for {i}:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

# Calculate and display overall accuracy metrics for the loaded model
mae_overall_loaded = np.mean(np.abs(y_test - pred_e1d1))
mse_overall_loaded = np.mean(np.square(y_test - pred_e1d1))

print('Overall Metrics for the LSTM Model:')
print(f'Overall Mean Absolute Error: {mae_overall_loaded}')
print(f'Overall Mean Squared Error: {mse_overall_loaded}')
