# Stock Price Analysis & Prediction using LSTM
###  Overview

This project performs a stock market price analysis and builds an LSTM-based predictive model to forecast future stock prices. It uses real historical data from Yahoo Finance and applies data preprocessing, feature engineering, technical analysis, and deep learning for accurate predictions.

### Key Objectives

* Data Collection: Download real stock data from Yahoo Finance

* Exploratory Analysis: Visualize price movements, volume, and trends

* Technical Analysis: Compute moving averages and trading indicators

* Risk Analysis: Measure volatility and risk using statistical methods

* Machine Learning: Build an LSTM neural network for price prediction

* Evaluation: Test and compare prediction accuracy with real data

### Features

1. Fetch real-time historical stock data
2. Perform EDA with charts and insights
3. Calculate technical indicators (Moving Averages, Volatility)
4. Model future stock prices using LSTM
5. Evaluate accuracy with RMSE and visualization

### Installation

Clone the repository:

git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction


Install dependencies:

pip install -r requirements.txt


Or open in Google Colab:


📦 Requirements

Create a requirements.txt file with:

pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
yfinance

📊 Steps in the Notebook
1. Data Collection

Download stock data from Yahoo Finance using yfinance:

import yfinance as yf
data = yf.download('AAPL', start='2018-01-01', end='2023-12-31')
print(data.head())

2. Exploratory Data Analysis (EDA)

Visualize closing price and volume trends:

plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Closing Price')
plt.title('Apple Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

3. Technical Analysis

Compute moving averages:

data['MA50'] = data['Close'].rolling(50).mean()
data['MA200'] = data['Close'].rolling(200).mean()


Plot them:

plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Closing Price')
plt.plot(data['MA50'], label='50-Day MA')
plt.plot(data['MA200'], label='200-Day MA')
plt.legend()
plt.show()

4. Risk Analysis

Calculate volatility using standard deviation:

volatility = data['Close'].pct_change().std()
print(f"Volatility: {volatility}")

5. Data Preprocessing for LSTM

Normalize data using MinMaxScaler

Create sequences for LSTM input

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[['Close']])

6. Build & Train LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=64, epochs=10)

7. Prediction & Evaluation

Plot actual vs predicted prices:

predictions = model.predict(x_test)
plt.figure(figsize=(12,6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.legend()
plt.show()


Calculate RMSE:

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"RMSE: {rmse}")

📈 Example Output




✅ Future Enhancements

Add multiple stocks and compare performance

Include more features: Open, High, Low, Volume

Use advanced architectures like GRU or Transformers

Deploy as a web app using Streamlit or Flask

📜 License

This project is licensed under the MIT License.
