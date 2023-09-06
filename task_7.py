import time
import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# CryptoCompare API endpoints
BASE_URL = "https://min-api.cryptocompare.com/data/"
HISTO_DAY_URL = BASE_URL + "histoday"
PAIR = "BTC/USDT"  # Trading pair
LIMIT = 200  # Number of data points to retrieve

# Define strategy parameters
short_window = 50
long_window = 200

# Initial account balance (for demonstration)
account_balance = 10000  # USDT

# Streamlit setup
st.title("Crypto Trading Bot and Price Prediction")
st.sidebar.title("Bot Settings")

# User-defined settings
bot_speed = st.sidebar.slider("Bot Speed (seconds)", 1, 60, 10)
initial_balance = st.sidebar.number_input("Initial Account Balance (USDT)", min_value=1, value=10000)
pair = st.sidebar.text_input("Trading Pair (e.g., BTC/USDT)", value="BTC/USDT")
short_window = st.sidebar.number_input("Short Window (days)", min_value=1, value=50)
long_window = st.sidebar.number_input("Long Window (days)", min_value=1, value=200)

# Display user-defined settings
st.sidebar.subheader("Bot Configuration")
st.sidebar.write(f"Initial Balance: {initial_balance} USDT")
st.sidebar.write(f"Trading Pair: {pair}")
st.sidebar.write(f"Short Window: {short_window} days")
st.sidebar.write(f"Long Window: {long_window} days")

col1, col2, col3 = st.columns(3)
start_bot = col1.button("Start Bot")
stop_bot = col2.button("Stop Bot")
predict_button = col3.button("Predict Price")

# Placeholder for price prediction result
predicted_price = None

def fetch_data(pair, limit):
    params = {
        "fsym": pair.split("/")[0],
        "tsym": pair.split("/")[1],
        "limit": limit,
        "aggregate": 1
    }

    response = requests.get(HISTO_DAY_URL, params=params)
    data = response.json()
    
    df = pd.DataFrame(data['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df['close'] = df['close'].astype(float)
    
    return df

def calculate_moving_averages(df, short_window, long_window):
    df['SMA50'] = df['close'].rolling(window=short_window).mean()
    df['SMA200'] = df['close'].rolling(window=long_window).mean()
    return df

def execute_order(order_type, price, quantity):
    global account_balance
    
    if order_type == 'buy':
        st.write(f"Executing BUY order at price {price}, quantity {quantity}")
        account_balance -= price * quantity
        st.write(f"Updated account balance: {account_balance} USDT")
    
    elif order_type == 'sell':
        st.write
    elif order_type == 'sell':
        st.write(f"Executing SELL order at price {price}, quantity {quantity}")
        account_balance += price * quantity
        st.write(f"Updated account balance: {account_balance} USDT")


def fetch_cryptocurrency_data(symbol, interval, limit):
    base_url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": limit,
        "aggregate": 1,  # You can adjust this for different intervals
        "e": "CCCAGG"  # Use the CryptoCompare aggregation
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    
    return data["Data"]["Data"]

def predicted_price():
    # Fetch and store data for Bitcoin (BTC)
        btc_data = fetch_cryptocurrency_data("BTC", "1d", 1000)
        btc_df = pd.DataFrame(btc_data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        btc_df = btc_df[["timestamp", "open", "high", "low", "close", "volume"]]  # Select relevant columns
        

        btc_data = btc_df
        # Feature selection
        features = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # Keep only the selected features
        btc_data = btc_data[features]
        btc_data.fillna(method='ffill', inplace=True)
        # Create input features (X) and target variable (y)
        coin_data = btc_data

        # Convert timestamp to datetime and set it as index
        coin_data['timestamp'] = pd.to_datetime(coin_data['timestamp'], unit='ms')
        coin_data.set_index('timestamp', inplace=True)

        # Extract the 'close' prices for modeling
        prices = coin_data['close']

        # Split data into training and testing sets
        train_size = int(len(prices) * 0.8)
        train, test = prices[:train_size], prices[train_size:]

        # Build ARMA model
        order = (2, 0, 0)  # ARMA(p=2, q=0)
        model = ARIMA(train, order=order)
        model_fit = model.fit()

        # Make predictions
        predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)
        st.write(f"Predicted Price as on the basis of data scraped : {predictions}")


def main():
    global predicted_price 
    if predict_button:
            predicted_price()

    while start_bot:
        # Fetch and preprocess data
        data = fetch_data(pair, LIMIT)
        data = calculate_moving_averages(data, short_window, long_window)

        st.write(f"Data fetched at {data.index[-1]}")
        st.write(f"SMA50: {data['SMA50'].iloc[-1]}, SMA200: {data['SMA200'].iloc[-1]}")


        # Implement a simple crossover strategy
        if data['SMA50'].iloc[-2] < data['SMA200'].iloc[-2] and data['SMA50'].iloc[-1] >= data['SMA200'].iloc[-1]:
            st.write(f"Buy signal at {data.index[-1]}")
            # Hypothetical buy order
            execute_order('buy', data['close'].iloc[-1], 1)
        
        elif data['SMA50'].iloc[-2] > data['SMA200'].iloc[-2] and data['SMA50'].iloc[-1] <= data['SMA200'].iloc[-1]:
            st.write(f"Sell signal at {data.index[-1]}")
            # Hypothetical sell order
            execute_order('sell', data['close'].iloc[-1], 1)
        
        else:
            st.write("Hold")  # Neither buy nor sell conditions met
        
        time.sleep(bot_speed)  # Sleep for user-defined seconds

if __name__ == '__main__':
    main()
    if stop_bot:  # Stop the bot when the "Stop Bot" button is pressed
        st.write("Bot stopped.")

