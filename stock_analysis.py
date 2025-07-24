"""
# 📈 Stock Price Analysis & Strategy Project

This Python script:
- Downloads historical stock data (using yfinance)
- Calculates technical indicators: Moving Averages, RSI, Bollinger Bands
- Plots candlestick chart with indicators (Plotly)
- Backtests a simple Moving Average Crossover strategy
- Forecasts future stock prices (ARIMA)

## 🛠 Libraries
pip install yfinance pandas matplotlib seaborn plotly ta statsmodels
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def download_data(ticker='AAPL', period='2y'):
    """
    Download historical stock data.
    """
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period=period, interval='1d')
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    """
    Calculate technical indicators: MA20, MA50, RSI, Bollinger Bands.
    """
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    return df

def plot_candlestick(df, ticker):
    """
    Plot candlestick chart with Moving Averages & Bollinger Bands.
    """
    print("📊 Plotting candlestick chart...")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='blue'), name='MA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='orange'), name='MA50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='green', width=1), name='BB High'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='red', width=1), name='BB Low'))
    fig.update_layout(title=f"{ticker} - Candlestick Chart with Indicators", xaxis_rangeslider_visible=False, height=600)
    fig.show()

def plot_rsi(df, ticker):
    """
    Plot RSI indicator.
    """
    print("📊 Plotting RSI...")
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df['RSI'], color='purple')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title(f'{ticker} - RSI (14-day)')
    plt.show()

def backtest_strategy(df, ticker):
    """
    Backtest simple MA20/MA50 crossover strategy.
    """
    print("⚙️ Running backtest...")
    df['Signal'] = np.where(df['MA20'] > df['MA50'], 1, -1)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    
    cumulative_market = (1 + df['Daily_Return']).cumprod()
    cumulative_strategy = (1 + df['Strategy_Return']).cumprod()
    
    plt.figure(figsize=(12,6))
    plt.plot(df.index, cumulative_market, label='Market Return')
    plt.plot(df.index, cumulative_strategy, label='Strategy Return')
    plt.title(f'{ticker} - Backtest: MA20 vs MA50 Crossover')
    plt.legend()
    plt.show()

def forecast_prices(df, ticker, steps=30):
    """
    Forecast future closing prices using ARIMA.
    """
    print("🔮 Forecasting future prices...")
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    forecast_dates = pd.date_range(df.index[-1], periods=steps+1, freq='B')[1:]

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Historical')
    plt.plot(forecast_dates, forecast, color='red', label='Forecast')
    plt.title(f'{ticker} - Forecast (Next {steps} Business Days)')
    plt.legend()
    plt.show()

def main():
    ticker = 'AAPL'
    df = download_data(ticker)
    df = add_indicators(df)
    
    plot_candlestick(df, ticker)
    plot_rsi(df, ticker)
    backtest_strategy(df, ticker)
    forecast_prices(df, ticker)

    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
