import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Strategy Backtester", layout="wide")
st.title("📊 Stock Strategy Backtester")

with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Stock Ticker:", "AAPL").upper()
    period = st.selectbox("Time Period:", ["1y", "2y", "5y"], index=0)
    initial_capital = st.number_input("Initial Investment ($):", value=10000)

@st.cache_data
def load_data(symbol, p):
    data = yf.download(symbol, period=p)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

if ticker:
    df = load_data(ticker, period)
    
    if not df.empty:
        # --- 1. CALCULATE INDICATORS ---
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # --- 2. GENERATE SIGNALS & POSITIONS ---
        # 1 = Buy/Hold, 0 = Sell/Cash
        df['Signal'] = 0.0
        # Buy condition
        buy_cond = (df['MACD'] > df['Signal_Line']) & (df['RSI'] < 50)
        # Sell condition
        sell_cond = (df['MACD'] < df['Signal_Line']) & (df['RSI'] > 50)
        
        # Simple Logic: Enter on Buy, Exit on Sell
        position = 0
        signals = []
        for i in range(len(df)):
            if buy_cond.iloc[i]:
                position = 1
            elif sell_cond.iloc[i]:
                position = 0
            signals.append(position)
        
        df['Position'] = signals
        
        # --- 3. BACKTEST CALCULATIONS ---
        df['Market_Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Market_Returns'] * df['Position'].shift(1)
        
        df['Cumulative_Market'] = (1 + df['Market_Returns']).cumprod() * initial_capital
        df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod() * initial_capital

        # Metrics
        total_return = ((df['Cumulative_Strategy'].iloc[-1] - initial_capital) / initial_capital) * 100
        market_return = ((df['Cumulative_Market'].iloc[-1] - initial_capital) / initial_capital) * 100
        final_val = df['Cumulative_Strategy'].iloc[-1]

        # --- 4. DISPLAY RESULTS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Portfolio Value", f"${final_val:,.2f}")
        col2.metric("Strategy Total Return", f"{total_return:.2f}%", delta=f"{total_return - market_return:.2f}% vs Market")
        col3.metric("Buy & Hold Return", f"{market_return:.2f}%")

        # Plotting
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, row_heights=[0.6, 0.4],
                            subplot_titles=("Portfolio Value ($) Over Time", "MACD & RSI Context"))

        # Portfolio Comparison
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy'], name='Strategy', line=dict(color='#00ff00', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market'], name='Buy & Hold', line=dict(color='gray', dash='dot')), row=1, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='cyan')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='orange')), row=2, col=1)

        fig.update_layout(height=700, template="plotly_dark", showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Ticker data unavailable.")
