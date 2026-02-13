import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Ultimate Pro Signal Tracker", layout="wide")
st.title("📈 Ultimate Strategy & Signal Dashboard")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Stock Ticker:", "AAPL").upper()
    period = st.selectbox("Time Period:", ["6mo", "1y", "2y", "5y"], index=1)
    
    st.subheader("Chart View")
    show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    show_rsi = st.checkbox("Show RSI Chart", value=True)
    
    st.markdown("---")
    initial_cap = st.number_input("Backtest Capital ($):", value=10000)

@st.cache_data
def load_data(symbol, p):
    data = yf.download(symbol, period=p)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

if ticker:
    df = load_data(ticker, period)
    
    if not df.empty:
        # --- 1. CALCULATIONS ---
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal_Line']

        # RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # Bollinger Bands (20-day SMA)
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

        # --- 2. SIGNAL GENERATION (MACD + RSI Confluence) ---
        df['Buy_Signal'] = np.where((df['MACD'] > df['Signal_Line']) & 
                                    (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)) & 
                                    (df['RSI'] < 45), df['Close'], np.nan)
        
        df['Sell_Signal'] = np.where((df['MACD'] < df['Signal_Line']) & 
                                     (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)) & 
                                     (df['RSI'] > 55), df['Close'], np.nan)

        # --- 3. BACKTESTING ---
        df['Position'] = np.nan
        df.loc[df['Buy_Signal'].notna(), 'Position'] = 1
        df.loc[df['Sell_Signal'].notna(), 'Position'] = 0
        df['Position'] = df['Position'].ffill().fillna(0)
        
        df['Market_Ret'] = df['Close'].pct_change()
        df['Strat_Ret'] = df['Market_Ret'] * df['Position'].shift(1)
        df['Cum_ROI'] = (1 + df['Strat_Ret'].fillna(0)).cumprod() * initial_cap

        # --- 4. METRICS ---
        total_ret = ((df['Cum_ROI'].iloc[-1] - initial_cap) / initial_cap) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
        m2.metric("Strategy ROI", f"{total_ret:.2f}%")
        m3.metric("Portfolio Value", f"${df['Cum_ROI'].iloc[-1]:,.2f}")

        # --- 5. PLOTTING ---
        # Determine number of rows dynamically
        rows = 2 + (1 if show_rsi else 0)
        row_heights = [0.5, 0.25, 0.25] if show_rsi else [0.7, 0.3]
        
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.07, row_heights=row_heights,
                            subplot_titles=(f"{ticker} Price & Volatility", "MACD Momentum", "RSI (Strength)"))

        # Row 1: Price, BB, and Signals
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white', width=1.5)), row=1, col=1)
        
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(173, 216, 230, 0.3)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='rgba(173, 216, 230, 0.3)', width=1), fill='tonexty'), row=1, col=1)

        if show_signals:
            fig.add_trace(go.Scatter(x=df.index, y=df['Buy_Signal'], name='Buy', mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Sell_Signal'], name='Sell', mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000')), row=1, col=1)

        # Row 2: MACD
        colors = ['red' if val < 0 else 'green' for val in df['Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Histogram', marker_color=colors, opacity=0.5), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='cyan')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='orange')), row=2, col=1)

        # Row 3: RSI (Optional)
        if show_rsi:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='magenta')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=400 + (200 * rows), template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. SUMMARY & EXPORT ---
        st.subheader("Strategy Summary")
        recent_signals = df[(df['Buy_Signal'].notna()) | (df['Sell_Signal'].notna())].tail(10)
        st.dataframe(recent_signals[['Close', 'MACD', 'RSI', 'Cum_ROI']])
        
        csv = df.to_csv().encode('utf-8')
        st.download_button(label="Download Full Backtest CSV", data=csv, file_name=f"{ticker}_analysis.csv", mime='text/csv')

    else:
        st.error("Ticker not found. Please try again.")
