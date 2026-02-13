import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- APP CONFIG ---
st.set_page_config(page_title="Technical Stock Analyzer Pro", layout="wide")
st.title("📈 Pro Stock Analysis Dashboard")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker:", "AAPL").upper()
    period = st.selectbox("Time Period:", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=1)
    
    st.subheader("Chart Display")
    show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    show_rsi = st.checkbox("Show RSI Chart", value=True)
    show_adx = st.checkbox("Show ADX (Trend Strength)", value=True)

# --- DATA LOADING ---
@st.cache_data
def load_data(symbol, p):
    try:
        data = yf.download(symbol, period=p)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return pd.DataFrame()

if ticker:
    df = load_data(ticker, period)
    
    if not df.empty and len(df) > 30:
        # --- 1. CALCULATE INDICATORS ---
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal_Line']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

        # ADX (Average Directional Index - Simplified)
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = df['Low'].diff().clip(upper=0).abs()
        tr = pd.concat([df['High'] - df['Low'], 
                       (df['High'] - df['Close'].shift(1)).abs(), 
                       (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=14).mean()

        # --- 2. SIGNAL GENERATION ---
        df['Buy_Signal'] = np.where((df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)) & (df['RSI'] < 50), df['Close'], np.nan)
        df['Sell_Signal'] = np.where((df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)) & (df['RSI'] > 50), df['Close'], np.nan)

        # --- 3. METRICS ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${df['Close'].iloc[-1]:,.2f}")
        c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
        trend_status = "Strong" if df['ADX'].iloc[-1] > 25 else "Weak/Sideways"
        c3.metric("Trend Strength", trend_status, f"ADX: {df['ADX'].iloc[-1]:.1f}")
        c4.metric("MACD Hist", f"{df['Hist'].iloc[-1]:.3f}")

        # --- 4. PLOTTING ---
        rows = 2 + show_rsi + show_adx
        heights = [0.4] + [0.2] * (rows - 1)
        
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=heights)

        # Main Chart
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')), row=1, col=1)
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(173,216,230,0.2)')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='rgba(173,216,230,0.2)'), fill='tonexty'), row=1, col=1)
        if show_signals:
            fig.add_trace(go.Scatter(x=df.index, y=df['Buy_Signal'], name='Buy', mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Sell_Signal'], name='Sell', mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000')), row=1, col=1)

        # MACD
        curr_row = 2
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#00d4ff')), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='#ff9900')), row=curr_row, col=1)
        curr_row += 1

        # RSI
        if show_rsi:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#b39ddb')), row=curr_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=curr_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=curr_row, col=1)
            curr_row += 1

        # ADX
        if show_adx:
            fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX (Strength)', line=dict(color='yellow')), row=curr_row, col=1)
            fig.add_hline(y=25, line_dash="dot", line_color="white", row=curr_row, col=1)

        fig.update_layout(height=300 + (200 * rows), template="plotly_dark", showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Enter a valid ticker or select a longer time frame.")
