import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- APP CONFIG ---
st.set_page_config(page_title="Technical Stock Analyzer", layout="wide")
st.title("📈 Technical Stock Analysis Dashboard")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker:", "AAPL").upper()
    
    # Updated: Added "1mo" back to the list as requested
    period = st.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
    
    st.subheader("Chart Display")
    show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    show_rsi = st.checkbox("Show RSI Chart", value=True)
    show_adx = st.checkbox("Show Trend Strength (ADX)", value=True)

# --- DATA LOADING ---
@st.cache_data
def load_data(symbol, p):
    try:
        data = yf.download(symbol, period=p)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        return pd.DataFrame()

if ticker:
    df = load_data(ticker, period)
    
    # Note: Indicators like RSI and ADX need at least 14-26 periods of data
    if not df.empty and len(df) > 26:
        # --- 1. CALCULATE INDICATORS ---
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal_Line']

        # Wilder's RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

        # ADX
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

        # --- 2. MULTI-TIER SIGNAL LOGIC ---
        macd_cross_up = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
        macd_cross_down = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))
        
        rsi_buy_filter = df['RSI'] < 50
        rsi_sell_filter = df['RSI'] > 50
        strong_trend = df['ADX'] > 25

        # Tier 1: Pure MACD
        df['MACD_Only_Buy'] = np.where(macd_cross_up & ~rsi_buy_filter, df['Close'], np.nan)
        df['MACD_Only_Sell'] = np.where(macd_cross_down & ~rsi_sell_filter, df['Close'], np.nan)

        # Tier 2: Standard (MACD + RSI)
        df['Standard_Buy'] = np.where(macd_cross_up & rsi_buy_filter & ~strong_trend, df['Close'], np.nan)
        df['Standard_Sell'] = np.where(macd_cross_down & rsi_sell_filter & ~strong_trend, df['Close'], np.nan)

        # Tier 3: Strong (MACD + RSI + ADX)
        df['Strong_Buy'] = np.where(macd_cross_up & rsi_buy_filter & strong_trend, df['Close'], np.nan)
        df['Strong_Sell'] = np.where(macd_cross_down & rsi_sell_filter & strong_trend, df['Close'], np.nan)

        # --- 3. ALERTS & METRICS ---
        if not np.isnan(df['Strong_Buy'].iloc[-1]):
            st.success(f"🚀 **STRONG BUY ALERT:** {ticker} triggered a High-Confidence Trend signal!")
        elif not np.isnan(df['Strong_Sell'].iloc[-1]):
            st.error(f"⚠️ **STRONG SELL ALERT:** {ticker} triggered a High-Confidence Downward signal!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
        col2.metric("Wilder's RSI", f"{df['RSI'].iloc[-1]:.1f}")
        col3.metric("MACD Level", f"{df['MACD'].iloc[-1]:.3f}")
        col4.metric("ADX Trend", f"{df['ADX'].iloc[-1]:.1f}", "Strong" if df['ADX'].iloc[-1] > 25 else "Weak")

        # --- 4. PLOTTING ---
        rows = 2 + show_rsi + show_adx
        row_heights = [0.4] + [0.2] * (rows - 1)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)

        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='white', width=1)), row=1, col=1)
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(173, 216, 230, 0.2)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='rgba(173, 216, 230, 0.2)', width=1), fill='tonexty'), row=1, col=1)

        if show_signals:
            # Strong Tiers
            fig.add_trace(go.Scatter(x=df.index, y=df['Strong_Buy'], name='STRONG BUY (MACD+RSI+ADX)', mode='markers', marker=dict(symbol='diamond', size=13, color='#00ff00', line=dict(width=2, color='white'))), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Strong_Sell'], name='STRONG SELL (MACD+RSI+ADX)', mode='markers', marker=dict(symbol='diamond', size=13, color='#ff0000', line=dict(width=2, color='white'))), row=1, col=1)
            # Standard Tiers
            fig.add_trace(go.Scatter(x=df.index, y=df['Standard_Buy'], name='BUY (MACD+RSI)', mode='markers', marker=dict(symbol='triangle-up', size=10, color='#26a69a')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Standard_Sell'], name='SELL (MACD+RSI)', mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ef5350')), row=1, col=1)
            # Pure MACD Tiers
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Only_Buy'], name='MACD CROSS (Pure)', mode='markers', marker=dict(symbol='circle', size=7, color='#00ff00', opacity=0.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Only_Sell'], name='MACD CROSS (Pure)', mode='markers', marker=dict(symbol='circle', size=7, color='#ff0000', opacity=0.5)), row=1, col=1)

        # Subplots
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#00d4ff')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='#ff9900')), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Hist', marker_color=['#ef5350' if v < 0 else '#26a69a' for v in df['Hist']]), row=2, col=1)

        current_row = 3
        if show_rsi:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#b39ddb')), row=current_row, col=1)
            current_row += 1
        if show_adx:
            fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX Strength', line=dict(color='yellow')), row=current_row, col=1)

        fig.update_layout(height=400 + (rows * 150), template="plotly_dark", xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        # --- 5. TIER DEFINITIONS ---
        st.markdown("---")
        st.subheader("📊 Signal Tier Definitions")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.info("**1. Pure MACD (Circles)**\n\nTriggers on any MACD crossover. This is the most sensitive signal and carries the highest risk of false entries.")
        with d2:
            st.success("**2. Standard (Triangles)**\n\nMACD crossover + RSI Filter. Ensures you aren't buying into an already overextended market.")
        with d3:
            st.warning("**3. Strong (Diamonds)**\n\nMACD + RSI + ADX (>25). The gold standard: implies a strong trend is present to back the move.")

    else:
        st.error(f"Not enough data for {ticker} in the selected period. Indicators require at least 26 data points.")
