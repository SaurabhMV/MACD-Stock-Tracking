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
    period = st.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
    interval = st.selectbox("Chart Interval:", ["15m", "30m", "60m", "1h", "1d", "1wk"], index=4)
    
    st.subheader("Chart Display")
    show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    show_rsi = st.checkbox("Show RSI Chart", value=True)
    show_adx = st.checkbox("Show Trend Strength (ADX)", value=True)

# --- DATA LOADING ---
@st.cache_data
def load_data(symbol, p, i):
    try:
        data = yf.download(symbol, period=p, interval=i)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return pd.DataFrame()

def calculate_indicators(df):
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
    # ADX
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = df['Low'].diff().clip(upper=0).abs()
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr + 0.001))
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr + 0.001))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 0.001)
    df['ADX'] = dx.rolling(window=14).mean()
    return df

if ticker:
    df = load_data(ticker, period, interval)
    
    if not df.empty and len(df) > 26:
        df = calculate_indicators(df)

        # --- CROSS-TIMEFRAME CHECKER ---
        # Fetch Daily data to determine the "Anchor" trend
        df_daily = load_data(ticker, "1y", "1d")
        if not df_daily.empty:
            df_daily = calculate_indicators(df_daily)
            daily_macd = df_daily['MACD'].iloc[-1]
            daily_sig = df_daily['Signal_Line'].iloc[-1]
            anchor_trend = "BULLISH" if daily_macd > daily_sig else "BEARISH"
        else:
            anchor_trend = "UNKNOWN"

        # --- 2. MULTI-TIER SIGNAL LOGIC ---
        macd_cross_up = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
        macd_cross_down = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))
        rsi_buy_filter, rsi_sell_filter, strong_trend = df['RSI'] < 50, df['RSI'] > 50, df['ADX'] > 25

        df['MACD_Only_Buy'] = np.where(macd_cross_up & ~rsi_buy_filter, df['Close'], np.nan)
        df['MACD_Only_Sell'] = np.where(macd_cross_down & ~rsi_sell_filter, df['Close'], np.nan)
        df['Standard_Buy'] = np.where(macd_cross_up & rsi_buy_filter & ~strong_trend, df['Close'], np.nan)
        df['Standard_Sell'] = np.where(macd_cross_down & rsi_sell_filter & ~strong_trend, df['Close'], np.nan)
        df['Strong_Buy'] = np.where(macd_cross_up & rsi_buy_filter & strong_trend, df['Close'], np.nan)
        df['Strong_Sell'] = np.where(macd_cross_down & rsi_sell_filter & strong_trend, df['Close'], np.nan)

        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'], df['BB_Lower'] = df['BB_Mid'] + (df['BB_Std'] * 2), df['BB_Mid'] - (df['BB_Std'] * 2)

        # --- 3. ALERTS & METRICS ---
        m1, m2 = st.columns([2, 1])
        with m1:
            if not np.isnan(df['Strong_Buy'].iloc[-1]):
                st.success(f"🚀 **STRONG BUY ALERT:** High-Confidence signal on {interval} chart!")
                if anchor_trend == "BEARISH": st.warning("⚠️ **TREND CONFLICT:** You are buying into a Bearish Daily trend.")
            elif not np.isnan(df['Strong_Sell'].iloc[-1]):
                st.error(f"⚠️ **STRONG SELL ALERT:** High-Confidence signal on {interval} chart!")
                if anchor_trend == "BULLISH": st.warning("⚠️ **TREND CONFLICT:** You are selling into a Bullish Daily trend.")

        with m2:
            st.info(f"**Daily Anchor Trend:** {anchor_trend}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"${df['Close'].iloc[-1]:,.2f}")
        col2.metric("Wilder's RSI", f"{df['RSI'].iloc[-1]:.1f}")
        col3.metric("MACD Hist", f"{df['Hist'].iloc[-1]:.3f}")
        col4.metric("ADX", f"{df['ADX'].iloc[-1]:.1f}", "Strong" if df['ADX'].iloc[-1] > 25 else "Weak")

        # --- 4. PLOTTING ---
        rows = 2 + show_rsi + show_adx
        row_heights = [0.4] + [0.2] * (rows - 1)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)

        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white', width=1)), row=1, col=1)
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(173, 216, 230, 0.2)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='rgba(173, 216, 230, 0.2)', width=1), fill='tonexty'), row=1, col=1)

        if show_signals:
            fig.add_trace(go.Scatter(x=df.index, y=df['Strong_Buy'], name='Strong Buy', mode='markers', marker=dict(symbol='diamond', size=13, color='#00ff00', line=dict(width=2, color='white'))), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Strong_Sell'], name='Strong Sell', mode='markers', marker=dict(symbol='diamond', size=13, color='#ff0000', line=dict(width=2, color='white'))), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Standard_Buy'], name='Buy', mode='markers', marker=dict(symbol='triangle-up', size=10, color='#26a69a')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Standard_Sell'], name='Sell', mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ef5350')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Only_Buy'], name='MACD Cross (Pure)', mode='markers', marker=dict(symbol='circle', size=7, color='#00ff00', opacity=0.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Only_Sell'], name='MACD Cross (Pure)', mode='markers', marker=dict(symbol='circle', size=7, color='#ff0000', opacity=0.5)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#00d4ff')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='#ff9900')), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Hist', marker_color=['#ef5350' if v < 0 else '#26a69a' for v in df['Hist']]), row=2, col=1)

        current_row = 3
        if show_rsi:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#b39ddb')), row=current_row, col=1)
            current_row += 1
        if show_adx:
            fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX', line=dict(color='yellow')), row=current_row, col=1)

        fig.update_layout(height=400 + (rows * 150), template="plotly_dark", xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Signal Tier Definitions")
        d1, d2, d3 = st.columns(3)
        with d1: st.info("**1. Pure MACD (Circles)**\n\nRaw momentum crossover.")
        with d2: st.success("**2. Standard (Triangles)**\n\nMACD + RSI Filter.")
        with d3: st.warning("**3. Strong (Diamonds)**\n\nMACD + RSI + ADX (>25).")

    else:
        st.error(f"Insufficient data for {ticker} at the {interval} interval.")
