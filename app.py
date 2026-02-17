import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import time

# --- APP CONFIG & STYLING ---
st.set_page_config(page_title="Pro Trading Terminal", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: Configuration & Live Updates ---
with st.sidebar:
    st.header("🛠️ Configuration")
    ticker = st.text_input("Stock Ticker:", "AAPL").upper()
    
    with st.expander("⏳ Timeframe Settings", expanded=True):
        period = st.selectbox("Historical Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
        interval = st.selectbox("Chart Interval:", ["15m", "30m", "60m", "1h", "1d", "1wk"], index=4)
    
    with st.expander("🎨 Visual Overlays", expanded=False):
        show_signals = st.checkbox("Buy/Sell Signals", value=True)
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_rsi = st.checkbox("RSI Sub-chart", value=True)
        show_adx = st.checkbox("ADX Sub-chart", value=True)

    # --- AUTO REFRESH CONTROLS ---
    with st.expander("🔄 Live Update Settings", expanded=True):
        auto_refresh = st.toggle("Enable Auto-Refresh", value=False)
        refresh_interval = st.select_slider(
            "Refresh Frequency:",
            options=[10, 30, 60, 300, 600],
            value=60,
            format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}m"
        )
    
    st.divider()
    # Timestamp in EST
    est_tz = timezone(timedelta(hours=-5))
    current_time = datetime.now(est_tz).strftime("%Y-%m-%d %H:%M:%S EST")
    st.caption(f"Last updated: {current_time}")

# --- DATA ENGINE ---
@st.cache_data(ttl=refresh_interval)
def load_data(symbol, p, i):
    try:
        data = yf.download(symbol, period=p, interval=i)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except: return pd.DataFrame()

def calculate_indicators(df):
    # MACD Calculation
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal_Line']
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))
    
    # ADX Calculation
    plus_dm, minus_dm = df['High'].diff().clip(lower=0), df['Low'].diff().clip(upper=0).abs()
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / (atr + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(window=14).mean()
    return df

st.title(f"🔭 {ticker}: Technical Convergence Terminal")

if ticker:
    df = load_data(ticker, period, interval)
    
    if not df.empty and len(df) > 26:
        df = calculate_indicators(df)
        
        # Cross-Timeframe Daily Anchor
        df_daily = load_data(ticker, "1y", "1d")
        if not df_daily.empty:
            df_daily = calculate_indicators(df_daily)
            anchor_trend = "BULLISH 🟢" if df_daily['MACD'].iloc[-1] > df_daily['Signal_Line'].iloc[-1] else "BEARISH 🔴"
        else: anchor_trend = "UNKNOWN"

        # UPDATED SIGNAL LOGIC (ADX REMOVED)
        up, down = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)), (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))
        r_buy, r_sell = df['RSI'] < 50, df['RSI'] > 50
        
        # Strong: MACD + RSI
        df['Strong_Buy'] = np.where(up & r_buy, df['Close'], np.nan)
        df['Strong_Sell'] = np.where(down & r_sell, df['Close'], np.nan)
        # Standard: MACD Only (filtered to avoid overlapping Strong icons)
        df['Standard_Buy'] = np.where(up & ~r_buy, df['Close'], np.nan)
        df['Standard_Sell'] = np.where(down & ~r_sell, df['Close'], np.nan)

        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'], df['BB_Lower'] = df['BB_Mid'] + (df['BB_Std'] * 2), df['BB_Mid'] - (df['BB_Std'] * 2)

        # --- METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"${df['Close'].iloc[-1]:,.2f}")
        m2.metric("Wilder's RSI", f"{df['RSI'].iloc[-1]:.1f}")
        m3.metric("ADX Strength", f"{df['ADX'].iloc[-1]:.1f}")
        m4.metric("Daily Anchor", anchor_trend)

        if not np.isnan(df['Strong_Buy'].iloc[-1]): st.success("🚀 **ENTRY ALERT:** Strong Buy detected!")
        elif not np.isnan(df['Strong_Sell'].iloc[-1]): st.error("⚠️ **EXIT ALERT:** Strong Sell detected!")

        # --- TABS ---
        tab1, tab2 = st.tabs(["📈 Analysis Chart", "📚 Strategy Guide"])

        with tab1:
            titles = ["Price Action & Signals", "Momentum (MACD)"]
            if show_rsi: titles.append("Relative Strength (RSI)")
            if show_adx: titles.append("Trend Strength (ADX)")

            rows = 2 + show_rsi + show_adx
            
            # --- UPDATED DYNAMIC ROW HEIGHTS (Larger MACD) ---
            if rows == 2: h = [0.6, 0.4]
            elif rows == 3: h = [0.5, 0.3, 0.2]
            else: h = [0.4, 0.3, 0.15, 0.15]

            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
                                row_heights=h, subplot_titles=titles)

            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#FFFFFF', width=1.5)), row=1, col=1)
            if show_bb:
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(173, 216, 230, 0.1)'), name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(173, 216, 230, 0.1)'), fill='tonexty', name='BB Lower'), row=1, col=1)

            if show_signals:
                fig.add_trace(go.Scatter(x=df.index, y=df['Strong_Buy'], name='💎 Strong Buy', mode='markers', marker=dict(symbol='diamond', size=12, color='#00FF00', line=dict(width=1, color='white'))), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Strong_Sell'], name='💎 Strong Sell', mode='markers', marker=dict(symbol='diamond', size=12, color='#FF4B4B', line=dict(width=1, color='white'))), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Standard_Buy'], name='🔺 Standard Buy', mode='markers', marker=dict(symbol='triangle-up', size=9, color='#00FF00')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Standard_Sell'], name='🔺 Standard Sell', mode='markers', marker=dict(symbol='triangle-down', size=9, color='#FF4B4B')), row=1, col=1)

            fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Momentum', marker_color=['#FF4B4B' if v < 0 else '#00FF00' for v in df['Hist']]), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#00D4FF', width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='#FF9900', width=1.5)), row=2, col=1)

            curr_r = 3
            if show_rsi:
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#B39DDB')), row=curr_r, col=1)
                curr_r += 1
            if show_adx:
                fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX', line=dict(color='#FDD835')), row=curr_r, col=1)

            fig.update_layout(height=900, template="plotly_dark", margin=dict(l=10, r=10, t=80, b=10), hovermode="x unified")
            fig.update_annotations(font=dict(family="Helvetica", size=14, color="#FFFFFF"), bgcolor="#1e2130", bordercolor="#31333f", borderwidth=1.5, borderpad=5)
            
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.header("Strategy Architecture")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("💎 Strong Tier")
                st.write("**Criteria:** MACD Crossover + RSI Filter.\n\nIndicates momentum confirmed by relative strength conditions.")
            with c2:
                st.subheader("🔺 Standard Tier")
                st.write("**Criteria:** MACD Crossover Only.\n\nPure momentum signals without the RSI overlay.")
            
            st.info("**Visual Note:** The MACD chart has been enlarged to 30% of the total view to improve line visibility.")

        # --- LIVE RERUN LOGIC ---
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

    else:
        st.error(f"Waiting for sufficient data for {ticker} (Indicators require >26 periods)...")
