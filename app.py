import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Pro Signal Tracker", layout="wide")
st.title("📈 Stock Strategy & Signal Dashboard")

# Sidebar Controls
with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Stock Ticker:", "AAPL").upper()
    period = st.selectbox("Time Period:", ["6mo", "1y", "2y", "5y"], index=0)
    show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
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
        # --- CALCULATIONS ---
        # 1. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 2. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # 3. Signal Generation Logic
        df['Buy_Signal'] = np.where((df['MACD'] > df['Signal_Line']) & 
                                    (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)) & 
                                    (df['RSI'] < 45), df['Close'], np.nan)
        
        df['Sell_Signal'] = np.where((df['MACD'] < df['Signal_Line']) & 
                                     (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)) & 
                                     (df['RSI'] > 55), df['Close'], np.nan)

        # 4. Backtesting Logic
        df['Position'] = np.nan
        df.loc[df['Buy_Signal'].notna(), 'Position'] = 1
        df.loc[df['Sell_Signal'].notna(), 'Position'] = 0
        df['Position'] = df['Position'].ffill().fillna(0)
        
        df['Market_Ret'] = df['Close'].pct_change()
        df['Strat_Ret'] = df['Market_Ret'] * df['Position'].shift(1)
        df['Cum_ROI'] = (1 + df['Strat_Ret'].fillna(0)).cumprod() * initial_cap

        # --- METRICS ---
        total_ret = ((df['Cum_ROI'].iloc[-1] - initial_cap) / initial_cap) * 100
        curr_price = df['Close'].iloc[-1]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${curr_price:,.2f}")
        m2.metric("Strategy ROI", f"{total_ret:.2f}%")
        m3.metric("Final Value", f"${df['Cum_ROI'].iloc[-1]:,.2f}")

        # --- PLOTTING ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, row_heights=[0.7, 0.3],
                            subplot_titles=(f"{ticker} Price Action", "MACD Momentum"))

        # Row 1: Price & Signals
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='white', width=1.5)), row=1, col=1)

        if show_signals:
            fig.add_trace(go.Scatter(x=df.index, y=df['Buy_Signal'], name='Buy',
                                     mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Sell_Signal'], name='Sell',
                                     mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000')), row=1, col=1)

        # Row 2: MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='cyan')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='orange')), row=2, col=1)
        
        fig.update_layout(height=800, template="plotly_dark", showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Signal Summary & Export
        st.subheader("Recent Trade Signals")
        recent_signals = df[(df['Buy_Signal'].notna()) | (df['Sell_Signal'].notna())].tail(10)
        
        if not recent_signals.empty:
            st.dataframe(recent_signals[['Close', 'MACD', 'RSI', 'Cum_ROI']])
            
            # Export CSV
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Full Backtest Data (CSV)",
                data=csv,
                file_name=f"{ticker}_backtest.csv",
                mime='text/csv',
            )
        else:
            st.info("No clear signals detected in the recent period.")

    else:
        st.error("Ticker not found. Check your spelling (e.g., TSLA, BTC-USD).")
