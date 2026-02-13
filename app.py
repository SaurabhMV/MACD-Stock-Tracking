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
    
    # Updated: Added "3mo" back to the list
    period = st.selectbox("Time Period:", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=1)
    
    st.subheader("Chart Display")
    show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    show_rsi = st.checkbox("Show RSI Chart", value=True)

# --- DATA LOADING ---
@st.cache_data
def load_data(symbol, p):
    try:
        data = yf.download(symbol, period=p)
        # Handle multi-index columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        return pd.DataFrame()

if ticker:
    df = load_data(ticker, period)
    
    if not df.empty and len(df) > 26:
        # --- 1. CALCULATE INDICATORS ---
        # MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal_Line']

        # RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20-day, 2 std dev)
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

        # --- 2. SIGNAL GENERATION ---
        # Buy: MACD crosses ABOVE Signal AND RSI < 50
        df['Buy_Signal'] = np.where(
            (df['MACD'] > df['Signal_Line']) & 
            (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)) & 
            (df['RSI'] < 50), 
            df['Close'], np.nan
        )
        
        # Sell: MACD crosses BELOW Signal AND RSI > 50
        df['Sell_Signal'] = np.where(
            (df['MACD'] < df['Signal_Line']) & 
            (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)) & 
            (df['RSI'] > 50), 
            df['Close'], np.nan
        )

        # --- 3. DASHBOARD METRICS ---
        current_price = df['Close'].iloc[-1]
        price_change = current_price - df['Close'].iloc[-2]
        pct_change = (price_change / df['Close'].iloc[-2]) * 100
        current_rsi = df['RSI'].iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
        col2.metric("Current RSI", f"{current_rsi:.1f}", "Overbought > 70 | Oversold < 30", delta_color="off")
        col3.metric("MACD Level", f"{df['MACD'].iloc[-1]:.3f}", f"{df['Hist'].iloc[-1]:.3f} Hist")

        # --- 4. PLOTTING ---
        rows = 2 + (1 if show_rsi else 0)
        row_heights = [0.5, 0.25, 0.25] if show_rsi else [0.7, 0.3]
        
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=row_heights,
                            subplot_titles=(f"{ticker} Price Action", "MACD Momentum", "RSI Strength"))

        # Row 1: Price Chart
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='white', width=1)), row=1, col=1)
        
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(173, 216, 230, 0.3)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='rgba(173, 216, 230, 0.3)', width=1), fill='tonexty'), row=1, col=1)

        if show_signals:
            fig.add_trace(go.Scatter(x=df.index, y=df['Buy_Signal'], name='BUY Signal', mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00', line=dict(width=1, color='white'))), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Sell_Signal'], name='SELL Signal', mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000', line=dict(width=1, color='white'))), row=1, col=1)

        # Row 2: MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD Line', line=dict(color='#00d4ff', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line', line=dict(color='#ff9900', width=1.5)), row=2, col=1)
        
        colors = ['#ef5350' if val < 0 else '#26a69a' for val in df['Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='MACD Hist', marker_color=colors), row=2, col=1)

        # Row 3: RSI (Optional)
        if show_rsi:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#b39ddb', width=2)), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Layout settings
        fig.update_layout(height=800, template="plotly_dark", 
                          xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        st.plotly_chart(fig, use_container_width=True)

        # --- 5. SIGNAL HISTORY ---
        st.subheader("Recent Signals")
        signals_df = df[(df['Buy_Signal'].notna()) | (df['Sell_Signal'].notna())].copy()
        
        if not signals_df.empty:
            # Clean up dataframe for display
            display_cols = ['Close', 'MACD', 'Signal_Line', 'RSI']
            st.dataframe(signals_df[display_cols].tail(10).style.format("{:.2f}"))
            
            # Export CSV
            csv = df.to_csv().encode('utf-8')
            st.download_button(label="Download Data as CSV", data=csv, file_name=f"{ticker}_analysis.csv", mime='text/csv')
        else:
            st.info("No Buy/Sell signals generated in the selected period.")

    else:
        st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
