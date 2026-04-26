import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

DB = "live_trading_memory.db"

# ─────────────────────────────
# DATABASE
# ─────────────────────────────
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ticker TEXT,
        close REAL,
        prediction TEXT,
        confidence REAL,
        action TEXT,
        train_acc REAL,
        test_acc REAL
    )
    """)
    conn.commit()
    conn.close()

def save_log(ticker, close, prediction, confidence, action, train_acc, test_acc):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO logs VALUES (NULL,?,?,?,?,?,?,?,?)
    """, (
        str(datetime.now()),
        ticker,
        float(close),
        prediction,
        float(confidence),
        action,
        float(train_acc),
        float(test_acc)
    ))
    conn.commit()
    conn.close()

def load_logs():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM logs ORDER BY id DESC", conn)
    conn.close()
    return df

# ─────────────────────────────
# DATA & FEATURES
# ─────────────────────────────
@st.cache_data(ttl=120)
def fetch(ticker="NVDA"):
    return yf.Ticker(ticker).history(period="1y", auto_adjust=True).dropna()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# UPDATED: Added momentum and copy safety
def build_features(df):
    df = df.copy()
    df["ret"] = df["Close"].pct_change()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["volatility"] = df["ret"].rolling(20).std()
    df["rsi"] = rsi(df["Close"])

    # NEW → gives direction awareness
    df["momentum"] = df["Close"] / df["Close"].shift(10) - 1

    return df.dropna()

# ─────────────────────────────
# MODEL
# ─────────────────────────────
# UPDATED: Balanced class weights and 0.55 threshold logic
def train_model(df):
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    # Updated feature set
    features = ["ma20", "ma50", "volatility", "rsi", "momentum"]
    X = df[features]
    y = df["target"]

    split = int(len(df) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # FIX 1 → class_weight="balanced" removes SELL bias
    model = LogisticRegression(max_iter=300, class_weight="balanced")
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    # Prepare latest data for prediction
    latest = scaler.transform([X.iloc[-1]])

    # FIX 2 → probability-based decision
    proba = model.predict_proba(latest)[0][1]

    # Signal logic with 55% threshold
    if proba > 0.55:
        signal = "BUY"
    else:
        signal = "SELL"

    confidence = max(proba, 1 - proba)

    return {
        "signal": signal,
        "confidence": confidence,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "df": df
    }

# ─────────────────────────────
# ENGINE & UI
# ─────────────────────────────
@st.cache_data(ttl=3600)
def get_full_company_data(ticker):
    t = yf.Ticker(ticker)
    return {
        "info": t.info,
        "financials": t.financials,
        "balance": t.balance_sheet,
        "cashflow": t.cashflow,
        "recommendations": t.recommendations
    }

@st.cache_data(ttl=60)
def run_engine(ticker):
    init_db()
    raw_df = fetch(ticker)
    processed_df = build_features(raw_df)
    result = train_model(processed_df)
    
    close_price = float(processed_df["Close"].iloc[-1])
    save_log(
        ticker, 
        close_price, 
        result["signal"], 
        result["confidence"], 
        result["signal"] + " EXECUTED",
        result["train_acc"],
        result["test_acc"]
    )
    result["close"] = close_price
    return result

# UI Setup
st.set_page_config(layout="wide", page_title="AI Trading Dashboard")
st.title("🚀 AI Trading Dashboard")

refresh = st.sidebar.slider("Refresh Interval (sec)", 5, 60, 10)
st_autorefresh(interval=refresh * 1000, key="live_refresh")

ticker_input = st.text_input("Enter Ticker (e.g., NVDA, AAPL, BTC-USD)", "NVDA")

try:
    res = run_engine(ticker_input)
    plot_df = res["df"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Charts", "AI Insights", "History", "Company"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Signal", res["signal"])
        c2.metric("Confidence", f"{res['confidence']*100:.2f}%")
        c3.metric("Price", f"${res['close']:.2f}")

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ma20"], name="MA20", line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ma50"], name="MA50", line=dict(dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.write("### Model Performance")
        st.write(f"**Training Accuracy:** {res['train_acc']:.2%}")
        st.write(f"**Testing Accuracy:** {res['test_acc']:.2%}")
        st.json({k: v for k, v in res.items() if k != 'df'})

    with tab4:
        st.dataframe(load_logs(), use_container_width=True)

    with tab5:
        co_data = get_full_company_data(ticker_input)
        st.subheader(co_data['info'].get('longName', ticker_input))
        st.write(co_data['info'].get('longBusinessSummary', 'No summary available.'))
        st.dataframe(co_data['financials'])

except Exception as e:
    st.error(f"Error loading ticker {ticker_input}: {e}")
