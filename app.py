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
# DATA
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


def build_features(df):
    df["ret"] = df["Close"].pct_change()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["volatility"] = df["ret"].rolling(20).std()
    df["rsi"] = rsi(df["Close"])
    return df.dropna()


# ─────────────────────────────
# MODEL
# ─────────────────────────────
def train_model(df):
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    X = df[["ma20", "ma50", "volatility", "rsi"]]
    y = df["target"]

    split = int(len(df) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    pred = model.predict(scaler.transform([X.iloc[-1]]))[0]
    conf = model.predict_proba(scaler.transform([X.iloc[-1]])).max()

    return {
        "signal": "BUY" if pred else "SELL",
        "confidence": conf,
        "train_acc": model.score(X_train, y_train),
        "test_acc": model.score(X_test, y_test),
        "df": df
    }


# ─────────────────────────────
# FULL COMPANY DATA
# ─────────────────────────────
@st.cache_data(ttl=3600)
def get_full_company_data(ticker):
    t = yf.Ticker(ticker)
    return {
        "info": t.info,
        "financials": t.financials,
        "balance": t.balance_sheet,
        "cashflow": t.cashflow,
        "earnings": t.earnings,
        "recommendations": t.recommendations
    }


# ─────────────────────────────
# RUN ENGINE
# ─────────────────────────────
@st.cache_data(ttl=60)
def run(ticker):
    init_db()

    df = fetch(ticker)
    df = build_features(df)

    result = train_model(df)

    close = float(df["Close"].iloc[-1])

    save_log(
        ticker,
        close,
        result["signal"],
        result["confidence"],
        result["signal"] + " EXECUTED",
        result["train_acc"],
        result["test_acc"]
    )

    result["close"] = close
    return result


# ─────────────────────────────
# UI
# ─────────────────────────────
st.set_page_config(layout="wide")
st.title("🚀 AI Trading Dashboard")

refresh = st.slider("Refresh", 5, 30, 5)
st_autorefresh(interval=refresh * 1000, key="live")

ticker = st.text_input("Ticker", "NVDA")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Charts", "AI Signals", "Trade History", "Company"
])

result = run(ticker)
df = result["df"]

# OVERVIEW
with tab1:
    st.metric("Signal", result["signal"])
    st.metric("Confidence", f"{result['confidence']*100:.2f}%")
    st.metric("Price", f"${result['close']:.2f}")

# CHARTS
with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma20"], name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma50"], name="MA50"))
    st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(px.pie(
        names=["Confidence", "Uncertainty"],
        values=[result["confidence"], 1 - result["confidence"]]
    ))

# AI SIGNALS
with tab3:
    st.write(result)

# TRADE HISTORY
with tab4:
    logs = load_logs()
    st.dataframe(logs)

    if len(logs) > 0:
        st.plotly_chart(px.pie(logs, names="prediction"))

# COMPANY (FULL DATA)
with tab5:
    data = get_full_company_data(ticker)
    info = data["info"]

    st.markdown(f"## {info.get('longName')}")

    st.write(info.get("longBusinessSummary"))

    st.subheader("📊 Key Stats")
    st.write({
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Market Cap": info.get("marketCap"),
        "PE": info.get("trailingPE"),
        "Beta": info.get("beta")
    })

    st.subheader("💰 Financials")
    st.dataframe(data["financials"].T)

    st.subheader("📊 Balance Sheet")
    st.dataframe(data["balance"].T)

    st.subheader("💸 Cash Flow")
    st.dataframe(data["cashflow"].T)

    st.subheader("📈 Earnings")
    st.dataframe(data["earnings"])

    st.subheader("📊 Analyst Recommendations")
    st.dataframe(data["recommendations"].tail(10))
