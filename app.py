import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        INSERT INTO logs (
            timestamp, ticker, close, prediction,
            confidence, action, train_acc, test_acc
        ) VALUES (?,?,?,?,?,?,?,?)
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
def fetch(ticker="NVDA", period="1y"):
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    return df.dropna()


def get_company_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "Name": info.get("longName"),
        "Sector": info.get("sector"),
        "Market Cap": info.get("marketCap"),
        "PE Ratio": info.get("trailingPE"),
        "52W High": info.get("fiftyTwoWeekHigh"),
        "52W Low": info.get("fiftyTwoWeekLow")
    }


# ─────────────────────────────
# FEATURES
# ─────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_features(df):
    df = df.copy()

    df["ret"] = df["Close"].pct_change()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["volatility"] = df["ret"].rolling(20).std()
    df["rsi"] = rsi(df["Close"], 14)

    return df.dropna()


# ─────────────────────────────
# MODEL
# ─────────────────────────────
def train_model(df):
    df = df.copy()

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    features = ["ma20", "ma50", "volatility", "rsi"]

    X = df[features]
    y = df["target"]

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    latest = scaler.transform([X.iloc[-1]])
    pred = model.predict(latest)[0]
    conf = model.predict_proba(latest).max()

    signal = "BUY" if pred == 1 else "SELL"

    return signal, conf, train_acc, test_acc, df


# ─────────────────────────────
# ANOMALY
# ─────────────────────────────
def anomalies(df):
    vol = df["ret"].std()
    threshold = max(0.02, vol * 2.5)
    df["anomaly"] = df["ret"].abs() > threshold
    return df, int(df["anomaly"].sum())


# ─────────────────────────────
# MAIN ENGINE
# ─────────────────────────────
def run(ticker="NVDA"):
    init_db()

    df = fetch(ticker)
    df = build_features(df)

    signal, conf, train_acc, test_acc, df = train_model(df)
    df, anomaly_count = anomalies(df)

    close = float(df["Close"].iloc[-1])
    action = f"{signal} EXECUTED"

    save_log(ticker, close, signal, conf, action, train_acc, test_acc)

    return {
        "prediction": signal,
        "confidence": conf,
        "close": close,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "anomalies": anomaly_count,
        "df": df
    }


def memory():
    return load_logs()


# ─────────────────────────────
# UI
# ─────────────────────────────
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

st.title("🚀 AI Trading Dashboard")

# 🎨 Styling
st.markdown("""
<style>
.stMetric {
    background-color: #111;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

ticker = st.text_input("Enter Stock Ticker", "NVDA")

# 📊 Company Info
info = get_company_info(ticker)
st.subheader("📊 Company Info")
st.write(info)

if st.button("Run AI Model"):
    with st.spinner("Running AI model..."):
        result = run(ticker)

        st.success("Model Executed")

        color = "green" if result["prediction"] == "BUY" else "red"

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", result["prediction"])
        col2.metric("Confidence", f"{result['confidence']*100:.2f}%")
        col3.metric("Price", f"${result['close']:.2f}")

        st.markdown(f"<h2 style='color:{color}'>{result['prediction']}</h2>", unsafe_allow_html=True)

        col4, col5 = st.columns(2)
        col4.metric("Train Accuracy", f"{result['train_acc']*100:.2f}%")
        col5.metric("Test Accuracy", f"{result['test_acc']*100:.2f}%")

        df = result["df"]

        # 📈 Price Chart
        st.subheader("📈 Price Chart")
        st.line_chart(df["Close"])

        # 📊 Volume
        st.subheader("📊 Volume")
        st.bar_chart(df["Volume"])

        # 🚨 Anomalies
        df, anomaly_count = anomalies(df)
        anomaly_df = df[df["anomaly"] == True]

        st.subheader("🚨 Anomalies Detected")
        st.write(anomaly_df[["Close", "ret"]])

        st.write("Total anomalies:", anomaly_count)


if st.checkbox("Show Trade History"):
    logs = memory()
    st.dataframe(logs)
