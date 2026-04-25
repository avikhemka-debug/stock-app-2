import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
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
@st.cache_data(ttl=120)
def fetch(ticker="NVDA", period="1y"):
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    return df.dropna()


def get_company_info(ticker):
    try:
        df = fetch(ticker, "5d")
        return {
            "Ticker": ticker,
            "Last Price": round(df["Close"].iloc[-1], 2),
            "Day High": round(df["High"].iloc[-1], 2),
            "Day Low": round(df["Low"].iloc[-1], 2),
            "Volume": int(df["Volume"].iloc[-1])
        }
    except:
        return {"Info": "Limited data available"}


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

    split = int(len(df) * 0.85)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=300)
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
# ENGINE
# ─────────────────────────────
@st.cache_data(ttl=60)
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
# UI SETUP
# ─────────────────────────────
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("🚀 AI Trading Dashboard")

refresh_rate = st.slider("Refresh every (seconds)", 5, 30, 5)
st_autorefresh(interval=refresh_rate * 1000, key="live")

ticker = st.text_input("Enter Stock Ticker", "NVDA")

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Charts",
    "AI Signals",
    "History"
])


# ─────────────────────────────
# SNAPSHOT TAB (STATIC)
# ─────────────────────────────
with tab1:
    st.subheader("📊 Company Snapshot")
    info = get_company_info(ticker)

    st.markdown(f"""
    **Ticker:** {info.get("Ticker", "-")}  
    **Last Price:** {info.get("Last Price", "-")}  
    **Day High:** {info.get("Day High", "-")}  
    **Day Low:** {info.get("Day Low", "-")}  
    **Volume:** {info.get("Volume", "-")}  
    """)


# ─────────────────────────────
# SAFE RUN (NO UI BREAK)
# ─────────────────────────────
try:
    result = run(ticker)
except Exception:
    st.warning("AI model temporarily failed.")
    st.stop()

df = result["df"]
df, anomaly_count = anomalies(df)

color = "green" if result["prediction"] == "BUY" else "red"


# ─────────────────────────────
# AI SIGNALS (FIXED)
# ─────────────────────────────
with tab3:
    st.subheader("🧠 AI Signal Engine")

    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", result["prediction"])
    col2.metric("Confidence", f"{result['confidence']*100:.2f}%")
    col3.metric("Price", f"${result['close']:.2f}")

    st.markdown(
        f"<h2 style='color:{color}'>{result['prediction']}</h2>",
        unsafe_allow_html=True
    )

    st.caption(
        f"Train Accuracy: {result['train_acc']*100:.1f}% | "
        f"Test Accuracy: {result['test_acc']*100:.1f}%"
    )

    st.write("Anomalies detected:", anomaly_count)


# ─────────────────────────────
# CHARTS
# ─────────────────────────────
with tab2:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=[df.index[-1]],
        y=[df["Close"].iloc[-1]],
        mode="markers",
        marker=dict(size=12),
        name=result["prediction"]
    ))

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.bar_chart(df["Volume"])


# ─────────────────────────────
# HISTORY
# ─────────────────────────────
with tab4:
    st.subheader("📜 Trade Memory")
    st.dataframe(memory())
