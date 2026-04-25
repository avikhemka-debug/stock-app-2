import streamlit as st
import yfinance as yf

st.title("NVDA Trading App")

ticker = st.text_input("Enter Stock Ticker", "NVDA")

data = yf.download(ticker, period="1mo")

if not data.empty:
    st.line_chart(data["Close"])
else:
    st.write("No data found")
