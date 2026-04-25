import streamlit as st
import yfinance as yf

st.title("NVDA Trading App")

ticker = st.text_input("Enter Stock Ticker", "NVDA")

try:
    data = yf.download(ticker, period="1mo")

    if data is None or data.empty:
        st.error("⚠️ Data not loading. Try again or change ticker.")
    else:
        st.line_chart(data["Close"])
        st.write(data.tail())

except Exception as e:
    st.error("Something went wrong while fetching data.")
