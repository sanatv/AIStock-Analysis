import os
import streamlit as st
import yfinance as yf
import pandas as pd
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from openai import OpenAI
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Financial Analysis", layout="wide")

# 1. Secrets
#OPENAI_ORG = st.secrets.get("openai_org") or os.getenv("OPENAI_ORG")
OPENAI_KEY = st.secrets.get("openai_key") or os.getenv("OPENAI_KEY")
if not OPENAI_KEY:
    st.error("🔒 Please configure your OpenAI API key in Streamlit secrets or env var.")
    st.stop()
client = OpenAI(organization=OPENAI_ORG, api_key=OPENAI_KEY)

# 2. SEC downloader
dl = Downloader(download_path="sec-edgar-filings")

# 3. Caching data fetches
@st.cache_data
def fetch_income(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).financials

@st.cache_data
def fetch_filings(ticker: str) -> tuple[str, str]:
    filings = {}
    for ftype in ("10-K", "10-Q"):
        dl.get(ftype, ticker, limit=1)
        path = f"sec-edgar-filings/{ticker}/{ftype}/"
        try:
            latest = sorted(os.listdir(path), reverse=True)[0]
            with open(f"{path}{latest}/full-submission.txt", encoding="utf-8") as f:
                text = BeautifulSoup(f.read(), "lxml").get_text()
            filings[ftype] = text[:15000]
        except Exception:
            filings[ftype] = f"No {ftype} found."
    return filings["10-K"], filings["10-Q"]

# 4. Plot utility
METRIC_MAP = {
    "Total Revenue": "TotalRevenue",
    "Cost of Revenue": "CostOfRevenue",
    "Operating Income": "OperatingIncome",
    "Basic EPS": "BasicEps",
}

def plot_trends(df: pd.DataFrame, ticker: str):
    avail = {k: v for k, v in METRIC_MAP.items() if v in df.index}
    if not avail:
        st.warning("No matching metrics to plot.")
        return

    fig, axes = plt.subplots(len(avail), 1, figsize=(10, 4*len(avail)))
    if len(avail) == 1:
        axes = [axes]

    for ax, (label, idx) in zip(axes, avail.items()):
        series = df.loc[idx].iloc[::-1]
        ax.plot(series.index.strftime("%Y-%m-%d"), series.values, marker="o")
        ax.set_title(f"{label} Trend for {ticker}")
        ax.set_ylabel(label)
        ax.grid(True)

    st.pyplot(fig)

# 5. UI
st.title("🔍 Financial Analysis Dashboard")
ticker = st.text_input("📈 Stock Ticker:", "MSFT").upper().strip()

if st.button("Generate Analysis"):
    with st.spinner("Loading income statement..."):
        inc_df = fetch_income(ticker)
    if inc_df.empty:
        st.error("No income data available.")
    else:
        st.subheader("Income Statement")
        st.dataframe(inc_df)

        with st.spinner("Downloading filings…"):
            ten_k, ten_q = fetch_filings(ticker)

        st.subheader("📈 Trends")
        plot_trends(inc_df, ticker)

        st.subheader("🗃️ 10-K (preview)")
        st.text_area("10-K", ten_k, height=200)
        st.subheader("🗃️ 10-Q (preview)")
        st.text_area("10-Q", ten_q, height=200)

        # GPT commentary…
        # (same pattern as your prompt, but ensure you respect token limits)
