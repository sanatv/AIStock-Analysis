import os
import streamlit as st
import yfinance as yf
import pandas as pd
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from openai import OpenAI
import matplotlib.pyplot as plt

# Streamlit secrets should be configured in secrets.toml
# [openai]
# openai_key = "YOUR_API_KEY"
# openai_org = "YOUR_ORG_ID"  # Optional

# 1. Fetch and validate OpenAI credentials
OPENAI_KEY = st.secrets.get("openai_key")
OPENAI_ORG = st.secrets.get("openai_org")
if not OPENAI_KEY:
    st.error("🔒 Please configure your OpenAI API key in Streamlit secrets as 'openai_key'.")
    st.stop()

@st.cache_data(show_spinner=False)
def init_openai_client() -> OpenAI:
    """Initialize the OpenAI client using validated secrets."""
    return OpenAI(
        api_key=OPENAI_KEY,
        organization=OPENAI_ORG
    )

@st.cache_data(show_spinner=False)
def get_income_statement(ticker: str) -> pd.DataFrame:
    """Fetch income statement from yfinance."""
    stock = yf.Ticker(ticker)
    return stock.financials

@st.cache_data(show_spinner=False)
def download_and_parse_filings(ticker: str) -> tuple[str, str]:
    """Download and parse latest 10-K and 10-Q filings."""
    dl = Downloader("Vats Inc", "sanatv@gmail.com")
    filing_texts = {}
    base_dir = os.path.join("sec-edgar-filings", ticker)
    for name, ftype in [("10-K", "10-K"), ("10-Q", "10-Q")]:
        try:
            dl.get(ftype, ticker, limit=1)
            filing_dir = os.path.join(base_dir, ftype)
            latest = sorted(os.listdir(filing_dir), reverse=True)[0]
            path = os.path.join(filing_dir, latest, "full-submission.txt")
            with open(path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'lxml')
                text = soup.get_text(separator='\n')
                filing_texts[name] = text[:15000]
        except Exception:
            filing_texts[name] = f"No {name} filing found or error parsing."
    return filing_texts.get("10-K", ""), filing_texts.get("10-Q", "")

@st.cache_data(show_spinner=False)
def plot_income_statement_trends(income: pd.DataFrame, ticker: str):
    """Plot key income statement metrics over time."""
    METRIC_MAP = {
        "Total Revenue": "Total Revenue",
        "Cost of Revenue": "CostOfRevenue",
        "Operating Income": "Operating Income",
        "Basic EPS": "Basic EPS"
    }
    available = {label: idx for label, idx in METRIC_MAP.items() if idx in income.index}
    if not available:
        st.warning("No matching metrics to plot.")
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 5 * len(available)))
    if len(available) == 1:
        axes = [axes]

    for ax, (label, idx) in zip(axes, available.items()):
        series = income.loc[idx].iloc[::-1]
        ax.plot(series.index.astype(str), series.values, marker='o', linewidth=2)
        ax.set_title(f"{label} Trend for {ticker}")
        ax.set_xlabel('Period')
        ax.set_ylabel(label)
        ax.grid(True)

    st.pyplot(fig)


def get_chatgpt_commentary(openai_client: OpenAI, income_str: str, ten_k: str, ten_q: str, ticker: str) -> str:
    """Generate detailed financial analysis using ChatGPT."""
    prompt = f"""
Please analyze the following financial documents for {ticker}:

1. Income Statement:
{income_str}

2. Latest 10-K Report:
{ten_k}

3. Latest 10-Q Report:
{ten_q}

You are a highly experienced financial analyst. Carefully examine the Income Statement and SEC filings above and perform the following tasks:

1. **Detailed Analysis**:
   - Identify key metrics, trends, and notable changes.
2. **Calculate & Interpret Key Ratios**:
   - Revenue Growth, Gross Margin %, Operating Margin, Net Profit Margin, EPS, ROE, ROA.
3. **Compare to Industry Standards**.
4. **Summary & Key Takeaways**.
5. **Recommendations**.
6. **Separate 10-K and 10-Q Analysis** with clear headings.

"""
    response = openai_client.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "system", "content": "You are a financial analyst providing detailed insights."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.set_page_config(page_title="Financial Analysis Dashboard", layout="wide")
st.title("🔍 Financial Analysis Dashboard")

# Initialize OpenAI client
client = init_openai_client()

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, AAPL):", value="MSFT").upper().strip()

if st.button("Generate Analysis"):
    with st.spinner("Fetching Income Statement..."):
        income_df = get_income_statement(ticker)
    
    st.subheader("📊 Income Statement")
    if income_df is not None and not income_df.empty:
        st.dataframe(income_df)
    else:
        st.error("No income statement data available.")
        st.stop()

    with st.spinner("Downloading SEC filings..."):
        ten_k, ten_q = download_and_parse_filings(ticker)

    st.subheader("📈 Trends")
    plot_income_statement_trends(income_df, ticker)

    st.subheader("🗃️ SEC Filings Preview")
    st.markdown("**Latest 10-K**")
    st.text_area("10-K Content", ten_k, height=200)
    st.markdown("**Latest 10-Q**")
    st.text_area("10-Q Content", ten_q, height=200)

    with st.spinner("Generating ChatGPT commentary..."):
        income_str = income_df.to_string()
        commentary = get_chatgpt_commentary(client, income_str, ten_k, ten_q, ticker)
    
    st.subheader("🤖 ChatGPT Analysis")
    st.markdown(commentary)
# End of app.py
