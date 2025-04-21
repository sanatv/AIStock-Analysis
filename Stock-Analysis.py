import os
import streamlit as st
import yfinance as yf
import pandas as pd
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from openai import OpenAI
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="Financial Analysis Dashboard", layout="wide")

# Streamlit secrets configuration (secrets.toml)
# [openai]
# openai_key = "YOUR_API_KEY"
# openai_org = "YOUR_ORG_ID"  # Optional

# 1. Fetch and validate OpenAI credentials
OPENAI_KEY = st.secrets.get("openai_key")
OPENAI_ORG = st.secrets.get("openai_org")
if not OPENAI_KEY:
    st.error("🔒 Please configure your OpenAI API key in Streamlit secrets as 'openai_key'.")
    st.stop()

# 2. Initialize OpenAI client as a cached resource (non-pickleable)
@st.cache_resource
def init_openai_client() -> OpenAI:
    return OpenAI(
        api_key=OPENAI_KEY,
        organization=OPENAI_ORG
    )

# 3. Fetch income statement with caching
@st.cache_data(show_spinner=False)
def get_income_statement(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    return stock.financials

# 4. Download and parse SEC filings with caching

@st.cache_data(show_spinner=False)

def download_view_parse_filings(ticker, count=1):
    dl = Downloader("Vats Inc", "sanatv@gmail.com")
    try:
        filings = [("10-K", "10-K"), ("10-Q", "10-Q"), ("Annual Report", "ARS")]

        for filing_name, filing_type in filings:
            st.info(f"📥 Downloading latest {count} {filing_name} filing(s) for {ticker}...")
            dl.get(filing_type, ticker, limit=count)
            st.success(f"{filing_name} download complete.")

        base_dir = os.path.join("sec-edgar-filings", ticker)

        def parse_filing(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            soup = BeautifulSoup(content, 'lxml')
            text_content = soup.get_text(separator='\n')
            return text_content[:5000]

        for filing_name, filing_type in filings:
            filing_dir = os.path.join(base_dir, filing_type)
            if os.path.exists(filing_dir):
                latest_filing = sorted(os.listdir(filing_dir), reverse=True)[0]
                filing_file = os.path.join(filing_dir, latest_filing, "full-submission.txt")
                st.subheader(f"{filing_name} - Latest Filing")
                if os.path.exists(filing_file):
                    parsed_text = parse_filing(filing_file)
                    st.text_area("📄 Filing Preview (First 5000 characters)", parsed_text, height=400)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

# 5. Plot income statement trends
@st.cache_data(show_spinner=False)
def plot_income_statement_trends(income: pd.DataFrame, ticker: str) -> None:
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

# 6. Generate ChatGPT commentary

def get_chatgpt_commentary(
    openai_client: OpenAI,
    income_str: str,
    ten_k: str,
    ten_q: str,
    ticker: str
) -> str:
    prompt = (
        f"""
Please analyze the following financial documents for {ticker}:

1. Income Statement:
{income_str}

2. Latest 10-K Report:
{ten_k}

3. Latest 10-Q Report:
{ten_q}
Create table and format it with Bol section where it make sense
Tasks:
1. Detailed Analysis: key metrics, trends, and notable changes.
2. Calculate & Interpret Key Ratios: Revenue Growth, Gross Margin %, Operating Margin, Net Profit Margin, EPS, ROE, ROA.
3. Compare to Industry Standards.
4. Summary & Key Takeaways.
5. Recommendations.
6. Separate 10-K and 10-Q Analysis with clear headings.
"""
    )
    response = openai_client.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "system", "content": "You are a financial analyst providing detailed insights."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# 7. Streamlit UI

ticker = st.text_input(
    "Enter Stock Ticker (e.g., MSFT, AAPL):",
    value="MSFT"
).upper().strip()

if st.button("Generate Analysis"):
    client = init_openai_client()

    with st.spinner("Fetching Income Statement..."):
        income_df = get_income_statement(ticker)

    st.subheader("📊 Income Statement")
    if income_df is None or income_df.empty:
        st.error("No income statement data available.")
        st.stop()
    st.dataframe(income_df)

    # with st.spinner("Downloading SEC filings..."):
    #     ten_k, ten_q = download_and_parse_filings(ticker)


    st.title("📂 SEC Filing Downloader and Parser")
    ticker_input = st.text_input("Enter Ticker Symbol (e.g., AAPL):")
    filing_count = st.number_input("Number of filings to download per type:", min_value=1, max_value=5, value=1)

    if st.button("Download and Parse"):
        if ticker_input:
            download_view_parse_filings(ticker_input.upper(), filing_count)
        else:
            st.warning("Please enter a valid ticker symbol.")

    st.subheader("📈 Trends")
    plot_income_statement_trends(income_df, ticker)

    st.subheader("🗃️ SEC Filings Preview")
    st.markdown("**Latest 10-K**")
    st.text_area("10-K Content", ten_k, height=200)
    st.markdown("**Latest 10-Q**")
    st.text_area("10-Q Content", ten_q, height=200)

    with st.spinner("Generating ChatGPT commentary..."):
        commentary = get_chatgpt_commentary(
            init_openai_client(),
            income_df.to_string(),
            ten_k,
            ten_q,
            ticker
        )
    st.subheader("🤖 ChatGPT Analysis")
    st.markdown(commentary)
# End of app.py
