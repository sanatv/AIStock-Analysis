import os
import streamlit as st
import yfinance as yf
import pandas as pd
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from openai import OpenAI
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Page config & theming
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 2. Secrets & clients
# ------------------------------------------------------------------------------
OPENAI_KEY = st.secrets.get("openai_key")
OPENAI_ORG = st.secrets.get("openai_org")
if not OPENAI_KEY:
    st.error("🔒 Please configure your OpenAI API key in Streamlit secrets as 'openai_key'.")
    st.stop()

@st.cache_resource
def init_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_KEY, organization=OPENAI_ORG)

client = init_openai_client()

# ------------------------------------------------------------------------------
# 3. Data fetching & transformation
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_income_statement(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    return stock.financials

@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_balance_sheet(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    return stock.balance_sheet


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_company_info(ticker: str):
    return yf.Ticker(ticker).info

# SEC Downloader setup
dl = Downloader("Your Company Name", "your-email@example.com")

@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_and_parse_filings(ticker):
    filings = [("10-K", "10-K"), ("10-Q", "10-Q")]
    filing_texts = {}
    base_dir = os.path.join("sec-edgar-filings", ticker)

    try:
        for filing_name, filing_type in filings:
            dl.get(filing_type, ticker, limit=1)
            filing_dir = os.path.join(base_dir, filing_type)
            if os.path.exists(filing_dir):
                latest_filing_folder = sorted(os.listdir(filing_dir), reverse=True)[0]
                filing_file = os.path.join(filing_dir, latest_filing_folder, "full-submission.txt")
                with open(filing_file, 'r', encoding='utf-8') as file:
                    content = file.read()
                    soup = BeautifulSoup(content, 'lxml')
                    filing_texts[filing_name] = soup.get_text(separator='\\n')[:15000]
            else:
                filing_texts[filing_name] = f"No {filing_name} filing found."

    except Exception as e:
        st.error(f"Error: {e}")
        filing_texts["10-K"] = filing_texts["10-Q"] = f"Error: {e}"

    return filing_texts.get("10-K", ""), filing_texts.get("10-Q", "")

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
    
def plot_balance_trends(balance: pd.DataFrame, ticker: str) -> None:
    METRICS = {
        "Total Assets": "Total Assets",
        "Total Liabilities": "Total Liab"
    }
    available = {label: idx for label, idx in METRICS.items() if idx in balance.index}
    if not available:
        st.warning("No balance sheet metrics found to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, idx in available.items():
        series = balance.loc[idx].iloc[::-1]
        ax.plot(series.index.astype(str), series.values, label=label, marker='o')

    ax.set_title(f"{ticker} Balance Sheet Trends")
    ax.set_xlabel("Period")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def get_chatgpt_commentary(openai_client: OpenAI, income_str: str,balance_str: str,  ten_k: str, ten_q: str, ticker: str) -> str:
    prompt = f"""
Please analyze the following financial documents for {ticker}:

1. Income Statement:
{income_str}

2. Balance Sheet:
{balance_str}

3. Latest 10-K Report:
{ten_k}

4. Latest 10-Q Report:
{ten_q}

Tasks:
1. Detailed Analysis: key metrics, trends, and notable changes.
2. Calculate & Interpret Key Ratios: Revenue Growth, Gross Margin %, Operating Margin, Net Profit Margin, EPS,Liquidity Ratios, Levergae Ratios, Efficiency ratios, Book Value Ratios, ROE, ROA.
3. Compare to Industry Standards.
4. Summary & Key Takeaways.
5. Recommendations.
6. Separate 10-K and 10-Q Analysis with clear headings.
Create table and format it with Bold section where it makes sense.
"""
    response = openai_client.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "system", "content": "You are a financial analyst providing detailed insights."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ------------------------------------------------------------------------------
# 4. UI: Sidebar & Layout
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Stock Selection")
    ticker = st.text_input("Enter Ticker Symbol:", value="MSFT").upper().strip()
    show_analysis = st.button("Generate Full Analysis")

if show_analysis:
    st.markdown("## 📊 Company Overview")
    try:
        info = get_company_info(ticker)
        col1, col2, col3 = st.columns(3)
        col1.metric("Revenue", f"${info.get('totalRevenue', 0)/1e9:.2f}B" if info.get("totalRevenue") else "N/A")
        col2.metric("EPS", f"${info.get('forwardEps', 'N/A')}")
        col3.metric("P/E Ratio", f"{info.get('forwardPE', 'N/A')}")
    except:
        st.warning("Could not retrieve overview metrics.")

    # Tabs for structured view
    tabs = st.tabs(["📈 Income Statement", "📊 Balance Sheet", "📄 SEC Filings", "🤖 AI Commentary"])
    
    with tabs[0]:
        st.subheader("Income Statement (Raw)")
        with st.spinner("Fetching Income Statement..."):
            income_df = get_income_statement(ticker)

        if income_df is None or income_df.empty:
            st.error("No income statement data available.")
        else:
            st.dataframe(income_df)
            st.subheader("Income Statement Trends")
            plot_income_statement_trends(income_df, ticker)
    with tabs[1]:
        st.subheader("Balance Sheet (Raw)")
        with st.spinner("Fetching Balance Sheet..."):
            balance_df = get_balance_sheet(ticker)

    if balance_df is None or balance_df.empty:
        st.error("No balance sheet data available.")
    else:
        st.dataframe(balance_df)
        st.subheader("Balance Sheet Trends")
        plot_balance_trends(balance_df, ticker)


    with tabs[2]:
        st.subheader("SEC Filings")
        with st.spinner("Downloading and Parsing SEC filings..."):
            ten_k, ten_q = download_and_parse_filings(ticker)

        st.markdown("**Latest 10-K Filing (Preview)**")
        st.text_area("10-K Content", ten_k[:5000], height=150)
        st.markdown("**Latest 10-Q Filing (Preview)**")
        st.text_area("10-Q Content", ten_q[:5000], height=150)

    with tabs[3]:
        st.subheader("AI Analysis and Recommendations")
        with st.spinner("Generating commentary with AI..."):
            commentary = get_chatgpt_commentary(client, income_df.to_string(), balance_df.to_string(),ten_k, ten_q, ticker)
        st.markdown(commentary, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:0.8em; color:grey;'>"
        "Built with Streamlit • Data from Yahoo Finance & SEC Edgar • Powered by OpenAI • sanatv@gmail.com"
        "</div>",
        unsafe_allow_html=True
    )
