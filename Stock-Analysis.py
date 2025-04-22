import os
import streamlit as st
import yfinance as yf
import pandas as pd
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from openai import OpenAI
import matplotlib.pyplot as plt
import requests

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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


# @st.cache_data(show_spinner=False, ttl=60 * 60)
# def get_company_info(ticker: str):
#     return yf.Ticker(ticker).info
# from fuzzywuzzy import process

@st.cache_data(ttl=3600)
# def search_ticker_by_name(company_name: str) -> str:
#     """
#     Search for a stock ticker by partial or full company name using fuzzy matching.
#     """
#     # Preload a common list of companies
#     sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
#     choices = dict(zip(sp500["Security"], sp500["Symbol"]))

#     # Fuzzy match against company names
#     match, score = process.extractOne(company_name, choices.keys())
#     ticker = choices.get(match)

#     return ticker

def global_search_ticker(query: str):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotes_count": 10, "news_count": 0}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return []

        data = response.json()
        results = []

        for item in data.get("quotes", []):
            if "symbol" not in item or item.get("quoteType") not in ["EQUITY", "ETF"]:
                continue

            result = {
                "symbol": item["symbol"],
                "shortname": item.get("shortname", ""),
                "exchange": item.get("exchange", ""),
                "type": item.get("quoteType", "")
            }

            results.append(result)

        # 🧠 Prioritize US exchanges (NMS = NASDAQ, NYQ = NYSE)
        prioritized = [r for r in results if r["exchange"] in ["NMS", "NYQ"]]
        others = [r for r in results if r not in prioritized]

        return prioritized + others

    except Exception as e:
        st.error(f"Ticker search error: {e}")
        return []

def clean_financial_dataframe(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # Format column dates
    df.columns = [
        col.strftime("%b %Y") if isinstance(col, pd.Timestamp) else str(col)
        for col in df.columns
    ]

    # Clean index (line items)
    df.index = df.index.astype(str)
    df.index = df.index.str.replace("_", " ").str.title()

    # Format values to $ in billions
    df = df.applymap(lambda x: f"${x/1e9:,.2f}B" if isinstance(x, (int, float)) else x)

    df.index.name = f"{label} Line Item"
    return df




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

@st.cache_data(ttl=60)
def get_realtime_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d", interval="1m")
        if data.empty:
            return None, None, None
        current_price = data["Close"].iloc[-1]
        previous_close = data["Close"].iloc[0]
        change = current_price - previous_close
        percent_change = (change / previous_close) * 100
        return round(current_price, 2), round(change, 2), round(percent_change, 2)
    except:
        return None, None, None

import altair as alt


@st.cache_data(ttl=600)
def get_chart_data(ticker: str, range_key: str) -> pd.DataFrame:
    interval_map = {
        "1D": "5m",
        "5D": "15m",
        "1M": "1h",
        "3M": "1d",
        "6M": "1d",
        "YTD": "1d",
        "1Y": "1d",
        "5Y": "1wk",
        "MAX": "1mo"
    }

    today = pd.Timestamp.today()
    start_map = {
        "1D": today - pd.Timedelta(days=1),
        "5D": today - pd.Timedelta(days=5),
        "1M": today - pd.DateOffset(months=1),
        "3M": today - pd.DateOffset(months=3),
        "6M": today - pd.DateOffset(months=6),
        "YTD": pd.Timestamp(year=today.year, month=1, day=1),
        "1Y": today - pd.DateOffset(years=1),
        "5Y": today - pd.DateOffset(years=5),
        "MAX": pd.Timestamp("1990-01-01")  # fallback for full history
    }

    interval = interval_map[range_key]
    start_date = start_map[range_key]

    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, interval=interval)
    df = df.reset_index()
    return df[["Date", "Close"]] if "Date" in df.columns else df[["Datetime", "Close"]]

import altair as alt

def plot_chart_with_range(ticker: str, range_key: str):
    df = get_chart_data(ticker, range_key)
    if df.empty:
        st.warning("⚠️ No chart data available for selected range.")
        return

    x_field = "Datetime" if "Datetime" in df.columns else "Date"

    chart = alt.Chart(df).mark_line(color="steelblue").encode(
        x=alt.X(f"{x_field}:T", title="Date"),
        y=alt.Y("Close:Q", title="Price ($)"),
        tooltip=[f"{x_field}:T", "Close:Q"]
    ).properties(
        title=f"{ticker} – {range_key} Price Trend",
        height=350
    ).interactive()

    st.altair_chart(chart, use_container_width=True)




def display_company_metrics(ticker: str):
    try:
        ticker_obj = yf.Ticker(ticker)
        fast_info = ticker_obj.fast_info or {}
        full_info = ticker_obj.info or {}

        # Get key metrics with smart fallback
        market_cap = fast_info.get("market_cap") or full_info.get("marketCap") or 0
        pe_ratio = fast_info.get("pe_ratio") or full_info.get("forwardPE")
        eps = full_info.get("forwardEps") or full_info.get("trailingEps")
        dividend_yield = fast_info.get("dividend_yield") or full_info.get("dividendYield")
        year_high = fast_info.get("year_high") or full_info.get("fiftyTwoWeekHigh")

        # Real-time price
        current_price, change, pct = get_realtime_price(ticker)
        arrow = "🔺" if change and change >= 0 else "🔻"
        color = "green" if change and change >= 0 else "red"

        # Layout: 2 rows of 3 columns
        st.markdown("## 📊 Company Metrics & Real-Time Price")
        row1 = st.columns(3)
        row2 = st.columns(3)

        row1[0].metric("📦 Market Cap", f"${market_cap/1e9:.2f}B" if market_cap else "N/A")
        row1[1].metric("📊 EPS", f"${eps:.2f}" if eps else "N/A")
        row1[2].metric("📈 P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")

        row2[0].metric("💸 Dividend Yield", f"{dividend_yield:.4f}" if dividend_yield else "N/A")
        row2[1].metric("🟢 52W High", f"${year_high:.2f}" if year_high else "N/A")

        if current_price is not None:
            row2[2].markdown(
                f"""
                <div style='font-size:1em; color:{color}; font-weight:bold;'>
                    ${current_price} {arrow}<br/>({change}, {pct}%)
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            row2[2].metric("Price", "N/A")

    except Exception as e:
        st.warning(f"⚠️ Unable to display company metrics: {e}")


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
        # model="o3-mini-2025-01-31",
        model="gpt-4.1",
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
	st.markdown("### 🔍 Smart Ticker Lookup")
	
	company_query = st.text_input("Enter Company Name or Ticker:", value="Apple")
	
	ticker = None  # final ticker to be resolved
	
	# Run smart search if query is entered
	if company_query:
	    results = global_search_ticker(company_query)
	
	    if results:
	        options = [
	            f"{r['symbol']} - {r['shortname']} ({r['exchange']})"
	            for r in results
	        ]
	        selected = st.selectbox("Select matching ticker:", options)
	        ticker = selected.split(" - ")[0]
	    else:
	        st.warning("⚠️ No matching ticker found.")
	
	# Always provide manual override
	manual_ticker = st.text_input("Or manually enter a known ticker (optional):")
	
	# Prioritize manual entry if filled
	if manual_ticker.strip():
	    ticker = manual_ticker.strip().upper()
	
	# Show final resolved ticker
	if ticker:
	    st.success(f"✅ Resolved Ticker: `{ticker}`")



display_company_metrics(ticker)
st.markdown("## 📉 Stock Price Chart")

range_options = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"]
selected_range = st.selectbox("Select Time Range:", range_options, index=6)  # default = 1Y

plot_chart_with_range(ticker, selected_range)


# plot_1d_price_chart(ticker)


# if show_analysis:
#     st.markdown("## 📊 Company Overview")
# try:
#     ticker_obj = yf.Ticker(ticker)
#     fast_info = ticker_obj.fast_info or {}
#     full_info = ticker_obj.info or {}

#     # Smart Fallbacks
#     market_cap = fast_info.get("market_cap") or full_info.get("marketCap")
#     pe_ratio   = fast_info.get("pe_ratio")   or full_info.get("forwardPE")
#     eps        = full_info.get("forwardEps") or full_info.get("trailingEps")
#     dividend_yield = fast_info.get("dividend_yield") or full_info.get("dividendYield")
#     year_high  = fast_info.get("year_high") or full_info.get("fiftyTwoWeekHigh")

#     # Display in top row
#     col1, col2, col3 = st.columns(3)
#     col1.metric("📦 Market Cap", f"${market_cap/1e9:.2f}B" if market_cap else "N/A")
#     col2.metric("📊 EPS", f"${eps:.2f}" if eps else "N/A")
#     col3.metric("📈 P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")

#     # # Optional extra metrics row
#     col4, col5 = st.columns(2)
#     col4.metric("💸 Dividend Yield", f"{dividend_yield:.2f}%" if dividend_yield else "N/A")
#     col5.metric("🔺 52W High", f"${year_high:.2f}" if year_high else "N/A")

# except Exception as e:
#     st.warning(f"⚠️ Unable to fetch summary metrics: {e}")


   # Tabs for structured view
tabs = st.tabs(["📈 Income Statement", "📊 Balance Sheet", "📄 SEC Filings", "🤖 AI Commentary","💬 Company Chatbot"])
    
with tabs[0]:
	st.subheader("Income Statement (Raw)")
	with st.spinner("Fetching Income Statement..."):
		income_df = get_income_statement(ticker)

	if income_df is None or income_df.empty:
	    st.warning("No income statement data available for this company.")
	else:
	    formatted_income = clean_financial_dataframe(income_df, "Income")
	    st.dataframe(formatted_income, use_container_width=True)
	    plot_income_statement_trends(income_df, ticker)
with tabs[1]:
	st.subheader("Balance Sheet (Raw)")
	with st.spinner("Fetching Balance Sheet..."):
		balance_df = get_balance_sheet(ticker)

	if balance_df is None or balance_df.empty:
	    st.warning("No balance sheet data available.")
	else:
	    formatted_bal = clean_financial_dataframe(balance_df, "Balance")
	    st.dataframe(formatted_bal, use_container_width=True)

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
# 👇 Chat context
company_context = f"""
Here is the financial data for {ticker}:

### Income Statement:
{income_df.to_string()}

### Balance Sheet:
{balance_df.to_string()}

### 10-K Summary:
{ten_k[:5000]}

### 10-Q Summary:
{ten_q[:5000]}
"""

with tabs[4]:
    st.subheader("💬 Ask Questions About the Company")

    user_input = st.text_input("Ask me anything about this company:")
    if user_input:
        openai_client = init_openai_client()

        messages = [
            {"role": "system", "content": f"You are a financial assistant. Use the context below to answer questions:\n\n{company_context}"},
            {"role": "user", "content": user_input}
        ]

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )
            reply = response.choices[0].message.content
            st.session_state.chat_history.append((user_input, reply))
        except Exception as e:
            st.error(f"Chatbot error: {e}")
            reply = None

    # Show chat history
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(f"Q{i}: {q}"):
            st.markdown(a)


st.markdown("---")
st.markdown(
	"<div style='text-align:center; font-size:0.8em; color:grey;'>"
	"Built with Streamlit • Data from Yahoo Finance & SEC Edgar • Powered by OpenAI • sanatv@gmail.com"
	"</div>",
	unsafe_allow_html=True
)
