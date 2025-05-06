import os
import streamlit as st
import yfinance as yf
import pandas as pd
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from openai import OpenAI
import matplotlib.pyplot as plt
import requests
import plotly.graph_objects as go
import streamlit as st
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS




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


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

OPENAI_KEY = st.secrets.get("openai_key")
OPENAI_ORG = st.secrets.get("openai_org")
if not OPENAI_KEY:
    st.error("🔒 Please configure your OpenAI API key in Streamlit secrets as 'openai_key'.")
    st.stop()

@st.cache_resource
def init_openai_client_ai_commentry() -> OpenAI:
    return OpenAI(api_key=OPENAI_KEY, organization=OPENAI_ORG)

# client = init_openai_client_ai_commentry()


@st.cache_resource
def init_openai_client():
    api_key = st.secrets.get("openai_key")
    if not api_key:
        st.error("🔒 Please configure your OpenAI API key in Streamlit secrets.")
        st.stop()
    return ChatOpenAI(api_key=api_key, model="gpt-4.1", temperature=0)

search_tool = DuckDuckGoSearchRun()

def create_retriever(context):
    embeddings = OpenAIEmbeddings(api_key=st.secrets.get("openai_key"))
    vector_store = FAISS.from_texts([context], embeddings)
    return vector_store.as_retriever()

def setup_agent(company_context):
    retriever = create_retriever(company_context)
    retriever_tool = create_retriever_tool(
        retriever,
        name="company_data",
        description="Searches financial details about the company provided in the context."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable financial assistant. Use provided tools to answer questions."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [retriever_tool, search_tool]

    agent = create_openai_tools_agent(init_openai_client(), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    return agent_executor




# # Initialize OpenAI Client (cached)
# @st.cache_resource
# def init_openai_client():
#     return ChatOpenAI(model="gpt-4-turbo", temperature=0)





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

@st.cache_data(ttl=3600)

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

def add_yoy_change(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 2:
        return df  # Not enough periods to compare

    # Parse numeric from string and compute % change from last two periods
    def parse_num(x):
        if isinstance(x, str) and x.startswith("$"):
            return float(x.replace("$", "").replace("B", "").replace(",", ""))
        return x

    numeric_df = df.applymap(parse_num)

    # Compute change between most recent and previous column
    col_latest, col_prev = numeric_df.columns[:2]
    df["YoY Change"] = numeric_df[col_latest].subtract(numeric_df[col_prev]) \
        .divide(numeric_df[col_prev]) \
        .apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")

    return df
def highlight_growth(val):
    if isinstance(val, str) and "%" in val:
        try:
            v = float(val.replace("%", ""))
            if v > 0:
                return "color: green"
            elif v < 0:
                return "color: red"
        except:
            return ""
    return ""

def download_button(df: pd.DataFrame, filename: str):
    csv_data = df.to_csv().encode('utf-8')
    st.download_button(
        label=f"📥 Download {filename}",
        data=csv_data,
        file_name=f"{filename}.csv",
        mime="text/csv"
    )

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

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def generate_pdf_from_markdown(commentary: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Set up starting point
    text_object = c.beginText(40, 750)
    text_object.setFont("Helvetica", 10)

    for line in commentary.splitlines():
        if text_object.getY() < 50:  # Start a new page if at the bottom
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText(40, 750)
            text_object.setFont("Helvetica", 10)
        text_object.textLine(line.strip())

    c.drawText(text_object)
    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer.read()


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

   # Tabs for structured view
tabs = st.tabs(["📈 Income Statement", "📊 Balance Sheet", "📄 SEC Filings", "🤖 AI Commentary","💬 AI Chatbot","🧠 Vision Assistant (PDF/Image Understanding)"])
    
with tabs[0]:
    st.subheader("🔄 Income Flow (Latest Year)")

    # Fetch Income Statement
    with st.spinner("Fetching Income Statement..."):
        income_df = get_income_statement(ticker)

    if income_df is None or income_df.empty:
        st.warning("No income statement data available for this company.")
    else:
        import plotly.graph_objects as go

        # Safely detect latest year
        year_cols = [str(col) for col in income_df.columns if any(char.isdigit() for char in str(col))]
        latest_year_col = year_cols[0]

        # Key financial items for Sankey (full path)
        selected_items = [
            "Total Revenue",
            "Cost Of Revenue",
            "Gross Profit",
            "Operating Expense",
            "Research And Development",
            "Selling General And Administration",
            "Operating Income",
            "Tax Provision",
            "Net Income"
        ]

        # Prepare data
        df_flow = income_df.loc[income_df.index.isin(selected_items), [latest_year_col]].copy()
        df_flow[latest_year_col] = pd.to_numeric(df_flow[latest_year_col], errors='coerce').fillna(0)
        df_flow.index = df_flow.index.str.replace("_", " ").str.title()

        # Short aliases
        aliases = {
            "Research And Development": "R&D",
            "Selling General And Administration": "SG&A"
        }
        df_flow.rename(index=aliases, inplace=True)

        # Convert values to millions
        df_flow[latest_year_col] /= 1e6

        # Explicit flow structure ensuring visibility
        labels = [
            "Total Revenue", "Cost Of Revenue", "Gross Profit",
            "Operating Expense", "R&D", "SG&A",
            "Operating Income","Tax Provision", "Net Income"
        ]

        # Define logical Sankey flows clearly
        flow_map = [
            ("Total Revenue", "Cost Of Revenue"),
            ("Total Revenue", "Gross Profit"),
            ("Gross Profit", "Operating Expense"),
            ("Operating Expense", "R&D"),
            ("Operating Expense", "SG&A"),
            ("Gross Profit", "Operating Income"),
            ("Operating Income", "Net Income"),
            ("Operating Income","Tax Provision")
        ]

        # Construct Sankey data with error checking
        source, target, value, hovertext = [], [], [], []
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        revenue_value = df_flow.at["Total Revenue", latest_year_col] or 1

        for src, tgt in flow_map:
            src_val = df_flow.at[src, latest_year_col] if src in df_flow.index else 0
            tgt_val = df_flow.at[tgt, latest_year_col] if tgt in df_flow.index else 0
            flow_value = min(src_val, tgt_val)
            
            if flow_value > 0:
                source.append(label_to_index[src])
                target.append(label_to_index[tgt])
                value.append(flow_value)
                pct = (flow_value / revenue_value) * 100
                hovertext.append(f"{src} → {tgt}<br><b>${flow_value:,.1f}M</b><br>{pct:.2f}% of Revenue")

        # Clear color mapping
        node_colors = [
            "#1f77b4",  # Revenue - Blue
            "#ff7f0e",  # COGS - Orange
            "#2ca02c",  # Gross Profit - Green
            "#9467bd",  # Operating Expenses - Purple
            "#c49c94",  # R&D - Beige
            "#8c564b",  # SG&A - Brown
            "#17becf",  # Operating Income - Cyan
            "#18bedf",  # Tax Provision
            "#d62728"   # Net Income - Red
        ]

        # Sankey visualization
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="rgba(80,80,80,0)", width=1),
                label=labels,
                color=node_colors,
                hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font=dict(color="white"))
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                customdata=hovertext,
                hovertemplate="%{customdata}<extra></extra>",
                color="rgba(15,15,10,15)"
            )
        ))

        fig.update_layout(
            title_text=f"📊 Income Flow Breakdown – {latest_year_col[:4]}",
            font=dict(size=13, color="#111"),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)



        st.subheader("Income Statement (Raw)")
        with st.spinner("Fetching Income Statement..."):
            income_df = get_income_statement(ticker)

        if income_df is None or income_df.empty:
            st.warning("No income statement data available for this company.")
        else:
            formatted_income = clean_financial_dataframe(income_df, "Income")
            formatted_income = add_yoy_change(formatted_income)

            styled = formatted_income.style.applymap(highlight_growth, subset=["YoY Change"])
            st.dataframe(styled, use_container_width=True)

            download_button(formatted_income, f"{ticker}_income_statement")
            plot_income_statement_trends(income_df, ticker)

            

with tabs[1]:
	st.subheader("Balance Sheet (Raw)")
	with st.spinner("Fetching Balance Sheet..."):
		balance_df = get_balance_sheet(ticker)

	if balance_df is None or balance_df.empty:
	    st.warning("No balance sheet data available.")
	else:
	    formatted_bal = clean_financial_dataframe(balance_df, "Balance")
	    formatted_bal = add_yoy_change(formatted_bal)
	    # st.dataframe(formatted_bal, use_container_width=True)
	    
	    styled = formatted_bal.style.applymap(highlight_growth, subset=["YoY Change"])
	    st.dataframe(styled, use_container_width=True)
	    download_button(formatted_bal, f"{ticker}_balance_sheet")


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
    st.subheader("🤖 AI Analysis and Recommendations")

    # Prepare context from financial data
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

    if st.button("🧠 Generate AI Commentary"):
        with st.spinner("Generating commentary with AI..."):
            client = init_openai_client_ai_commentry()
            commentary = get_chatgpt_commentary(
                client,
                income_df.to_string(),
                balance_df.to_string(),
                ten_k,
                ten_q,
                ticker
            )

            st.markdown(commentary, unsafe_allow_html=True)

            st.download_button(
                label="📥 Download AI Commentary as PDF",
                data=generate_pdf_from_markdown(commentary),
                file_name=f"{ticker}_AI_Commentary.pdf",
                mime="application/pdf"
            )
    else:
        st.info("👈 Click the button above to generate AI commentary based on financials.")



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
    
import openai
import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

@st.cache_resource
def init_openai_sdk():
    return openai.OpenAI(api_key=st.secrets.get("openai_key"))
import random

def generate_follow_up_questions():
    options = [
        "Show me the revenue growth over the last 5 years.",
        "What is the latest analyst rating for this company?",
        "Summarize the recent earnings call highlights.",
        "What are the major competitors of this company?",
        "How has the company's operating margin changed recently?",
        "What risks are listed in their latest 10-K filing?",
        "How much debt is reported in the balance sheet?",
        "What was the year-over-year change in Net Income?",
    ]
    return random.sample(options, 3)

with tabs[4]:
    st.subheader("💬 GPT-4o Assistant (Financial Context + Optional Web Search)")
    layout_mode = st.radio(
    "Choose display style:",
    ("🗨️ Chat Bubbles", "📂 Expandable Q&A"),
    horizontal=True
    )

    # Toggle for web search
    use_web = st.toggle("🌐 Enable Web Search with GPT-4o", value=True)

    # Session history
    if "web_chat_history" not in st.session_state:
        st.session_state.web_chat_history = []

    # Combine financial context
    context_parts = []

    if 'income_df' in locals() and not income_df.empty:
        context_parts.append("### Income Statement:\n" + income_df.to_string())

    if 'balance_df' in locals() and not balance_df.empty:
        context_parts.append("### Balance Sheet:\n" + balance_df.to_string())

    if 'ten_k' in locals() and ten_k:
        context_parts.append("### 10-K Filing Preview:\n" + ten_k[:3000])

    if 'ten_q' in locals() and ten_q:
        context_parts.append("### 10-Q Filing Preview:\n" + ten_q[:3000])

    company_context = "\n\n".join(context_parts) if context_parts else "No structured data available."

    # User input
    user_question = st.text_input("🧠 Ask about this company:", key="gpt4o_input")

    if user_question:
        with st.spinner("Thinking with GPT-4o..."):
            try:
                client = init_openai_sdk()

                full_prompt = f"""
	You are a smart financial assistant. First, try to answer from the structured financial data below. 
	If not sufficient, use your web search tool (if allowed).
	
	==== COMPANY DATA ====
	{company_context}
	
	==== QUESTION ====
	{user_question}
	"""

                response = client.responses.create(
                    model="gpt-4o",
                    input=full_prompt,
                    tools=[{"type": "web_search"}] if use_web else []
                )

                reply = response.output_text
                st.session_state.web_chat_history.append((user_question, reply))
            except Exception as e:
                st.error(f"❌ GPT-4o Error: {e}")
                reply = None

    # Display past conversation
from streamlit.components.v1 import html
	
st.markdown("### 🗂️ Previous Conversations")

for idx, (q, a) in enumerate(reversed(st.session_state.web_chat_history), 1):
    if layout_mode == "🗨️ Chat Bubbles":
        with st.container():
            # User bubble
            st.markdown(f"""
            <div style="display: flex; align-items: start; margin-bottom: 10px;">
                <div style="font-size: 24px; margin-right: 10px;">👤</div>
                <div style="background-color: #f0f2f6; padding: 10px 15px; border-radius: 12px; max-width: 85%;">
                    <strong>You:</strong><br>{q}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # GPT bubble
            st.markdown(f"""
            <div style="display: flex; align-items: start; margin-bottom: 25px;">
                <div style="font-size: 24px; margin-right: 10px;">🧠</div>
                <div style="background-color: #e8f5e9; padding: 10px 15px; border-radius: 12px; max-width: 85%;">
                    <strong>GPT-4o:</strong><br>{a}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 🔥 Smart Follow-Up Suggestions (only after the latest answer)
        if idx == 1:
            st.markdown("#### 🔥 You might also ask:")
            for idx_suggestion, suggestion in enumerate(generate_follow_up_questions()):
                if st.button(f"💬 {suggestion}", key=f"chatbubble_suggestion_{idx}_{idx_suggestion}"):
                    client = init_openai_sdk()
                    full_prompt = f"""
You are a smart financial assistant. First, try to answer from the structured financial data below. 
If not sufficient, use your web search tool (if allowed).

==== COMPANY DATA ====
{company_context}

==== QUESTION ====
{suggestion}
"""
                    response = client.responses.create(
                        model="gpt-4o",
                        input=full_prompt,
                        tools=[{"type": "web_search"}] if use_web else []
                    )
                    reply = response.output_text
                    st.session_state.web_chat_history.append((suggestion, reply))

                    # Clear input box
                    st.session_state["gpt4o_input"] = ""
                    st.experimental_rerun()

    else:
        with st.expander(f"Q{idx}: {q}"):
            st.markdown(a)

            # 🔥 Smart Follow-Up Suggestions (only after latest answer)
            if idx == 1:
                st.markdown("#### 🔥 Suggested Next Questions:")
                for idx_suggestion, suggestion in enumerate(generate_follow_up_questions()):
                    if st.button(f"💬 {suggestion}", key=f"expander_suggestion_{idx}_{idx_suggestion}"):
                        st.session_state["gpt4o_input"] = suggestion
                        st.experimental_rerun()




        # Download as PDF
        if st.button("📄 Download Q&A as PDF"):
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            y = height - 50

            for i, (q, a) in enumerate(st.session_state.web_chat_history):
                for line in [f"Q{i+1}: {q}", f"A: {a}"]:
                    for chunk in [line[i:i+90] for i in range(0, len(line), 90)]:
                        c.drawString(40, y, chunk)
                        y -= 15
                        if y < 40:
                            c.showPage()
                            y = height - 50

            c.save()
            st.download_button(
                label="⬇️ Download as PDF",
                data=buffer.getvalue(),
                file_name=f"{datetime.now().strftime('%Y-%m-%d')}_chat_history.pdf",
                mime="application/pdf"
            )

with tabs[5]:
    st.subheader("🧠 GPT-4o Vision Assistant (PDF/Image Understanding)")

    uploaded_file = st.file_uploader("📎 Upload an image or PDF", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file:
        file_type = uploaded_file.type
        st.success(f"✅ File uploaded: `{uploaded_file.name}`")

        # Read and convert to base64
        import base64
        file_bytes = uploaded_file.read()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")

        # Set appropriate media type
        if "pdf" in file_type:
            media_type = "application/pdf"
        elif "image" in file_type:
            media_type = file_type  # either image/jpeg or image/png
        else:
            st.error("Unsupported file type.")
            media_type = None

        vision_prompt = st.text_input("💬 What do you want to ask GPT-4o about this file?")

        if vision_prompt and media_type:
            with st.spinner("🔍 GPT-4o is analyzing the uploaded file..."):
                try:
                    import openai
                    client = openai.OpenAI(api_key=st.secrets.get("openai_key"))

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": vision_prompt},
                                    {"type": "file", "file": {"media_type": media_type, "data": file_base64}}
                                ]
                            }
                        ],
                        max_tokens=1500
                    )

                    vision_answer = response.choices[0].message.content
                    st.markdown("### 🧠 GPT-4o Response")
                    st.markdown(vision_answer)

                except Exception as e:
                    st.error(f"❌ GPT-4o Vision Error: {e}")
    else:
        st.info("Upload a PDF or image to begin.")



st.markdown("---")
st.markdown(
	"<div style='text-align:center; font-size:0.8em; color:grey;'>"
	"Built with Streamlit • Data from Yahoo Finance & SEC Edgar • Powered by OpenAI • sanatv@gmail.com"
	"</div>",
	unsafe_allow_html=True
)
