import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from prophet import Prophet
from scipy.stats import pearsonr
from PIL import Image
import base64
import io

# ========= BACKGROUND SETUP =========
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
        }}
        .faq-box {{
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

bg_base64 = get_base64("background.png")
set_bg()

# ========= DATA LOAD =========
@st.cache_data
def load_data():
    df = pd.read_csv("walmart_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# ========= APP TITLE =========
st.title("🛒 Walmart Time Series Sales Analysis & Forecasting")

st.markdown("""
---
### 🏢 **Business Problem & Project Overview**
Walmart, the world's largest retailer, faces the constant challenge of accurately forecasting demand across its vast network. Poor demand planning leads to empty shelves or wasted inventory—**directly impacting profits, customer loyalty, and market share**.

This project leverages time series analysis and machine learning to:
- Forecast weekly sales for the **top-performing stores**
- Understand how **economic, environmental, and pricing factors** impact sales
- **Empower business leaders to make data-driven inventory, staffing, and promotion decisions**
---
""")

# =============== DATA PREVIEW ================
with st.expander("🔎 Preview Data"):
    st.dataframe(df.head())

# =============== CORRELATION ANALYSIS ================
st.header("1️⃣ Correlation Analysis: What Drives Walmart's Sales?")

st.write("""
#### 💡 **Why it Matters:**  
Understanding which factors most influence weekly sales helps Walmart **allocate advertising, adjust prices, and plan inventory**. Correlations reveal which levers (like unemployment or temperature) most move the needle.
""")

# Unemployment vs Weekly Sales
corr_unemp, _ = pearsonr(df["Weekly_Sales"], df["Unemployment"])
st.subheader("🧑‍💼 Unemployment Rate vs Weekly Sales")
fig1 = px.scatter(df, x="Unemployment", y="Weekly_Sales", opacity=0.6, trendline="ols")
st.plotly_chart(fig1, use_container_width=True)
st.info(f"""
- **Correlation coefficient:** {corr_unemp:.2f}  
- **Business interpretation:**  
  - **Slight negative relationship.** When unemployment falls (more people have jobs), sales tend to rise, albeit modestly.
  - **Action:** Stores in regions with declining unemployment could plan for **increased foot traffic** and **stock up on high-demand goods**.
  - **Leadership insight:** Monitor local job markets for early signs of sales upticks.
""")

# Temperature vs Weekly Sales
corr_temp, _ = pearsonr(df["Weekly_Sales"], df["Temperature"])
st.subheader("🌡️ Temperature vs Weekly Sales")
fig2 = px.scatter(df, x="Temperature", y="Weekly_Sales", opacity=0.6, trendline="ols")
st.plotly_chart(fig2, use_container_width=True)
st.info(f"""
- **Correlation coefficient:** {corr_temp:.2f}
- **Business interpretation:**  
  - **Slight negative relationship.** Cooler temperatures correlate with higher sales—perhaps due to seasonal spikes (holidays in winter).
  - **Action:** During colder weeks, **increase stock of seasonal items**, winter apparel, and comfort food.
  - **Leadership insight:** Consider running weather-based promotions or staffing accordingly.
""")

# CPI vs Weekly Sales
corr_cpi, _ = pearsonr(df["Weekly_Sales"], df["CPI"])
st.subheader("💸 Consumer Price Index (CPI) vs Weekly Sales")
fig3 = px.scatter(df, x="CPI", y="Weekly_Sales", opacity=0.6, trendline="ols")
st.plotly_chart(fig3, use_container_width=True)
st.info(f"""
- **Correlation coefficient:** {corr_cpi:.2f}
- **Business interpretation:**  
  - **Slight negative relationship.** Higher inflation (CPI) can reduce purchasing power and depress sales.
  - **Action:** If CPI rises sharply, consider **promotions, value packs, or price matching** to retain customers.
  - **Leadership insight:** Use CPI trends for early warning of demand drops.
""")

# =========== REGRESSION ANALYSIS ===========
st.header("2️⃣ Regression: Quantifying the Effect of Temperature on Sales")

st.write("""
#### 📈 **Why Regression?**
Regression estimates how much weekly sales change for every unit change in temperature. This helps Walmart **quantify sensitivity to external factors** for scenario planning.

- **Model Fit (R²):** Only 4% of the variance in sales is explained by temperature (statistically significant, but practically small).
- **Business interpretation:**  
  - **Temperature alone is a weak sales driver.** Other factors (holidays, promotions) likely dominate.
  - **Action:** **Focus forecasting efforts** on features with higher explanatory power.
  - **Leadership insight:** Temperature can be one input, but don’t over-prioritize it in planning.
""")

# =========== FORECASTING TOP STORES ===========
st.header("3️⃣ Forecasting: 12-Week Sales Outlook for Top Stores")

st.write("""
#### 📊 **Why Forecasting?**
Sales forecasts help Walmart decide **how much to order, when to run promotions, and how to staff each store**.

- The **Prophet model** is robust to outliers and captures holiday/seasonal effects well.
- **Action:** Use forecasts to reduce stockouts, plan staffing, and proactively run regional marketing.

---
""")

top_stores = [2, 4, 13, 14, 20]
selected_store = st.selectbox("Select a Top-Performing Store:", top_stores)
store_df = df[df["Store"] == selected_store].sort_values("Date")[["Date", "Weekly_Sales"]].rename(columns={"Date": "ds", "Weekly_Sales": "y"})

def forecast_sales(store_df, periods=12, ci=0.99):
    model = Prophet(interval_width=ci)
    model.fit(store_df)
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return forecast, model

forecast, model = forecast_sales(store_df)
fig4 = px.line()
fig4.add_scatter(x=store_df["ds"], y=store_df["y"], mode="markers", name="Actual Sales")
fig4.add_scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecasted Sales")
fig4.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(dash='dash'), name="Upper Bound", opacity=0.3)
fig4.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", line=dict(dash='dash'), name="Lower Bound", opacity=0.3)
fig4.update_layout(title=f"Weekly Sales Forecast for Store {selected_store}", xaxis_title="Date", yaxis_title="Weekly Sales")
st.plotly_chart(fig4, use_container_width=True)

# RMSE Calculation
actual = store_df["y"].values[-29:]
pred = forecast["yhat"].values[-29-12:-12] if len(forecast) > len(store_df) else forecast["yhat"].values[-29:]
rmse = np.sqrt(np.mean((actual - pred) ** 2))

st.info(f"""
- **Forecast Error (RMSE):** ${rmse:,.0f} per week.
- **Business interpretation:**  
    - For Store {selected_store}, the forecast is typically off by about ${rmse:,.0f} per week.
    - **If RMSE is low:** Inventory plans are reliable. Proceed with confidence.
    - **If RMSE is high:** Investigate local promotions, competitor actions, or special events that may distort typical sales.
- **Action:** Use forecast and confidence intervals for supply orders. Adjust quickly if observed sales deviate sharply from forecast.
""")

# =========== NOTEBOOK VISUALIZATIONS ===========
st.header("4️⃣ Direct Notebook Visualizations")

st.write("""
#### 🌐 **Why show notebook charts?**
- Preserves your EDA, regression visuals, or any hand-crafted matplotlib/plotly chart.
- Makes your work reproducible and audit-friendly for business leaders.
""")

# To use notebook images: Export your notebook's key plots as PNG/JPG (right-click in Jupyter, Save As...), and place in a folder, e.g. 'notebook_charts'.
# Here is how to display them:
import os
if os.path.isdir("notebook_charts"):
    chart_files = [f for f in os.listdir("notebook_charts") if f.endswith((".png", ".jpg"))]
    for chart in chart_files:
        st.image(f"notebook_charts/{chart}", caption=chart, use_column_width=True)

# =========== CONCLUSION ===========
st.header("5️⃣ Business Takeaways")

st.markdown("""
- **Forecasting accuracy enables confident business decisions.** Trustworthy forecasts help minimize costs, lost sales, and overstock.
- **Data-driven culture:** Regularly update models with new data. Monitor RMSE for each store; where it's high, investigate and fine-tune.
- **Act on insights:** React to local job market changes, inflation, and seasonality for proactive business moves.
- **Continuous improvement:** Incorporate new data (e.g., online sales, special events) for even sharper forecasts.
""")

# =========== INTERACTIVE FAQ/Q&A ===========

st.header("❓ Interactive Q & A: Time Series & Project Concepts")

# FAQ Dictionary
faq_dict = {
    "What is a time series?": "A sequence of data points collected over time, such as weekly sales.",
    "Why is forecasting important in retail?": "It enables Walmart to optimize inventory, staffing, and promotions—maximizing profit and customer satisfaction.",
    "What is seasonality?": "Regular, repeating patterns in sales (like holidays or summer spikes).",
    "What does a confidence interval mean in forecasting?": "It gives a range (e.g., 99%) where the true future sales value is expected to fall.",
    "What is RMSE?": "Root Mean Squared Error: measures the average forecast error. Lower RMSE = better model.",
    "What is the main limitation of using only temperature for sales prediction?": "It ignores other drivers like promotions, holidays, and economic trends.",
    "What is the role of outlier handling?": "It ensures that unusual spikes/drops don't skew model predictions.",
    "How often should models be updated?": "Regularly—ideally, whenever new sales data is available.",
    "What is the Prophet model?": "A robust forecasting tool by Facebook that handles trend, seasonality, and holidays."
}

# Searchable FAQ
search = st.text_input("Type a question or keyword:")
if search:
    found = False
    for q, a in faq_dict.items():
        if search.lower() in q.lower() or search.lower() in a.lower():
            st.markdown(f"<div class='faq-box'><b>Q:</b> {q}<br><b>A:</b> {a}</div>", unsafe_allow_html=True)
            found = True
    if not found:
        st.warning("No matching FAQ found. Try another keyword.")
else:
    for q, a in faq_dict.items():
        st.markdown(f"<div class='faq-box'><b>Q:</b> {q}<br><b>A:</b> {a}</div>", unsafe_allow_html=True)

import streamlit as st
import google.generativeai as genai
import pdfplumber
import nbformat
import pandas as pd
pip install --upgrade google-generativeai


# === Functions to Extract Context ===

def extract_text_from_pdf(pdf_path, max_chars=2000):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join([page.extract_text() or '' for page in pdf.pages])
        return text[:max_chars]
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_notebook(nb_path, max_chars=1000):
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        text = ''
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                text += cell.source + '\n'
            elif cell.cell_type == 'code':
                text += f"[Code cell]: {cell.source[:100]}...\n"
        return text[:max_chars]
    except Exception as e:
        return f"Error reading notebook: {e}"

def extract_data_summary(csv_path, max_rows=3):
    try:
        df = pd.read_csv(csv_path)
        summary = []
        summary.append(f"Columns: {list(df.columns)}")
        summary.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        summary.append(f"Sample data:\n{df.head(max_rows).to_string(index=False)}")
        for col in df.select_dtypes(include='number').columns:
            summary.append(f"Stats for '{col}': min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
        if 'Date' in df.columns:
            summary.append(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        if 'Store' in df.columns:
            summary.append(f"Number of unique stores: {df['Store'].nunique()}")
        return "\n".join(summary)
    except Exception as e:
        return f"Error extracting data summary: {e}"

# === Streamlit UI ===

st.markdown("---")
st.header("🔎 Advanced: Ask Anything About This Project (Gemini Chatbot)")

# API key input (for local or use st.secrets for deployment)
api_key = st.secrets["gemini_api_key"] if "gemini_api_key" in st.secrets else st.text_input("Enter your Gemini API Key:", type="password")

if api_key:
    genai.configure(api_key=api_key)

    # Load all context sources
    pdf_text = extract_text_from_pdf("Project - Walmart Time series analysis - Research Report.pdf", max_chars=2000)
    nb_text = extract_text_from_notebook("Project_Walmart_time_series_analysis.ipynb", max_chars=1000)
    data_text = extract_data_summary("walmart_cleaned.csv", max_rows=3)

    context = f"""
PROJECT REPORT:
{pdf_text}

NOTEBOOK SUMMARY:
{nb_text}

DATA SUMMARY:
{data_text}
"""
    st.write("The chatbot uses your project report, notebook, and a summary of your data to answer questions.")

    user_q = st.text_area("Ask a question about the project or data:")

    if st.button("Ask Gemini"):
        if not user_q.strip():
            st.warning("Please type your question.")
        else:
            prompt = f"""You are a helpful data science project expert. Use ONLY the information in the following project context to answer the user's question. If asked about data, respond using the 'DATA SUMMARY' section.

PROJECT CONTEXT:
{context}

QUESTION:
{user_q}

ANSWER:
"""
            with st.spinner("Gemini is thinking..."):
                try:
                    model = genai.GenerativeModel("models/gemini-1.5-pro-002")
                    response = model.generate_content(prompt)
                    st.success(response.text)
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
else:
    st.info("Please enter your Gemini API key to use the chatbot.")

