
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import linregress
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="📈 Walmart Time Series Forecasting", layout="wide")

# Background styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("image.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Walmart Time Series Sales Forecasting & Analysis")

# 1. Project Description
st.markdown("""<small>
<h4>📘 Project Overview</h4>
<b>Problem:</b> Walmart struggles with aligning supply and demand, leading to inventory inefficiencies. This project applies data-driven approaches to analyze factors affecting sales and forecast demand for top-performing stores.<br><br>

<b>Objectives:</b>
<ul>
<li>Analyze the impact of <b>Unemployment, Temperature, and CPI</b> on Weekly Sales</li>
<li>Perform correlation and regression analysis</li>
<li>Forecast next 12 weeks of sales for the top 5 stores using <b>FB Prophet</b></li>
</ul>
</small>""", unsafe_allow_html=True)

# 2. Load cleaned data
df = pd.read_csv("walmart_cleaned.csv", parse_dates=["Date"])

# Correlation Analysis
st.subheader("🔍 Correlation Analysis with Weekly Sales")

corr_text = '''
We evaluated how strongly different variables correlate with Weekly Sales.

- **Unemployment vs Weekly Sales**: r = -0.10 → Weak negative correlation
- **Temperature vs Weekly Sales**: r = -0.06 → Weak negative correlation
- **CPI vs Weekly Sales**: r = -0.07 → Weak negative correlation

This suggests higher sales may occur when unemployment, CPI, and temperature drop.
'''
st.markdown(corr_text)

# Plot correlations
cols_to_check = ['Unemployment', 'Temperature', 'CPI']
for col in cols_to_check:
    fig = px.scatter(df, x=col, y='Weekly_Sales', trendline='ols', title=f'{col} vs Weekly Sales')
    st.plotly_chart(fig, use_container_width=True)

# Regression analysis
st.subheader("📉 Regression Analysis (Temperature vs Weekly Sales)")

reg_result = linregress(df['Temperature'], df['Weekly_Sales'])

st.markdown(f"""
**Interpretation:**

- R^2: {reg_result.rvalue**2:.2f}
- p-value: {reg_result.pvalue:.4f}

While statistically significant (p < 0.05), the R^2 is only {reg_result.rvalue**2:.2f}, meaning the model explains very little variance in Weekly Sales.
""")

fig_reg = px.scatter(df, x='Temperature', y='Weekly_Sales', trendline='ols', title="OLS Regression: Temperature vs Weekly Sales")
st.plotly_chart(fig_reg, use_container_width=True)

# Forecasting
st.subheader("🔮 Sales Forecasting Using FB Prophet")
st.markdown("We use Prophet to forecast 12 weeks of future sales for the top 5 stores with the best historical performance.")

top_stores = [2, 4, 13, 14, 20]

for store_id in top_stores:
    st.markdown(f"### 📦 Forecast for Store {store_id}")
    store_df = df[df["Store"] == store_id][["Date", "Weekly_Sales"]].rename(columns={"Date": "ds", "Weekly_Sales": "y"})

    model = Prophet(interval_width=0.99)
    model.fit(store_df)

    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)

    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("**Insights:** The forecast accounts for trend and seasonality. The blue shaded region represents 99% confidence interval.")



# Q&A Chatbot
st.subheader("💬 Ask a Question About Time Series Analysis")
st.caption("Examples: What is time series data?, What is seasonality?, What is ADF test?, What is RMSE?")

faq = {
    "What is time series data?": """Time series data is a sequence of data points collected at consistent time intervals.\n\n**Example:** Weekly sales of a store.""",
    "What is stationarity?": """A time series is stationary if its statistical properties like mean, variance, and autocorrelation remain constant over time.\n\n**Example:** Flat seasonal sales.""",
    "What is seasonality?": """Seasonality is a repeating pattern in time series data occurring at regular intervals.\n\n**Example:** Increased sales every December.""",
    "What is trend?": """A trend is the long-term increase or decrease in the data over time.\n\n**Example:** A steady rise in yearly e-commerce sales.""",
    "What is RMSE?": """Root Mean Squared Error (RMSE) measures the average prediction error.\n\n**Example:** RMSE of 1000 means predictions are off by $1000 on average.""",
    "What is FB Prophet?": """FB Prophet is a forecasting model developed by Facebook.\n\n**Used here** to forecast Walmart store sales.""",
    "What is autocorrelation?": """Autocorrelation measures how current values relate to past values.\n\n**Example:** Sales this week may be similar to last week.""",
    "What is the Augmented Dickey-Fuller (ADF) test?": """The ADF test is a statistical test used to check whether a time series is stationary.

**If p-value < 0.05**, we reject the null hypothesis (data is stationary).

**Used in this project** to test if Walmart store sales are stationary before forecasting.""",
    "Why is stationarity important in time series forecasting?": """Many forecasting models assume that the data has constant mean and variance over time (stationarity).

Non-stationary data can produce misleading forecasts unless transformed (e.g., via differencing or log transform).""",
    "Why use Prophet over ARIMA?": """Prophet is more robust to missing values, outliers, and non-linear trends.

It is also easier to use and configure compared to ARIMA, which requires stationarity and manual tuning.""",
    "What do confidence intervals in forecasting mean?": """A confidence interval gives a range where future values are likely to fall with a certain probability.

**Example:** A 99% confidence interval means there is a 99% chance the future value falls within that shaded region.""",
    "How is correlation different from regression?": """Correlation measures the strength of association between two variables (but not causality).

Regression predicts one variable based on another, assuming a linear relationship.

**Both are used in this project** to explore and explain relationships in Walmart sales.""",
    "What are the limitations of RMSE?": """RMSE is sensitive to large errors (due to squaring) and doesn't explain *why* errors occur.

It also doesn't distinguish between over- and under-predictions.

**In this project:** Used to measure average deviation between actual and predicted sales.""",
    "What is decomposition in time series?": """Decomposition breaks down a time series into trend, seasonality, and residual components.

**Used in this project** to understand patterns in store sales before forecasting."""
}

def get_faq_answer(question):
    matches = difflib.get_close_matches(question, faq.keys(), n=1, cutoff=0.4)
    if matches:
        return faq[matches[0]]
    else:
        return "❓ Sorry, I don't have an answer for that. Try rephrasing your question."

user_question = st.text_input("Type your question here")
if user_question:
    response = get_faq_answer(user_question)
    st.markdown(f"**🤖 Answer:**\n{response}")
