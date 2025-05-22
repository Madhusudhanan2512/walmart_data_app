
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import linregress
from prophet import Prophet
from prophet.plot import plot_plotly
import difflib

st.set_page_config(page_title="📈 Walmart Time Series Forecasting", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('background.png');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Walmart Time Series Sales Forecasting & Analysis")

df = pd.read_csv("walmart_cleaned.csv", parse_dates=["Date"])

# Correlation Section
st.subheader("🔍 Correlation Analysis with Weekly Sales")
for col in ['Unemployment', 'Temperature', 'CPI']:
    fig = px.scatter(df, x=col, y='Weekly_Sales', trendline='ols', title=f'{col} vs Weekly Sales')
    st.plotly_chart(fig, use_container_width=True)

# Regression Section
st.subheader("📉 Regression Analysis")
reg_result = linregress(df['Temperature'], df['Weekly_Sales'])
st.markdown(f"R^2: {reg_result.rvalue**2:.2f} | p-value: {reg_result.pvalue:.4f}")
fig = px.scatter(df, x='Temperature', y='Weekly_Sales', trendline='ols', title='Temperature vs Weekly Sales')
st.plotly_chart(fig, use_container_width=True)

# Forecasting with Prophet
st.subheader("🔮 Forecasting for Top Stores")
top_stores = [2, 4, 13, 14, 20]
for store_id in top_stores:
    st.markdown(f"#### Store {store_id}")
    store_df = df[df['Store'] == store_id][['Date', 'Weekly_Sales']].rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    store_df['ds'] = pd.to_datetime(store_df['ds'], errors='coerce')
    store_df.dropna(inplace=True)
    model = Prophet()
    model.fit(store_df)
    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig, use_container_width=True)

# Q&A Chatbot
st.subheader("💬 Ask a Question About Time Series Analysis")
st.caption("Try: What is seasonality?, What is stationarity?, What is FB Prophet?")
faq = {
    """What is ACF (Autocorrelation Function)?""": """**ACF (Autocorrelation Function)** shows how current values in a time series relate to **past values** at various lags.

- Helps detect repeating patterns or persistence over time.
- Used to identify seasonality or trend strength.

**Example:**
If ACF lag-1 = 0.9, today's sales are highly correlated with yesterday’s.""",
    """What is PACF (Partial Autocorrelation Function)?""": """**PACF** measures the correlation between current and past values **after removing the effect of intermediate lags**.

- Useful to determine how many lags to include in AR (AutoRegressive) models.
- Unlike ACF, it isolates the effect of each specific lag.

**Example:**
If PACF at lag 2 is strong, there’s direct correlation between today and 2 days ago.""",
    """What are the types of seasonality?""": """Seasonality can appear in different time patterns:

- **Daily**: Hourly website traffic
- **Weekly**: Grocery store weekend sales
- **Monthly**: Retail spikes around payday
- **Yearly**: Holiday or weather-based patterns

Prophet and decomposition methods detect and model such seasonality.""",
    """What is model drift in time series?""": """**Model drift** happens when your model's accuracy declines over time because:

- Customer behavior has changed
- External factors have shifted
- Holidays/events differ year to year

**Solution:**
Retrain models regularly and use rolling forecasts or cross-validation to adapt.""",
    """What is differencing in time series?""": """**Differencing** is a technique to remove trend and make a series stationary.

- First-order differencing: `y(t) - y(t-1)`
- Often applied before ARIMA modeling

**Benefit:** Makes time series more suitable for modeling by stabilizing mean.""",
}
def get_answer(q):
    match = difflib.get_close_matches(q, faq.keys(), n=1, cutoff=0.4)
    return faq[match[0]] if match else "Sorry, I don't have an answer for that."

q = st.text_input("Ask your question:")
if q:
    st.markdown(f"**🤖 Answer:** {get_answer(q)}")
