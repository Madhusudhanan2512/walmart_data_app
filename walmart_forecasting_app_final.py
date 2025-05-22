
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
    "What is time series data?": "Time series data is data recorded at regular time intervals. Example: weekly sales.",
    "What is stationarity?": "Stationarity means the statistical properties of the time series do not change over time.",
    "What is seasonality?": "Seasonality refers to repeating short-term cycles in a time series. Example: holiday sales.",
    "What is FB Prophet?": "Prophet is a time series forecasting model by Facebook. It handles trend, seasonality, and holidays.",
    "What is RMSE?": "Root Mean Squared Error, a metric to measure prediction error magnitude.",
    "What is the ADF test?": "ADF (Augmented Dickey-Fuller) test checks if a time series is stationary."
}
def get_answer(q):
    match = difflib.get_close_matches(q, faq.keys(), n=1, cutoff=0.4)
    return faq[match[0]] if match else "Sorry, I don't have an answer for that."

q = st.text_input("Ask your question:")
if q:
    st.markdown(f"**🤖 Answer:** {get_answer(q)}")
