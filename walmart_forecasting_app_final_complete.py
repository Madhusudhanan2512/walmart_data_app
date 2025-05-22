
# Q&A Chatbot
st.subheader("💬 Ask a Question About Time Series Analysis")
st.caption("Examples: What is time series data?, What is seasonality?, What is ADF test?, What is RMSE?")

faq = {
    "What is time series data?": "Time series data is a sequence of data points collected at consistent time intervals.

Example: Weekly sales of a store.",
    "What is stationarity?": "A time series is stationary if its statistical properties like mean, variance, and autocorrelation remain constant over time.

Example: Flat seasonal sales.",
    "What is seasonality?": "Seasonality is a repeating pattern in time series data occurring at regular intervals.

Example: Increased sales every December.",
    "What is trend?": "A trend is the long-term increase or decrease in the data over time.

Example: A steady rise in yearly e-commerce sales.",
    "What is RMSE?": "Root Mean Squared Error (RMSE) measures the average prediction error.

Example: RMSE of 1000 means predictions are off by $1000 on average.",
    "What is FB Prophet?": "FB Prophet is a forecasting model developed by Facebook.

Used here to forecast Walmart store sales.",
    "What is autocorrelation?": "Autocorrelation measures how current values relate to past values.

Example: Sales this week may be similar to last week.",
    "What is the Augmented Dickey-Fuller (ADF) test?": "The ADF test checks whether a time series is stationary.

If p-value < 0.05, we reject the null hypothesis (data is stationary).",
    "Why use Prophet over ARIMA?": "Prophet is easier to configure and handles missing values, outliers, and non-linear trends better than ARIMA.",
    "What do confidence intervals in forecasting mean?": "A confidence interval gives a range where future values are likely to fall with a certain probability.

Example: A 99% confidence interval means there is a 99% chance the value falls in that range.",
    "What is decomposition in time series?": "Decomposition breaks down a time series into trend, seasonality, and residual components.

Used here to analyze Walmart store sales."
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
