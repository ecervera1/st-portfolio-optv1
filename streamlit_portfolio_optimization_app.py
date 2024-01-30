import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from scipy import stats
import seaborn as sns

# Set style for plots
plt.style.use('fivethirtyeight')

# Title of the app
st.title('Portfolio Optimization Application')

# Sidebar inputs for stock tickers
st.sidebar.header('User Input Features')
default_tickers = ['AAPL', 'MSFT', 'AMZN']
ticker_list = st.sidebar.text_area('Enter the ticker symbols (comma separated):', ', '.join(default_tickers)).split(',')
ticker_list = [ticker.strip() for ticker in ticker_list]  # Strip whitespace

# Input for weights
weights_str = st.sidebar.text_input('Enter the weights for each stock (comma separated):', '0.33, 0.33, 0.34')
weights = np.array([float(w.strip()) for w in weights_str.split(',')])  # Convert to float and strip whitespace

# Date input
start_date = st.sidebar.date_input('Start date', datetime(2020, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.today())

# Button to run optimization
if st.sidebar.button('Run Optimization'):
    if len(ticker_list) != len(weights):
        st.error('The number of tickers and weights must be the same')
    else:
        try:
            # Fetching stock data
            df = pd.DataFrame()
            for ticker in ticker_list:
                try:
                    df[ticker] = web.DataReader(ticker, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
                except Exception as e:
                    st.error(f"Error fetching data for {ticker}: {e}")
                    continue

            # Expected returns and sample covariance
            mu = expected_returns.mean_historical_return(df)
            S = risk_models.sample_cov(df)

            # Optimize for maximal Sharpe ratio
            ef = EfficientFrontier(mu, S)
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()

            # Display results
            st.subheader('Optimized Weights')
            st.json(cleaned_weights)

            # Performance
            perf = ef.portfolio_performance(verbose=True)
            st.subheader('Performance')
            st.write('Expected annual return: {:.2f}%'.format(perf[0]*100))
            st.write('Annual volatility: {:.2f}%'.format(perf[1]*100))
            st.write('Sharpe Ratio: {:.2f}'.format(perf[2]))

        except Exception as e:
            st.error('Error in processing: \n' + str(e))

