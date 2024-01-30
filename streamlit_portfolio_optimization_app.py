import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
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
            # Fetching stock data using yfinance
            df = yf.download(ticker_list, start=start_date, end=end_date)['Adj Close']

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

            # Function to fetch financial data for a stock using yfinance
            def fetch_financial_data(ticker_symbol):
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    income_statement = ticker.financials  # Fetch income statement
                    balance_sheet = ticker.balance_sheet  # Fetch balance sheet
                    cash_flow = ticker.cashflow  # Fetch cash flow statement

                    financial_data = {
                        'incomeStatement': income_statement,
                        'balanceSheet': balance_sheet,
                        'cashflowStatement': cash_flow
                    }
                    return financial_data
                except Exception as e:
                    return None

            # Select financial statement type using a dropdown
            selected_statement = st.selectbox('Select Financial Statement Type:', ['Income Statement', 'Balance Sheet', 'Statement of Cash Flows'])

            # Display financial statements for the selected type
            for ticker in ticker_list:
                try:
                    st.subheader(f'Financial Statements for {ticker}')
                    financial_data = fetch_financial_data(ticker)
                    if financial_data is not None:
                        st.write(selected_statement)
                        statement_data = financial_data.get(selected_statement.lower(), None)
                        if statement_data is not None:
                            statement_df = pd.DataFrame(statement_data)
                            st.table(statement_df)
                        else:
                            st.write(f'{selected_statement} data not available for {ticker}')
                    else:
                        st.write(f'Financial data not available for {ticker}')
                except Exception as e:
                    st.error(f'Error fetching financial data for {ticker}: {e}')

        except Exception as e:
            st.error('Error in processing: \n' + str(e))
