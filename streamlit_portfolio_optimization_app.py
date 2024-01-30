import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier  # Import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns  # Import expected_returns
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

            # Financial Statement Analysis Section
            st.sidebar.header('Financial Statement Analysis')

            # Function to fetch financial data for a stock
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

            # Function to plot financial statements as tables
            def plot_financial_statements(financial_data):
                # Create a container for the financial statements
                st.subheader('Financial Statements')
                
                # Display Income Statement as a table
                st.write('Income Statement')
                income_statement = financial_data.get('income_statement', None)  # Replace with actual data source key
                if income_statement is not None:
                    income_statement_df = pd.DataFrame(income_statement)
                    st.table(income_statement_df)
                else:
                    st.write('Income Statement data not available')
            
                # Display Balance Sheet as a table
                st.write('Balance Sheet')
                balance_sheet = financial_data.get('balance_sheet', None)  # Replace with actual data source key
                if balance_sheet is not None:
                    balance_sheet_df = pd.DataFrame(balance_sheet)
                    st.table(balance_sheet_df)
                else:
                    st.write('Balance Sheet data not available')
            
                # Display Statement of Cash Flows as a table
                st.write('Statement of Cash Flows')
                cash_flows = financial_data.get('cash_flows', None)  # Replace with actual data source key
                if cash_flows is not None:
                    cash_flows_df = pd.DataFrame(cash_flows)
                    st.table(cash_flows_df)
                else:
                    st.write('Statement of Cash Flows data not available')

            for ticker in ticker_list:
                try:
                    st.subheader(f'Financial Statements for {ticker}')
                    financial_data = fetch_financial_data(ticker)
                    plot_financial_statements(financial_data)
                except Exception as e:
                    st.error(f'Error fetching financial data for {ticker}: {e}')

        except Exception as e:
            st.error('Error in processing: \n' + str(e))
