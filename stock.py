import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

class StockData:
    @staticmethod
    def get_sp500_tickers():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        return table['Symbol'].tolist()

    @staticmethod
    def get_stock_data(ticker, start_date, end_date):
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        return data, stock

    @staticmethod
    def calculate_indicators(data, info):
        # Technical indicators
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
        
        # MACD
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)
        
        # Fundamental indicators
        data['Market_Cap'] = info.get('marketCap', np.nan)
        data['P/E_Ratio'] = info.get('trailingPE', np.nan)
        data['Dividend_Yield'] = info.get('dividendYield', np.nan)
        data['Price_to_Book'] = info.get('priceToBook', np.nan)
        data['Debt_to_Equity'] = info.get('debtToEquity', np.nan)
        data['Free_Cash_Flow'] = info.get('freeCashflow', np.nan)
        
        return data

class MLModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    @property
    def feature_importances(self):
        return self.model.feature_importances_

class TradingStrategy:
    def __init__(self, num_stocks=30):
        self.num_stocks = num_stocks
        self.ml_model = MLModel()

    def select_top_stocks(self):
        tickers = StockData.get_sp500_tickers()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data
        
        stock_data = {}
        stock_info = {}
        for ticker in tickers:
            try:
                data, stock = StockData.get_stock_data(ticker, start_date, end_date)
                if len(data) > 0:
                    stock_info[ticker] = stock.info
                    stock_data[ticker] = StockData.calculate_indicators(data, stock_info[ticker])
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        features = ['MA50', 'MA200', 'RSI', 'Volatility', 'MACD', 'Signal_Line', 
                    'BB_Upper', 'BB_Lower', 'Market_Cap', 'P/E_Ratio', 'Dividend_Yield', 
                    'Price_to_Book', 'Debt_to_Equity', 'Free_Cash_Flow']
        
        X, y = [], []
        for ticker, data in stock_data.items():
            if len(data) > 200 and all(feature in data.columns for feature in features):
                X.append(data[features].iloc[-1].values)
                y.append(data['Daily_Return'].iloc[-20:].mean())  # Use average return over last 20 days as target
        
        X, y = np.array(X), np.array(y)
        
        self.ml_model.train(X, y)
        predictions = self.ml_model.predict(X)
        
        top_indices = predictions.argsort()[-self.num_stocks:][::-1]
        top_stocks = [list(stock_data.keys())[i] for i in top_indices]
        
        table_data = []
        for ticker in top_stocks:
            data = stock_data[ticker]
            info = stock_info[ticker]
            table_data.append({
                'Stock': ticker,
                'Name': info.get('longName', 'N/A'),
                'Price': data['Close'].iloc[-1],
                'Market Cap': info.get('marketCap', 'N/A'),
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'Dividend Yield': info.get('dividendYield', 'N/A'),
                'RSI': data['RSI'].iloc[-1],
                'Volatility': data['Volatility'].iloc[-1],
                'MACD': data['MACD'].iloc[-1],
                'Price to Book': info.get('priceToBook', 'N/A'),
                'Debt to Equity': info.get('debtToEquity', 'N/A'),
                'Free Cash Flow': info.get('freeCashflow', 'N/A')
            })
        
        return top_stocks, self.ml_model.feature_importances, pd.DataFrame(table_data)

    @staticmethod
    def create_custom_index(stocks, weights, start_date, end_date):
        data = pd.DataFrame()
        for stock in stocks:
            stock_data, _ = StockData.get_stock_data(stock, start_date, end_date)
            data[stock] = stock_data['Close']
        
        normalized_data = data / data.iloc[0]
        index = (normalized_data * weights).sum(axis=1)
        return index

class LLMExplainer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key) if api_key else None

    def get_explanation(self, stocks, feature_importances):
        if not self.client:
            return "LLM explanation not available. Please provide a valid OpenAI API key."

        features = ['MA50', 'MA200', 'RSI', 'Volatility', 'MACD', 'Signal_Line', 
                    'BB_Upper', 'BB_Lower', 'Market_Cap', 'P/E_Ratio', 'Dividend_Yield', 
                    'Price_to_Book', 'Debt_to_Equity', 'Free_Cash_Flow']
        
        feature_importance_str = ", ".join([f"{feature}: {importance:.4f}" for feature, importance in zip(features, feature_importances)])

        prompt = f"""Explain why these stocks might have been selected for an investment portfolio, 
        given the following feature importances: 
        {feature_importance_str}
        
        Stocks: {', '.join(stocks[:5])} (showing top 5 out of {len(stocks)}).
        
        Please provide insights on:
        1. Which features seem to be the most important in the stock selection process?
        2. How might these features contribute to the potential performance of the selected stocks?
        3. Are there any notable patterns or characteristics among the top selected stocks?
        4. What potential risks or considerations should investors be aware of when using this selection method?
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a financial analyst explaining stock selections based on machine learning model outputs."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in LLM explanation: {str(e)}"

import time

def main():
    st.title('Advanced ML-Enhanced Algorithmic Trading Strategy')

    # Sidebar
    st.sidebar.header('Strategy Parameters')
    num_stocks = st.sidebar.slider('Number of stocks for index', 10, 50, 10)
    investment_period = st.sidebar.slider('Investment period (days)', 30, 365, 90)

    # Add explanation in sidebar
    st.sidebar.header('How it works')
    st.sidebar.write("""
    THIS IS AN EXPERIMENTAL TRADING STRATEGY. USE AT YOUR OWN RISK.
    
    This advanced trading strategy uses machine learning to select top-performing stocks from the S&P 500.

    1. Data Collection: We fetch 2 years of historical data for all S&P 500 stocks.
    2. Feature Engineering: We calculate various technical and fundamental indicators:
       - Technical: Moving Averages, RSI, Volatility, MACD, Bollinger Bands
       - Fundamental: Market Cap, P/E Ratio, Dividend Yield, Price to Book, Debt to Equity, Free Cash Flow
    3. ML Model: A Random Forest Regressor is trained to predict short-term returns based on these features.
    4. Stock Selection: The model predicts returns for all stocks, and we select the top performers.
    5. Custom Index: We create an equally-weighted index of the selected stocks.
    6. Performance Comparison: We compare our custom index against the S&P 500.

    This enhanced model considers both technical and fundamental factors, providing a more comprehensive analysis for stock selection.
    """)

    # Initialize strategy
    strategy = TradingStrategy(num_stocks)

    # Select top stocks
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text('Initializing stock selection process...')
    time.sleep(1)

    status_text.text('Fetching S&P 500 tickers...')
    tickers = StockData.get_sp500_tickers()
    progress_bar.progress(10)
    time.sleep(1)

    status_text.text('Collecting historical data and calculating indicators...')
    total_tickers = len(tickers)
    for i, ticker in enumerate(tickers):
        try:
            # This part would be inside your TradingStrategy.select_top_stocks method
            # We're simulating it here for the purpose of logging
            status_text.text(f'Processing {ticker} ({i+1}/{total_tickers})')
            time.sleep(0.1)  # Simulate processing time
            progress_bar.progress(10 + int(40 * (i+1) / total_tickers))
        except Exception as e:
            status_text.text(f'Error processing {ticker}: {str(e)}')
            time.sleep(1)

    status_text.text('Training ML model...')
    progress_bar.progress(60)
    time.sleep(2)  

    status_text.text('Selecting top stocks...')
    progress_bar.progress(80)
    time.sleep(1)

    top_stocks, feature_importances, stock_table = strategy.select_top_stocks()
    progress_bar.progress(100)
    status_text.text('Stock selection complete!')
    time.sleep(1)

    status_text.empty()
    progress_bar.empty()

    st.write(f'Top {num_stocks} stocks selected for index composition:')
    st.dataframe(stock_table)

    # LLM Explanation
    api_key = os.environ.get("OPENAI_API_KEY") or st.text_input("Enter your OpenAI API key:", type="password")
    llm_explainer = LLMExplainer(api_key)
    
    explanation_status = st.empty()
    explanation_status.text('Generating explanation...')
    explanation = llm_explainer.get_explanation(top_stocks, feature_importances)
    explanation_status.empty()
    
    st.write("Explanation of stock selection:")
    st.write(explanation)

    # Calculate dates and create custom index
    end_date = datetime.now()
    start_date = end_date - timedelta(days=investment_period)
    two_year_start = end_date - timedelta(days=365*2)

    index_status = st.empty()
    index_status.text('Creating custom index and fetching historical data...')
    
    sp500 = yf.Ticker('^GSPC')
    sp500_data = sp500.history(start=start_date, end=end_date)['Close']
    custom_index = strategy.create_custom_index(top_stocks, [1/num_stocks]*num_stocks, start_date, end_date)

    # Fetch 2 years of historical data for each stock
    historical_data = {}
    for i, stock in enumerate(top_stocks):
        index_status.text(f'Fetching historical data for {stock} ({i+1}/{len(top_stocks)})')
        data, _ = StockData.get_stock_data(stock, two_year_start, end_date)
        historical_data[stock] = data['Close']

    index_status.empty()

    # Create a plot for 2-year trading history
    st.write('Generating 2-year trading history plot...')
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for stock in top_stocks:
        fig.add_trace(
            go.Scatter(x=historical_data[stock].index, y=historical_data[stock], name=stock),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(x=sp500_data.index, y=sp500_data, name='S&P 500', line=dict(color='black', width=3)),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="2-Year Trading History of Selected Stocks",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title="Stocks",
        height=600,
    )

    fig.update_yaxes(title_text="S&P 500 Index", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Combine strategies and plot results
    st.write('Generating strategy comparison plot...')
    combined_strategy = pd.DataFrame({
        'Custom Index': custom_index * 0.5,
        'S&P 500': sp500_data / sp500_data.iloc[0] * 0.5
    })
    combined_strategy['Total'] = combined_strategy.sum(axis=1)
    st.line_chart(combined_strategy)

    # Display performance metrics
    st.write('Performance Metrics:')
    total_return = (combined_strategy['Total'].iloc[-1] / combined_strategy['Total'].iloc[0] - 1) * 100
    st.write(f'Total Return: {total_return:.2f}%')
    custom_index_return = (combined_strategy['Custom Index'].iloc[-1] / combined_strategy['Custom Index'].iloc[0] - 1) * 100
    st.write(f'Custom Index Return: {custom_index_return:.2f}%')
    sp500_return = (combined_strategy['S&P 500'].iloc[-1] / combined_strategy['S&P 500'].iloc[0] - 1) * 100
    st.write(f'S&P 500 Return: {sp500_return:.2f}%')

    # Rebalancing suggestion
    st.write('Rebalancing Suggestion:')
    st.write('Consider rebalancing your portfolio every 30-90 days to maintain the 50/50 split between the custom index and S&P 500.')

if __name__ == "__main__":
    main()