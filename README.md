# Cherry Picker: Advanced ML-Enhanced Algorithmic Trading Strategy

Cherry Picker is a sophisticated algorithmic trading strategy that uses machine learning to select top-performing stocks from the S&P 500. This Streamlit-based application offers an interactive interface for users to explore and visualize the strategy's performance.

## Features

- Fetches real-time data for S&P 500 stocks
- Calculates technical and fundamental indicators
- Uses a Random Forest Regressor to predict short-term returns
- Selects top-performing stocks based on ML predictions
- Creates a custom index of selected stocks
- Compares performance against the S&P 500
- Provides LLM-generated explanations of stock selections (requires OpenAI API key)
- Interactive visualizations of trading history and strategy comparison

## Requirements

- Python 3.7+
- Streamlit
- yfinance
- pandas
- numpy
- scikit-learn
- OpenAI Python client
- plotly

## Installation

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Set your OpenAI API key as an environment variable: `export OPENAI_API_KEY='your-api-key-here'`

## Usage

Run the Streamlit app:
```bash
streamlit run cherry_picker.py
``` 
Then open your web browser and navigate to the provided local URL (typically http://localhost:8501).

## How it Works

1. **Data Collection**: Fetches 2 years of historical data for all S&P 500 stocks.
2. **Feature Engineering**: Calculates various technical and fundamental indicators.
3. **ML Model**: Trains a Random Forest Regressor to predict short-term returns based on these features.
4. **Stock Selection**: Predicts returns for all stocks and selects the top performers.
5. **Custom Index**: Creates an equally-weighted index of the selected stocks.
6. **Performance Comparison**: Compares the custom index against the S&P 500.

## Disclaimer

This is an experimental trading strategy. Use at your own risk. Past performance does not guarantee future results.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.

## License

[MIT](https://choosealicense.com/licenses/mit/)