import os
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Type, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables (make sure you have a .env file with OPENAI_API_KEY set)
load_dotenv()

# Optional: Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ChatOpenAI agent with the latest model
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    openai_api_key=openai_api_key  # Use the loaded API key
)

import json
import os
from sklearn.preprocessing import MinMaxScaler


app = FastAPI(title="Stock Price Checker API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

today = datetime.now().date()

llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
chat_history=[]

def get_stock_price(symbol: str) -> float:
    """Fetch the latest stock price for the given symbol using yfinance."""
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    if not todays_data.empty:
        return round(todays_data['Close'].iloc[-1], 2)
    else:
        raise ValueError("Could not retrieve data for the given symbol.")

def get_price_change_percent(symbol, days_ago):
    ticker = yf.Ticker(symbol)

    # Get today's date
    end_date = datetime.now()

    # Get the date N days ago
    start_date = end_date - timedelta(days=days_ago)

    # Convert dates to string format that yfinance can accept
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # Get the historical data
    historical_data = ticker.history(start=start_date, end=end_date)

    # Get the closing price N days ago and today's closing price
    old_price = historical_data['Close'].iloc[0]
    new_price = historical_data['Close'].iloc[-1]

    # Calculate the percentage change
    percent_change = ((new_price - old_price) / old_price) * 100

    return round(percent_change, 2)

def calculate_performance(symbol, days_ago):
    # Initialize a ticker object for the given stock symbol
    ticker = yf.Ticker(symbol)

    # Calculate the current date as the end date
    end_date = datetime.now()

    # Calculate the start date by subtracting the specified number of days from the end date
    start_date = end_date - timedelta(days=days_ago)

    # Format the start and end dates as strings in 'YYYY-MM-DD' format for the API call
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # Retrieve historical stock data for the specified date range
    historical_data = ticker.history(start=start_date, end=end_date)

    # Extract the closing price of the stock on the start date
    old_price = historical_data['Close'].iloc[0]

    # Extract the closing price of the stock on the most recent date
    new_price = historical_data['Close'].iloc[-1]

    # Calculate the percentage change in price from the start date to the end date
    percent_change = ((new_price - old_price) / old_price) * 100

    # Return the percentage change, rounded to 2 decimal places
    return round(percent_change, 2)

def get_best_performing(stocks, days_ago):
    # Initialize variables for tracking the best performing stock and its performance
    best_stock = None
    best_performance = None

    # Loop through each stock in the provided list
    for stock in stocks:
        try:
            # Calculate the stock's performance over the given time frame
            performance = calculate_performance(stock, days_ago)

            # Update best performing stock and its performance if necessary
            if best_performance is None or performance > best_performance:
                best_stock = stock
                best_performance = performance
        except Exception as e:
            # Handle any errors encountered during calculation
            print(f"Could not calculate performance for {stock}: {e}")
    
    # Return the best performing stock and its performance
    return best_stock, best_performance

def get_old_price(symbol, year, month, day):
    # Initialize a ticker object for the given stock symbol
    ticker = yf.Ticker(symbol)

    # Format the start date string using the provided year, month, and day
    start_date = '{}-{}-{}'.format(year, month, day)

    # Calculate the end date by adding one month to the start date for the data retrieval range
    end_date = '{}-{}-{}'.format(year, month + 1, day)

    # Retrieve historical stock data between the start and end dates
    historical_data = ticker.history(start=start_date, end=end_date)

    # Extract the closing price on the first day available in the historical data
    price = historical_data['Close'].iloc[0]

    # Return the price, rounded to 2 decimal places
    return round(price, 2)

def download_and_prepare_data(stockticker, period_days):
    ticker = yf.Ticker(stockticker)
    df = ticker.history(period=f"{period_days}d")
    df = df[['Close']]
    return df

def create_dataset(data, time_steps):
    dataset = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i + time_steps])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

import json
from datetime import datetime, timedelta


def predict_price_dynamic(stockticker, predictionDays, moving_average_days):
    # Calculate the future date
    data = yf.download(stockticker, period=f"{predictionDays}d", interval="1d")
    closing_prices = data["Close"]

    current_date = datetime.now()
    future_date = current_date + timedelta(days=predictionDays)
    target_date = future_date.strftime('%Y-%m-%d')

    # Mapping stock tickers to their corresponding JSON prediction files
    file_mapping = {
        "BTC-USD": 'static/predictionsBTC.json',
        "ETH-USD": 'static/predictionsETH.json',
        "DOGE-USD": 'static/predictionsDOGE.json',
        "META": 'static/predictionsMETA.json',
        "AAPL": 'static/predictionsAAPL.json',
        "AMZN": 'static/predictionsAMZN.json',
        "^GSPC": 'static/predictionsSP500.json'
    }

    # Get the path for the prediction file based on the stock ticker
    file_path = file_mapping.get(stockticker)
    if file_path:
        # Open and load the prediction data from JSON file
        with open(file_path, 'r') as json_file:
            predictions = json.load(json_file)
        
        # Check if the target date's prediction is available
        if target_date in predictions:
            predicted_price = predictions[target_date]
            if closing_prices.iloc[-1] < predicted_price:
                trend = "upward"
            else:
                trend = "downward"
            return (predicted_price, trend, "Data available for the target date.")
        else:
            # Find the latest available prediction before the target date
            closest_date = max((date for date in predictions if date < target_date), default=None)
            if closest_date:
                predicted_price = predictions[closest_date]
                return (predicted_price, "unknown", f"No prediction for {target_date}. Last available prediction is for {closest_date}.")
            else:
                # Return None if no predictions are available before the target date
                return (None, "unknown", "No available predictions before the target date.")
    else:
        
        if(moving_average_days < 10):
            long_term_rate = 1
        else:
            long_term_rate = moving_average_days / 20
        # Get closing prices
        closing_prices = data["Close"]

        # Calculate moving average
        moving_average = closing_prices.rolling(window=moving_average_days).mean()

        # Calculate standard deviation (volatility) based on closing prices
        std_deviation = np.std(closing_prices.pct_change())

        # Determine direction and adjust future price prediction dynamically based on volatility
        if closing_prices.iloc[-1] > moving_average.iloc[-1]:
            direction = "upward"
            prediction_percentage = (1 + std_deviation)*long_term_rate  # Using standard deviation to adjust prediction
        else:
            direction = "downward"
            prediction_percentage = (1 - std_deviation)*long_term_rate

        # Predict future price based on direction and volatility
        predicted_price = closing_prices.iloc[-1] * prediction_percentage

        return predicted_price, direction, prediction_percentage - 1







def calculate_investment_for_profit(ticker, profit_target, moving_average_days=7, prediction_period=60):
    predicted_price, direction, info = predict_price_dynamic(ticker, prediction_period, moving_average_days)
    if direction == "downward":
        return "Investing now is predicted to result in a loss, not a profit."
    if predicted_price is None:
        return info

    # Fetch current price
    current_data = yf.download(ticker, period="1d", interval="1d")
    current_price = current_data["Close"].iloc[-1]

    # Calculate the required investment to achieve the target profit
    if predicted_price > current_price:
        profit_per_share = predicted_price - current_price
        number_of_shares = profit_target / profit_per_share
        required_investment = number_of_shares * current_price
    else:
        return "Current predictions do not suggest a price increase sufficient for profit."

    return f"To make a profit of ${profit_target}, you need to invest approximately ${required_investment:.2f} now."

from typing import List

class QueryInput(BaseModel):
    query: str
    
class StockPriceCheckInput(BaseModel):
    """Input for Stock price check."""

    stockticker: str = Field(..., description="Ticker symbol for stock or index")

class StockPriceTool(BaseTool):
    name = "get_stock_ticker_price"
    description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"

    def _run(self, stockticker: str):
        price_response = get_stock_price(stockticker)

        return price_response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput

class CalculateInvestmentForProfit(BaseModel):
    "Input for stockticker, number of days for moving average and the historical period for calculation"
    stockticker: str = Field(...,description="Ticker symbol for stock or index")
    profit_target: int = Field(..., description="The profit target amount")
    moving_average_days: int = Field(...,description="Number of days for moving average")
    prediction_period: int = Field(...,description="days for calculation")

class CalculateInvestmentForProfit(BaseTool):
    name="calculate_investment_for_profit"
    description="The calculate_investment_for_profit function is designed to help investors understand the amount of capital required to achieve a specific profit target from investing in cryptocurrencies or stocks. Leveraging the power of the predict_price_dynamic function, it calculates the necessary investment to meet a desired profit, considering the asset's future price movement. To use this function, you need to specify the ticker symbol of the financial instrument as recognized by the Yahoo Finance API (yfinance), your profit target in dollars, and optionally, the number of days for the moving average calculation and the historical period for data analysis. DO NOT response with LaTEX format. Do not write any equation"

    def _run(self, stockticker: str, profit_target: int, moving_average_days: int, prediction_period: int):
      prediction = calculate_investment_for_profit(stockticker, profit_target, moving_average_days, prediction_period)
      return prediction

    def _arun(self, stockticker:str):
      raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = CalculateInvestmentForProfit


class StockPricePredictInput(BaseModel):
    stockticker: str = Field(...,description="Ticker symbol for stock or index")
    predictionDay: int = Field(...,description="Historical period for calculation")
    moving_average_days: int = Field(...,description="Number of days for moving average")


class StockPricePredictionTool(BaseTool):
    name="predict_price_dynamic"
    description="a useful tool when it is necessary to accurately predict the future price of a stock, or to calculate how much a user's money can increase over a given period of time. you should input the stock ticker used on the yfinance API. Todays date is {}".format(today)

    def _run(self, stockticker: str, predictionDay: int, moving_average_days: int):
      prediction = predict_price_dynamic(stockticker, predictionDay, moving_average_days)
      return prediction

    def _arun(self, stockticker:str):
      raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPricePredictInput

class StockDateInput(BaseModel):
    "Input for Stock ticker and date check"
    stockticker: str = Field(...,description="Ticker symbol for stock or index")
    year: int = Field(...,description="Year of the price input")
    month: int = Field(...,description="Month of the price input")
    day: int = Field(...,description="Day of the price input")

class StockDateTool(BaseTool):
    name = "get_old_price"
    description = "a handy tool when you need to find the price on a specific date. You should input the stock ticker used on the yfinance API and give the date in year, month and day."

    def _run(self, stockticker: str, year: int, month: int, day: int):
      price_response = get_old_price(stockticker, year, month, day)
      return price_response

    def _arun(self, stockticker: str, days_ago: int):
      raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockDateInput

class StockChangePercentageCheckInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stockticker: str = Field(..., description="Ticker symbol for stock or index")
    days_ago: int = Field(..., description="Int number of days to look back")

class StockPercentageChangeTool(BaseTool):
    name = "get_price_change_percent"
    description = "Useful for when you need to find out the percentage change in a stock's value. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stockticker: str, days_ago: int):
        price_change_response = get_price_change_percent(stockticker, days_ago)

        return price_change_response

    def _arun(self, stockticker: str, days_ago: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockChangePercentageCheckInput


class StockBestPerformingInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stocktickers: List[str] = Field(..., description="Ticker symbols for stocks or indices")
    days_ago: int = Field(..., description="Int number of days to look back")

class StockGetBestPerformingTool(BaseTool):
    name = "get_best_performing"
    description = "Useful for when you need to the performance of multiple stocks over a period. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stocktickers: List[str], days_ago: int):
        price_change_response = get_best_performing(stocktickers, days_ago)

        return price_change_response

    def _arun(self, stockticker: List[str], days_ago: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput

tools = [StockPriceTool(),StockPercentageChangeTool(), StockGetBestPerformingTool(), StockDateTool(), StockPricePredictionTool(), CalculateInvestmentForProfit()]


def update_chat_history(speaker, message):
  chat_history.append({"speaker": speaker, "message": message})

def format_chat_history():
    # Extract the last three entries from the chat history
    last_three_messages = chat_history[-7:]
    
    # Format the history by joining speaker and message into a single string for each entry
    formatted_history = ", ".join(
        [f'{entry["speaker"]}: "{entry["message"]}"' for entry in last_three_messages]
    )
    
    # Return the formatted conversation history, enclosed in braces to indicate aggregation
    return f"the conversation so far: {{{formatted_history}}}"


def ask_agent(user_input):
    """
    Receive user input, add it to the chat history, and get a response from the AI.
    
    :param user_input: Input received from the user
    """
    # Add user input to the chat history
    update_chat_history("user", user_input)
    
    # Format the chat history
    formatted_history = format_chat_history()
    
    # Add the last user message to the formatted chat history
    full_input = f"{formatted_history}, last user message: {user_input}"
    
    # Get the ai response
    ai_response = open_ai_agent.run(full_input)
    
    # Add AI response to the chat history
    update_chat_history("ai", ai_response)
    
    return ai_response

open_ai_agent = initialize_agent(tools,
                        llm,
                        agent=AgentType.OPENAI_FUNCTIONS,
                        verbose=True)
@app.post("/query/")
async def perform_query(query_input: QueryInput):
    try:
        response = ask_agent(query_input.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
