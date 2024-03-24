import os
import json
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from openai import OpenAI
import openai

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period="1y").iloc[-1].Close)

def calculate_SMA(ticker, window):
    df = yf.Ticker(ticker).history(period="1y").Close
    return str(df.rolling(window=window).mean().iloc[-1])

def calculate_EMA(ticker, window):
    df = yf.Ticker(ticker).history(period="1y").Close
    return str(df.ewm(window=window, adjust=False).mean().iloc[-1])

def calculate_RSI(ticker, window):
    df = yf.Ticker(ticker).history(period="1y").Close
    delta = df.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    return str(100 - (100 / (1 + RS)).iloc[-1])

def calculate_MACD(ticker):
    df = yf.Ticker(ticker).history(period="1y").Close
    shortEMA = df.ewm(span=12, adjust=False).mean()
    longEMA = df.ewm(span=26, adjust=False).mean()
    MACD = shortEMA - longEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    return f"{MACD.iloc[-1]},{signal.iloc[-1]}, {MACD.iloc[-1] - signal.iloc[-1]}"

def plot_stock_price(ticker):
    df = yf.Ticker(ticker).history(period="1y").Close
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df.values, label="Stock Price")
    plt.title(f"{ticker} Stock Price past year")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.savefig("stock_price.png")
    plt.close()

functions = [
    {
        "name": "get_stock_price",
        "description": "Get the latest stock price given the ticker symbol of a company",
        "parameters": {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker of the company (e.g. AAPL for Apple Inc.)'
                }
            },
            'required': ['ticker']
        },
    },
    {
        "name": "calculate_SMA",
        "description": "Calculate the Simple Moving Average of a stock given the ticker symbol and window size",
        "parameters": {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker of the company (e.g. AAPL for Apple Inc.)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The timeframe to consider while calculating the Simple Moving Average'
                }
            },
            'required': ['ticker', 'window']
        },
    },
    {
        "name": "calculate_EMA",
        "description": "Calculate the Exponential Moving Average of a stock given the ticker symbol and window size",
        "parameters": {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker of the company (e.g. AAPL for Apple Inc.)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size for the Exponential Moving Average'
                }
            },
            'required': ['ticker', 'window']
        },
    },
    {
        "name": "calculate_RSI",
        "description": "Calculate the Relative Strength Index of a stock given the ticker symbol and window size",
        "parameters": {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker of the company (e.g. AAPL for Apple Inc.)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size for the Relative Strength Index'
                }
            },
            'required': ['ticker', 'window']
        },
    },
    {
        "name": "calculate_MACD",
        "description": "Calculate the Moving Average Convergence Divergence of a stock given the ticker symbol",
        "parameters": {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker of the company'
                }
            },
            'required': ['ticker']
        },
    },
    {
        "name": "plot_stock_price",
        "description": "Plot the stock price of a company given the ticker symbol",
        "parameters": {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker of the company (e.g. AAPL for Apple Inc.)'
                }
            },
            'required': ['ticker']
        },
    }
]

available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price
}

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("Stock Market Analysis")

user_input = st.text_input("Your input:")

if user_input:
    try:
        st.session_state['messages'].append({"role": "user", "content": f'{user_input}'})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['messages'],
            functions=functions,
            function_call='auto'
        )

        response_message = response.choices[0].message

        if response_message.function_call:
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)
            if function_name in ['get_stock_price', 'calculate_MACD', 'plot_stock_price']:
                args_dict = {'ticker': function_args.get('ticker')}
            elif function_name in ['calculate_SMA', 'calculate_EMA', 'calculate_RSI']:
                args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}
            
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**args_dict)

            if function_name == 'plot_stock_price':
                st.image("stock_price.png")
            else:
                st.session_state['messages'].append(response_message)
                st.session_state['messages'].append({"role": "function", "name": function_name, "content": f'{function_response}'})
            
            second_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state['messages'],
            )
            st.text(second_response.choices[0].message.content)
            st.session_state['messages'].append({'role':'assistant','content':second_response.choices[0].message.content})
        else:
            st.text(response_message.content)
            st.session_state['messages'].append({'role':'assistant','content':response_message.content})
            
    except Exception as e:
        raise e
