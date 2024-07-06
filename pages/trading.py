import streamlit as st
import os
import dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError


# Load environment variables
dotenv.load_dotenv()

# # Set page title
# st.set_page_config(page_title="Alpaca Options Trading")



header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    st.page_link("main.py", label="Home", icon="🏠")
with header_col2:
    st.page_link("pages/trading.py", label="Trading", icon="📈")
st.divider()


# Add a title and description
st.title("Alpaca Options Trading")
st.write("This page allows you to trade options using Alpaca API.")

# Checkbox to toggle between paper trading and live trading
trading_mode = st.checkbox("Use Live Trading", value=False)

# Update API keys and base URL based on trading mode
if trading_mode:
    alpaca_api_key_id = os.getenv('ALPACA_LIVE_KEY_ID')
    alpaca_api_secret_key = os.getenv('ALPACA_LIVE_SECRET')
    paper = False
    base_url = "https://api.alpaca.markets"
else:
    alpaca_api_key_id = os.getenv('ALPACA_PAPER_KEY_API_ID')
    alpaca_api_secret_key = os.getenv('ALPACA_PAPER_API_SECRET')
    paper = True
    base_url = "https://paper-api.alpaca.markets"

# Initialize the trading client
try:
    st.write("Initializing trading client...")
    st.write("API Key ID:", alpaca_api_key_id)
    st.write("API Secret Key:", alpaca_api_secret_key)
    trading_client = TradingClient(api_key=alpaca_api_key_id, secret_key= alpaca_api_secret_key, paper=paper, url_override=base_url)
except APIError as e:
    st.error(f"Error initializing trading client: {e}")
    st.stop()

# # Function to get account information
# def get_account_info():
#     try:
#         return trading_client.get_account()
#     except APIError as e:
#         st.error(f"Error retrieving account information: {e}")
#         return None

# Function to place a market order
def place_market_order(symbol, qty, side, time_in_force):
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=time_in_force
    )
    try:
        return trading_client.submit_order(order_data=order_data)
    except APIError as e:
        st.error(f"Error placing market order: {e}")
        return None

# Get account information and display it
account_info = trading_client.get_account()
if account_info:
    st.subheader("Account Information")
    st.json(account_info)

    # Display account balance
    st.subheader("Account Balance")
    cash_balance = account_info.cash
    portfolio_value = account_info.portfolio_value
    st.write(f"**Cash Balance:** ${cash_balance}")
    st.write(f"**Portfolio Value:** ${portfolio_value}")

# Order form
st.subheader("Place a Market Order")
with st.form("market_order_form"):
    symbol = st.text_input("Symbol", "AAPL")
    qty = st.number_input("Quantity", min_value=0.001, value=1.0)
    side = st.selectbox("Order Side", [OrderSide.BUY, OrderSide.SELL])
    time_in_force = st.selectbox("Time in Force", [TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK])
    submit_button = st.form_submit_button(label="Submit Market Order")

    if submit_button:
        order_response = place_market_order(symbol, qty, side, time_in_force)
        if order_response:
            st.subheader("Order Response")
            st.json(order_response)

# Function to place a limit order
def place_limit_order(symbol, qty, side, time_in_force, limit_price):
    order_data = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=time_in_force,
        limit_price=limit_price
    )
    try:
        return trading_client.submit_order(order_data=order_data)
    except APIError as e:
        st.error(f"Error placing limit order: {e}")
        return None

# Order form for limit orders
st.subheader("Place a Limit Order")
with st.form("limit_order_form"):
    symbol = st.text_input("Limit Order Symbol", "AAPL")
    qty = st.number_input("Limit Order Quantity", min_value=0.001, value=1.0)
    side = st.selectbox("Limit Order Side", [OrderSide.BUY, OrderSide.SELL])
    time_in_force = st.selectbox("Limit Order Time in Force", [TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK])
    limit_price = st.number_input("Limit Price", min_value=0.01, value=100.0)
    submit_button = st.form_submit_button(label="Submit Limit Order")

    if submit_button:
        order_response = place_limit_order(symbol, qty, side, time_in_force, limit_price)
        if order_response:
            st.subheader("Limit Order Response")
            st.json(order_response)
