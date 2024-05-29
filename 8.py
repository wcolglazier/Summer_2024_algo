import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries

# Replace 'your_alpha_vantage_api_key' with your actual Alpha Vantage API key
api_key = 'ZXF37CQ2GKLEPKKZ'
ts = TimeSeries(key=api_key, output_format='pandas')

def fetch_intraday_data(symbol, interval='5min', outputsize='full'):
    try:
        data, meta = ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        data.index = pd.to_datetime(data.index)
        return data
    except ValueError as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Fetch data for the chosen interval
interval = '5min'

gold_data = fetch_intraday_data('BTC-USD', interval)
if gold_data.empty:
    print("Failed to fetch Gold data. Exiting.")
    exit()

# Attempt to get data for the last year
gold_data = gold_data[gold_data.index >= (pd.to_datetime(datetime.now()) - pd.Timedelta(days=365))]

sp500_data = fetch_intraday_data('SPY', interval)
if sp500_data.empty:
    print("Failed to fetch S&P 500 data. Exiting.")
    exit()

sp500_data = sp500_data[(sp500_data.index >= gold_data.index.min()) & (sp500_data.index <= gold_data.index.max())]

# Ensure SP500 data is aligned with Gold data
sp500_data = sp500_data.reindex(gold_data.index, method='nearest')

# EMA configuration
ema_short_length = 20
ema_long_length = 45

# Trade configuration
threshold = 0.07
stop_loss_percent = .2
take_profit_percent = .35  # Actual percentage for take profit

# Initial capital
initial_capital = 10000
current_capital = initial_capital

# Calculate EMAs
gold_data['EMA_Short'] = gold_data['Close'].ewm(span=ema_short_length, adjust=False).mean()
gold_data['EMA_Long'] = gold_data['Close'].ewm(span=ema_long_length, adjust=False).mean()

# Trade tracking list
trades = []
profit_loss = 0.0
trade_count = 0
profitable_trades = 0
losing_trades = 0
trade_durations = []

# Additional tracking
long_trades = 0
profitable_long_trades = 0
losing_long_trades = 0
profit_long = 0.0
loss_long = 0.0

# Track the number of trades happening at once and percentage of portfolio in each trade
trades_happening_at_once = []
percent_in_trade = []
total_percent_in_trade = []

# Initialize open trades count
open_trades_count = 0

# Portfolio value tracking
portfolio_values = []

# Function to handle stop loss and take profit
def check_trade_conditions(trade, row):
    global profit_loss, profitable_trades, losing_trades, trade_durations
    global profitable_long_trades, losing_long_trades
    global profit_long, loss_long, current_capital, open_trades_count

    stop_loss = trade['entry_price'] * (1 - stop_loss_percent / 100)
    take_profit = trade['entry_price'] * (1 + take_profit_percent / 100)

    if row['Close'] <= stop_loss:
        loss = trade['amount'] * (trade['entry_price'] - stop_loss) / trade['entry_price']
        profit_loss -= loss
        loss_long += loss
        current_capital -= loss
        trade['status'] = 'closed'
        trade['exit_price'] = stop_loss
        trade['exit_time'] = row.name
        duration = (row.name - trade['entry_time']).total_seconds() / 60
        if duration > 0:
            trade_durations.append(duration)
        else:
            print(f"Invalid trade duration: Entry time: {trade['entry_time']}, Exit time: {row.name}")
        losing_trades += 1
        losing_long_trades += 1
        open_trades_count -= 1  # Decrease open trades count
    elif row['Close'] >= take_profit:
        profit = trade['amount'] * (take_profit - trade['entry_price']) / trade['entry_price']
        profit_loss += profit
        profit_long += profit
        current_capital += profit
        trade['status'] = 'closed'
        trade['exit_price'] = take_profit
        trade['exit_time'] = row.name
        duration = (row.name - trade['entry_time']).total_seconds() / 60
        if duration > 0:
            trade_durations.append(duration)
        else:
            print(f"Invalid trade duration: Entry time: {trade['entry_time']}, Exit time: {row.name}")
        profitable_trades += 1
        profitable_long_trades += 1
        open_trades_count -= 1  # Decrease open trades count

# Iterate through the dataframe to check for trade conditions
for index, row in gold_data.iterrows():
    # Check for new trade opportunities
    timestamp_30_min_earlier = row.name - pd.Timedelta(minutes=30)
    if timestamp_30_min_earlier in gold_data.index:
        price_30_min_earlier = gold_data.loc[timestamp_30_min_earlier]['Close']
        if row['EMA_Short'] > row['EMA_Long']:
            price_change_drop = (row['Close'] - price_30_min_earlier) / price_30_min_earlier * 100
            if price_change_drop <= -threshold:
                # Calculate the amount to invest based on available capital
                amount_to_invest = current_capital / (open_trades_count + 1)  # Ensuring equal distribution of capital
                trade_percent = (amount_to_invest / current_capital) * 100
                percent_in_trade.append(trade_percent)
                trades.append({
                    'entry_price': row['Close'],
                    'entry_time': row.name,
                    'amount': amount_to_invest,
                    'is_long': True,
                    'status': 'open'
                })
                trade_count += 1
                long_trades += 1
                open_trades_count += 1  # Increase open trades count

    # Update the status of each trade
    for trade in trades:
        if trade['status'] == 'open':
            check_trade_conditions(trade, row)

    # Track the number of trades happening at once
    trades_happening_at_once.append(open_trades_count)

    # Track total percent of portfolio in active trades
    if open_trades_count > 0:
        total_percent_in_trade.append(min(100, sum(percent_in_trade[-open_trades_count:])))

    # Track portfolio value over time
    portfolio_values.append(current_capital)

# Calculate average trade duration
average_trade_duration = np.mean(trade_durations) if trade_durations else 0

# Calculate average, low, and high number of trades happening at once
average_trades_at_once = np.mean(trades_happening_at_once)
low_trades_at_once = np.min(trades_happening_at_once)
high_trades_at_once = np.max(trades_happening_at_once)

# Calculate average, low, and high percentage of the portfolio in each trade
average_percent_in_trade = np.mean(percent_in_trade) if percent_in_trade else 0
low_percent_in_trade = np.min(percent_in_trade) if percent_in_trade else 0
high_percent_in_trade = np.max(percent_in_trade) if percent_in_trade else 0

# Calculate average, low, and high total percent of portfolio in active trades
average_total_percent_in_trade = np.mean(total_percent_in_trade) if total_percent_in_trade else 0
low_total_percent_in_trade = np.min(total_percent_in_trade) if total_percent_in_trade else 0
high_total_percent_in_trade = np.max(total_percent_in_trade) if total_percent_in_trade else 0

# Calculate buy and hold strategy for gold
buy_and_hold_value = gold_data['Close'] / gold_data['Close'].iloc[0] * initial_capital

# Print summary results
print("\nSummary Results:")
print(f"EMA Short: {ema_short_length}, EMA Long: {ema_long_length}")
print(f"Threshold: {threshold}%, Stop Loss: {stop_loss_percent}%, Take Profit: {take_profit_percent}%")
print(f"Total Profit/Loss: {profit_loss}")
print(f"Total Trades: {trade_count}")
print(f"Profitable Trades: {profitable_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Average Trade Duration: {average_trade_duration} minutes")
print(f"Long Trades: {long_trades}")
print(f"Profitable Long Trades: {profitable_long_trades}")
print(f"Losing Long Trades: {losing_long_trades}")
print(f"Profit from Long Trades: {profit_long}")
print(f"Loss from Long Trades: {loss_long}")
print(f"Initial Capital: ${initial_capital}")
print(f"Ending Capital: ${current_capital}")
print(f"Net Profit/Loss: ${current_capital - initial_capital}")
