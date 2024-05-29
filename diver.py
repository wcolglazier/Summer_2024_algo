import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

# Fetch the last month of 5-minute interval data for the assets
assets = {
    "Gold": "GC=F",
    "Bitcoin": "BTC-USD",
    "QQQ": "QQQ",
    "XLK": "XLK",
    "Nvidia": "NVDA"
}

dfs = {}
for asset, ticker in assets.items():
    data = yf.Ticker(ticker)
    dfs[asset] = data.history(period="1mo", interval="5m")

# Fetch the S&P 500 data for the same period
sp500 = yf.Ticker("^GSPC")
sp500_df = sp500.history(start=dfs["Gold"].index.min(), end=dfs["Gold"].index.max(), interval="5m")

# Ensure dataframes are aligned to the same timeframe
common_index = dfs["Gold"].index
for asset in assets:
    dfs[asset] = dfs[asset].reindex(common_index, method='nearest')

# Ensure the S&P 500 dataframe is aligned to the same timeframe as the other assets
sp500_df = sp500_df.reindex(common_index, method='nearest')

# EMA configuration
ema_short_length = 20
ema_long_length = 45

# Trade configuration
threshold = 0.07
stop_loss_percent = 1.5 * 1.5
take_profit_percent = 3.75 * 1.5

# Initial capital
initial_capital = 10000
capital_per_asset = initial_capital / len(assets)
current_capital = initial_capital

# Create empty trade list for each asset
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

# Portfolio value tracking for each timestamp
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
        trade_durations.append((row.name - trade['entry_time']).total_seconds() / 60)
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
        trade_durations.append((row.name - trade['entry_time']).total_seconds() / 60)
        profitable_trades += 1
        profitable_long_trades += 1
        open_trades_count -= 1  # Decrease open trades count

# Iterate through the dataframes to check for trade conditions
for asset, df in dfs.items():
    # Calculate EMAs
    df['EMA_Short'] = df['Close'].ewm(span=ema_short_length, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=ema_long_length, adjust=False).mean()

    for index, row in df.iterrows():
        # Check for new trade opportunities
        timestamp_30_min_earlier = row.name - pd.Timedelta(minutes=30)
        if timestamp_30_min_earlier in df.index:
            price_30_min_earlier = df.loc[timestamp_30_min_earlier]['Close']
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

# After iterating through all dataframes, update portfolio values at each timestamp
for timestamp in common_index:
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
buy_and_hold_value = {asset: df['Close'] / df['Close'].iloc[0] * capital_per_asset for asset, df in dfs.items()}

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
print(f"Average Profit from Long Trades: {profit_long/profitable_long_trades if profitable_long_trades else 0}")
print(f"Average Loss from Long Trades: {loss_long/losing_long_trades if losing_long_trades else 0}")
print(f"Initial Capital: ${initial_capital}")
print(f"Ending Capital: ${current_capital}")
print(f"Net Profit/Loss: ${current_capital - initial_capital}")
print(f"Return on Investment (ROI): {((current_capital - initial_capital) / initial_capital) * 100:.2f}%")
print(f"Average Trades at Once: {average_trades_at_once:.2f}")
print(f"Low Trades at Once: {low_trades_at_once}")
print(f"High Trades at Once: {high_trades_at_once}")
print(f"Average Percent of Portfolio in Each Trade: {average_percent_in_trade:.2f}%")
print(f"Low Percent of Portfolio in Each Trade: {low_percent_in_trade:.2f}%")
print(f"High Percent of Portfolio in Each Trade: {high_percent_in_trade:.2f}%")
print(f"Average Total Percent of Portfolio in Active Trades: {average_total_percent_in_trade:.2f}%")
print(f"Low Total Percent of Portfolio in Active Trades: {low_total_percent_in_trade:.2f}%")
print(f"High Total Percent of Portfolio in Active Trades: {high_total_percent_in_trade:.2f}%")

# Print all trades
print("\nAll Trades:")
for trade in trades:
    print(trade)

# Create a DataFrame for trades
trades_df = pd.DataFrame(trades)

# Plotting the data
plt.figure(figsize=(14, 7))
for asset, df in dfs.items():
    plt.plot(df['Close'], label=f'Close Price ({asset})', alpha=0.5)
    plt.plot(df['EMA_Short'], label=f'EMA {ema_short_length} ({asset})', alpha=0.75)
    plt.plot(df['EMA_Long'], label=f'EMA {ema_long_length} ({asset})', alpha=0.75)

# Mark trades on the plot
for trade in trades:
    if trade['status'] == 'closed':
        color = 'g' if trade['exit_price'] >= trade['entry_price'] else 'r'
        plt.plot(trade['entry_time'], trade['entry_price'], marker='o', color='b', markersize=8, alpha=0.75)
        plt.plot(trade['exit_time'], trade['exit_price'], marker='x', color=color, markersize=8, alpha=0.75)

plt.title('Trading Strategy for All Assets')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.savefig('trading_strategy_chart.png')
plt.close()

# Plot portfolio value, buy-and-hold strategy, and S&P 500 value
plt.figure(figsize=(14, 7))
plt.plot(common_index, portfolio_values, label='Portfolio Value', alpha=0.75)
for asset in assets:
    plt.plot(dfs[asset].index, buy_and_hold_value[asset], label=f'Buy & Hold {asset} Value', alpha=0.75, linestyle='--')
plt.plot(sp500_df.index, sp500_df['Close'] / sp500_df['Close'].iloc[0] * initial_capital, label='S&P 500 Value', alpha=0.75)
plt.title('Portfolio Value vs. Buy & Hold Strategies and S&P 500')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.savefig('portfolio_vs_sp500_buy_hold.png')
plt.close()

# Get the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save trades and summary to PDF
pdf = FPDF()

# Add a page
pdf.add_page()

# Set title
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, txt="Trading Strategy Report", ln=True, align='C')
pdf.set_font("Arial", 'B', 12)
pdf.cell(200, 10, txt=f"Generated on: {current_datetime}", ln=True, align='C')

# Add summary
pdf.set_font("Arial", size=11)
summary_lines = [
    f"Total Profit/Loss: {profit_loss}",
    f"Total Trades: {trade_count}",
    f"Profitable Trades: {profitable_trades}",
    f"Losing Trades: {losing_trades}",
    f"Average Trade Duration: {average_trade_duration} minutes",
    f"Profit from Trades: {profit_long}",
    f"Loss from Trades: {loss_long}",
    f"Average Profit from Trades: {profit_long/profitable_long_trades if profitable_long_trades else 0}",
    f"Average Loss from Trades: {loss_long/losing_long_trades if losing_long_trades else 0}",
    f"Initial Capital: ${initial_capital}",
    f"Ending Capital: ${current_capital}",
    f"Net Profit/Loss: ${current_capital - initial_capital}",
    f"Return on Investment (ROI): {((current_capital - initial_capital) / initial_capital) * 100:.2f}%",
    f"Average Trades at Once: {average_trades_at_once:.2f}",
    f"Low Trades at Once: {low_trades_at_once}",
    f"High Trades at Once: {high_trades_at_once}",
    f"Average Percent of Portfolio in Each Trade: {average_percent_in_trade:.2f}%",
    f"Low Percent of Portfolio in Each Trade: {low_percent_in_trade:.2f}%",
    f"High Percent of Portfolio in Each Trade: {high_percent_in_trade:.2f}%",
    f"Average Total Percent of Portfolio in Active Trades: {average_total_percent_in_trade:.2f}%",
    f"Low Total Percent of Portfolio in Active Trades: {low_total_percent_in_trade:.2f}%",
    f"High Total Percent of Portfolio in Active Trades: {high_total_percent_in_trade:.2f}%"
]

for line in summary_lines:
    pdf.cell(200, 10, txt=line, ln=True)

# Add chart
pdf.add_page()
pdf.image('trading_strategy_chart.png', x=10, y=8, w=190)

# Add portfolio value vs S&P 500 chart
pdf.add_page()
pdf.image('portfolio_vs_sp500_buy_hold.png', x=10, y=8, w=190)

# Add trades
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(200, 10, txt="All Trades", ln=True, align='C')
pdf.set_font("Arial", size=10)
for trade in trades:
    trade_line = f"Entry Time: {trade['entry_time']}, Entry Price: {trade['entry_price']}, " \
                 f"Status: {trade['status']}"
    if trade['status'] == 'closed':
        trade_line += f", Exit Time: {trade['exit_time']}, Exit Price: {trade['exit_price']}"
    pdf.cell(200, 10, txt=trade_line, ln=True)

# Add dataframe
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(200, 10, txt="Trade Data", ln=True, align='C')
pdf.set_font("Arial", size=8)

for i in range(len(trades_df)):
    row = trades_df.iloc[i]
    row_data = f"{row['entry_time']} {row['entry_price']} {row['status']}"
    if row['status'] == 'closed':
        row_data += f" {row['exit_time']} {row['exit_price']} {row['amount']}"
    pdf.cell(200, 10, txt=row_data, ln=True)

# Save the PDF
pdf.output("trading_strategy_report.pdf")
