import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

# Define the list of stocks to analyze
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V"]

# Parameters for the strategy
ema_short_length = 20
ema_long_length = 45
threshold = 0.07
stop_loss_percent = 1.5
take_profit_percent = 3.75
initial_capital = 10000

# Initialize a list to store results
results = []

# Function to execute the trading strategy
def execute_strategy(ticker):
    # Fetch the last month of 5-minute interval data for the stock
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo", interval="5m")

    # Fetch the S&P 500 data for the same period
    sp500 = yf.Ticker("^GSPC")
    sp500_df = sp500.history(start=df.index.min(), end=df.index.max(), interval="5m")

    # Ensure SP500 data is aligned with stock data
    sp500_df = sp500_df.reindex(df.index, method='nearest')

    # Initialize variables
    current_capital = initial_capital
    trades = []
    profit_loss = 0.0
    trade_count = 0
    profitable_trades = 0
    losing_trades = 0
    trade_durations = []
    long_trades = 0
    profitable_long_trades = 0
    losing_long_trades = 0
    profit_long = 0.0
    loss_long = 0.0
    trades_happening_at_once = []
    percent_in_trade = []
    total_percent_in_trade = []
    open_trades_count = 0
    portfolio_values = []

    # Calculate EMAs
    df['EMA_Short'] = df['Close'].ewm(span=ema_short_length, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=ema_long_length, adjust=False).mean()

    # Function to handle stop loss and take profit
    def check_trade_conditions(trade, row, profit_loss, profitable_trades, losing_trades, trade_durations,
                               profitable_long_trades, losing_long_trades, profit_long, loss_long, current_capital, open_trades_count):
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
            open_trades_count -= 1
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
            open_trades_count -= 1

        return profit_loss, profitable_trades, losing_trades, trade_durations, profitable_long_trades, losing_long_trades, profit_long, loss_long, current_capital, open_trades_count

    # Iterate through the dataframe to check for trade conditions
    for index, row in df.iterrows():
        timestamp_30_min_earlier = row.name - pd.Timedelta(minutes=30)
        if timestamp_30_min_earlier in df.index:
            price_30_min_earlier = df.loc[timestamp_30_min_earlier]['Close']
            if row['EMA_Short'] > row['EMA_Long']:
                price_change_drop = (row['Close'] - price_30_min_earlier) / price_30_min_earlier * 100
                if price_change_drop <= -threshold:
                    amount_to_invest = current_capital / (open_trades_count + 1)
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
                    open_trades_count += 1

        for trade in trades:
            if trade['status'] == 'open':
                profit_loss, profitable_trades, losing_trades, trade_durations, profitable_long_trades, losing_long_trades, profit_long, loss_long, current_capital, open_trades_count = check_trade_conditions(
                    trade, row, profit_loss, profitable_trades, losing_trades, trade_durations, profitable_long_trades, losing_long_trades, profit_long, loss_long, current_capital, open_trades_count)

        trades_happening_at_once.append(open_trades_count)
        if open_trades_count > 0:
            total_percent_in_trade.append(min(100, sum(percent_in_trade[-open_trades_count:])))
        portfolio_values.append(current_capital)

    average_trade_duration = np.mean(trade_durations) if trade_durations else 0
    average_trades_at_once = np.mean(trades_happening_at_once)
    low_trades_at_once = np.min(trades_happening_at_once)
    high_trades_at_once = np.max(trades_happening_at_once)
    average_percent_in_trade = np.mean(percent_in_trade) if percent_in_trade else 0
    low_percent_in_trade = np.min(percent_in_trade) if percent_in_trade else 0
    high_percent_in_trade = np.max(percent_in_trade) if percent_in_trade else 0
    average_total_percent_in_trade = np.mean(total_percent_in_trade) if total_percent_in_trade else 0
    low_total_percent_in_trade = np.min(total_percent_in_trade) if total_percent_in_trade else 0
    high_total_percent_in_trade = np.max(total_percent_in_trade) if total_percent_in_trade else 0

    buy_and_hold_value = df['Close'] / df['Close'].iloc[0] * initial_capital

    # Store results
    results.append({
        'Ticker': ticker,
        'Total Profit/Loss': profit_loss,
        'Total Trades': trade_count,
        'Profitable Trades': profitable_trades,
        'Losing Trades': losing_trades,
        'Average Trade Duration': average_trade_duration,
        'Long Trades': long_trades,
        'Profitable Long Trades': profitable_long_trades,
        'Losing Long Trades': losing_long_trades,
        'Profit from Long Trades': profit_long,
        'Loss from Long Trades': loss_long,
        'Initial Capital': initial_capital,
        'Ending Capital': current_capital,
        'Net Profit/Loss': current_capital - initial_capital,
        'ROI': ((current_capital - initial_capital) / initial_capital) * 100,
        'Average Trades at Once': average_trades_at_once,
        'Low Trades at Once': low_trades_at_once,
        'High Trades at Once': high_trades_at_once,
        'Average Percent of Portfolio in Each Trade': average_percent_in_trade,
        'Low Percent of Portfolio in Each Trade': low_percent_in_trade,
        'High Percent of Portfolio in Each Trade': high_percent_in_trade,
        'Average Total Percent of Portfolio in Active Trades': average_total_percent_in_trade,
        'Low Total Percent of Portfolio in Active Trades': low_total_percent_in_trade,
        'High Total Percent of Portfolio in Active Trades': high_total_percent_in_trade,
        'Buy and Hold Ending Value': buy_and_hold_value.iloc[-1],
        'Buy and Hold ROI': ((buy_and_hold_value.iloc[-1] - initial_capital) / initial_capital) * 100,
        'SP500 Ending Value': sp500_df['Close'].iloc[-1] / sp500_df['Close'].iloc[0] * initial_capital,
        'SP500 ROI': ((sp500_df['Close'].iloc[-1] / sp500_df['Close'].iloc[0] * initial_capital - initial_capital) / initial_capital) * 100
    })

    # Plotting the data
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Close Price', alpha=0.5)
    plt.plot(df['EMA_Short'], label=f'EMA {ema_short_length}', alpha=0.75)
    plt.plot(df['EMA_Long'], label=f'EMA {ema_long_length}', alpha=0.75)

    for trade in trades:
        if trade['status'] == 'closed':
            color = 'g' if trade['exit_price'] >= trade['entry_price'] else 'r'
            plt.plot(trade['entry_time'], trade['entry_price'], marker='o', color='b', markersize=8, alpha=0.75)
            plt.plot(trade['exit_time'], trade['exit_price'], marker='x', color=color, markersize=8, alpha=0.75)

    plt.title(f'{ticker} Trading Strategy')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig(f'{ticker}_trading_strategy_chart.png')
    plt.close()

    # Plot portfolio value, buy-and-hold strategy, and S&P 500 value
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, portfolio_values, label='Portfolio Value', alpha=0.75)
    plt.plot(df.index, buy_and_hold_value, label='Buy & Hold Stock Value', alpha=0.75, linestyle='--')
    plt.plot(sp500_df.index, sp500_df['Close'] / sp500_df['Close'].iloc[0] * initial_capital, label='S&P 500 Value', alpha=0.75)
    plt.title(f'Portfolio Value vs. Buy & Hold {ticker} and S&P 500')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig(f'{ticker}_portfolio_vs_sp500_buy_hold.png')
    plt.close()

# Execute the strategy for each stock
for stock in stocks:
    execute_strategy(stock)

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

# Display results
import ace_tools as tools; tools.display_dataframe_to_user(name="Trading Strategy Results", dataframe=results_df)

# Get the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save summary and charts to PDF
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

# Add charts
for stock in stocks:
    pdf.add_page()
    pdf.image(f'{stock}_trading_strategy_chart.png', x=10, y=8, w=190)
    pdf.add_page()
    pdf.image(f'{stock}_portfolio_vs_sp500_buy_hold.png', x=10, y=8, w=190)

# Save the PDF
pdf.output("trading_strategy_report.pdf")
