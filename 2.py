import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from fpdf import FPDF
import yfinance as yf

# Fetch the last month of 5-minute interval data for Gold futures
gold = yf.Ticker("GC=F")
df = gold.history(period="1mo", interval="5m")
df.index = df.index.tz_localize(None)  # Remove timezone information
df.to_csv("gold_data.csv", index_label="datetime")  # Save to CSV for backtrader

# Fetch the S&P 500 data for the same period
sp500 = yf.Ticker("^GSPC")
sp500_df = sp500.history(start=df.index.min(), end=df.index.max(), interval="5m")
sp500_df.index = sp500_df.index.tz_localize(None)  # Remove timezone information
sp500_df.to_csv("sp500_data.csv", index_label="datetime")  # Save to CSV for backtrader

# Define the strategy
class EMAStrategy(bt.Strategy):
    params = (
        ('ema_short_length', 20),
        ('ema_long_length', 45),
        ('threshold', 0.07),
        ('stop_loss_percent', 1.5),
        ('take_profit_percent', 3.75),
    )

    def __init__(self):
        self.ema_short = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.ema_short_length)
        self.ema_long = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.ema_long_length)
        self.order = None
        self.buy_price = None
        self.stop_loss = None
        self.take_profit = None

        # Tracking variables
        self.trade_count = 0
        self.profitable_trades = 0
        self.losing_trades = 0
        self.trade_durations = []
        self.profit_long = 0.0
        self.loss_long = 0.0
        self.trades_happening_at_once = []
        self.percent_in_trade = []
        self.total_percent_in_trade = []
        self.open_trades_count = 0
        self.portfolio_values = []

    def next(self):
        if self.order:
            return

        if self.ema_short > self.ema_long:
            if len(self.data) >= 30:  # Ensure there is enough data
                price_30_min_earlier = self.data.close[-30]  # Get the close price 30 minutes earlier
                price_change_drop = (self.data.close[0] - price_30_min_earlier) / price_30_min_earlier * 100

                if price_change_drop <= -self.params.threshold:
                    self.order = self.buy()
                    self.log(f'BUY CREATE {self.data.close[0]:.2f}, Change: {price_change_drop:.2f}%')

        # Check for stop loss or take profit conditions
        if self.position:
            if self.data.close[0] <= self.stop_loss:
                self.order = self.sell()
                self.log(f'SELL CREATE {self.data.close[0]:.2f} due to STOP LOSS')
            elif self.data.close[0] >= self.take_profit:
                self.order = self.sell()
                self.log(f'SELL CREATE {self.data.close[0]:.2f} due to TAKE PROFIT')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
                self.buy_price = order.executed.price
                self.stop_loss = self.buy_price * (1 - self.params.stop_loss_percent / 100)
                self.take_profit = self.buy_price * (1 + self.params.take_profit_percent / 100)
                self.trade_count += 1
                self.open_trades_count += 1
                self.percent_in_trade.append(self.broker.get_cash() / (self.broker.get_value() * self.open_trades_count) * 100)
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
                self.open_trades_count -= 1
                duration = (self.data.datetime.date(0) - order.created.dt).total_seconds() / 60
                self.trade_durations.append(duration)
                if order.executed.price > self.buy_price:
                    self.profitable_trades += 1
                    self.profit_long += (order.executed.price - self.buy_price)
                else:
                    self.losing_trades += 1
                    self.loss_long += (self.buy_price - order.executed.price)
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
            self.portfolio_values.append(self.broker.get_value())

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

# Initialize cerebro
cerebro = bt.Cerebro()

# Add strategy
cerebro.addstrategy(EMAStrategy)

# Load data
data = bt.feeds.GenericCSVData(
    dataname='gold_data.csv',
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=-1,
    dtformat=('%Y-%m-%d %H:%M:%S'),  # Specify the correct datetime format
    timeframe=bt.TimeFrame.Minutes,
    compression=5,
)

cerebro.adddata(data)

# Set initial cash
initial_capital = 10000
cerebro.broker.set_cash(initial_capital)

# Run the strategy
results = cerebro.run()
strat = results[0]

# Summary statistics
current_capital = cerebro.broker.get_value()
profit_loss = current_capital - initial_capital
average_trade_duration = sum(strat.trade_durations) / len(strat.trade_durations) if strat.trade_durations else 0
average_trades_at_once = sum(strat.trades_happening_at_once) / len(strat.trades_happening_at_once) if strat.trades_happening_at_once else 0
low_trades_at_once = min(strat.trades_happening_at_once) if strat.trades_happening_at_once else 0
high_trades_at_once = max(strat.trades_happening_at_once) if strat.trades_happening_at_once else 0
average_percent_in_trade = sum(strat.percent_in_trade) / len(strat.percent_in_trade) if strat.percent_in_trade else 0
low_percent_in_trade = min(strat.percent_in_trade) if strat.percent_in_trade else 0
high_percent_in_trade = max(strat.percent_in_trade) if strat.percent_in_trade else 0
average_total_percent_in_trade = sum(strat.total_percent_in_trade) / len(strat.total_percent_in_trade) if strat.total_percent_in_trade else 0
low_total_percent_in_trade = min(strat.total_percent_in_trade) if strat.total_percent_in_trade else 0
high_total_percent_in_trade = max(strat.total_percent_in_trade) if strat.total_percent_in_trade else 0

# Print summary results
print("\nSummary Results:")
print(f"Total Profit/Loss: {profit_loss}")
print(f"Total Trades: {strat.trade_count}")
print(f"Profitable Trades: {strat.profitable_trades}")
print(f"Losing Trades: {strat.losing_trades}")
print(f"Average Trade Duration: {average_trade_duration} minutes")
print(f"Profit from Trades: {strat.profit_long}")
print(f"Loss from Trades: {strat.loss_long}")
print(f"Average Profit from Trades: {strat.profit_long/strat.profitable_trades if strat.profitable_trades else 0}")
print(f"Average Loss from Trades: {strat.loss_long/strat.losing_trades if strat.losing_trades else 0}")
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

# Generate report
# Create a PDF with FPDF
pdf = FPDF()

# Add a page
pdf.add_page()

# Set title
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, txt="Trading Strategy Report", ln=True, align='C')
pdf.set_font("Arial", 'B', 12)
pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')

# Add summary
pdf.set_font("Arial", size=11)
summary_lines = [
    f"Total Profit/Loss: {profit_loss}",
    f"Total Trades: {strat.trade_count}",
    f"Profitable Trades: {strat.profitable_trades}",
    f"Losing Trades: {strat.losing_trades}",
    f"Average Trade Duration: {average_trade_duration} minutes",
    f"Profit from Trades: {strat.profit_long}",
    f"Loss from Trades: {strat.loss_long}",
    f"Average Profit from Trades: {strat.profit_long/strat.profitable_trades if strat.profitable_trades else 0}",
    f"Average Loss from Trades: {strat.loss_long/strat.losing_trades if strat.losing_trades else 0}",
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
cerebro.plot(savefig='trading_strategy_chart.png')
pdf.image('trading_strategy_chart.png', x=10, y=8, w=190)

# Save the PDF
pdf.output("trading_strategy_report.pdf")

print("Report generated successfully!")
