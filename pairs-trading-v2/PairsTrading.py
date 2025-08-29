import numpy as np
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

#SETUP
print("=" * 80)
print("GOOGL-NFLX PAIRS TRADING WITH REALISTIC TRANSACTION COSTS")
print("=" * 80)

# Download data
tickers = ["GOOGL", "NFLX"]
start_date = "2021-01-01"
end_date = datetime.now()

print(f"Downloading data for {tickers} from {start_date}...")
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
data = raw_data["Close"]

# Check data quality
print(f"Data shape: {data.shape}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"Missing data - GOOGL: {data['GOOGL'].isna().sum()}, NFLX: {data['NFLX'].isna().sum()}")

# Remove any NaN values
data = data.dropna()
print(f"Clean data shape: {data.shape}")

#TRANSACTION COST PARAMETERS
print(f"\nTransaction Cost Structure:")
print("-" * 40)

# Realistic costs for small institutional trader
bid_ask_spread = 0.0003  # 3 bps per leg (tight for liquid stocks)
commission = 0.00001  # $0.001 per share, ~0.1bps for $100 stock
market_impact = 0.0001  # 1 bp market impact (small size)
slippage = 0.0002  # 2 bps slippage

cost_per_leg = bid_ask_spread + commission + market_impact + slippage
cost_per_round_trip = 2 * cost_per_leg  # Enter/exit both legs

print(f"Bid-ask spread: {bid_ask_spread * 10000:.1f} bps per leg")
print(f"Commission: {commission * 10000:.1f} bps per leg")
print(f"Market impact: {market_impact * 10000:.1f} bps per leg")
print(f"Slippage: {slippage * 10000:.1f} bps per leg")
print(f"TOTAL: {cost_per_leg * 10000:.1f} bps per leg")
print(f"Round-trip cost: {cost_per_round_trip * 10000:.1f} bps ({cost_per_round_trip:.4f})")

#STRATEGY PARAMETERS
lookback_window = 252  # 1 year for parameter estimation
entry_threshold = 1.5  # Z-score thresholds
exit_threshold = 0.0

print(f"\nStrategy Parameters:")
print(f"Lookback window: {lookback_window} days")
print(f"Entry threshold: ±{entry_threshold} z-score")
print(f"Exit threshold: {exit_threshold} z-score")

#BACKTEST IMPLEMENTATION
print(f"\nRunning backtest...")

# Store results
results = []

for i in range(lookback_window, len(data)):
    current_date = data.index[i]

    # Historical data for parameter estimation (no look-ahead bias)
    hist_start = i - lookback_window
    hist_end = i  # Use up to current day for parameter estimation

    googl_hist = data['GOOGL'].iloc[hist_start:hist_end]
    nflx_hist = data['NFLX'].iloc[hist_start:hist_end]

    # Current prices
    googl_current = data['GOOGL'].iloc[i]
    nflx_current = data['NFLX'].iloc[i]

    try:
        # Estimate hedge ratio: GOOGL = alpha + beta * NFLX + error
        X = sm.add_constant(nflx_hist)
        ols_model = sm.OLS(googl_hist, X).fit()
        alpha, beta = ols_model.params

        # Historical spread for normalization
        spread_hist = googl_hist - (alpha + beta * nflx_hist)
        mu = spread_hist.mean()
        sigma = spread_hist.std(ddof=1)

        # Current spread and z-score
        spread_current = googl_current - (alpha + beta * nflx_current)
        zscore = (spread_current - mu) / sigma if sigma > 0 else 0

        results.append({
            'date': current_date,
            'googl_price': googl_current,
            'nflx_price': nflx_current,
            'alpha': alpha,
            'beta': beta,
            'spread': spread_current,
            'zscore': zscore,
            'mu': mu,
            'sigma': sigma
        })

    except Exception as e:
        print(f"Error on {current_date}: {e}")
        continue

# Convert to DataFrame
backtest_df = pd.DataFrame(results)
backtest_df.set_index('date', inplace=True)

print(f"Backtest completed: {len(backtest_df)} trading days")

#TRADING LOGIC WITH TRANSACTION COSTS

# Generate trading signals
positions = []
current_position = 0

for _, row in backtest_df.iterrows():
    z = row['zscore']

    # State machine for positions
    if current_position == 0:  # No position
        if z < -entry_threshold:
            current_position = 1  # Long the spread
        elif z > entry_threshold:
            current_position = -1  # Short the spread
    elif current_position == 1:  # Long spread position
        if z >= exit_threshold:
            current_position = 0  # Exit to flat
    elif current_position == -1:  # Short spread position
        if z <= exit_threshold:
            current_position = 0  # Exit to flat

    positions.append(current_position)

backtest_df['position'] = positions

# Calculate returns
backtest_df['googl_ret'] = backtest_df['googl_price'].pct_change()
backtest_df['nflx_ret'] = backtest_df['nflx_price'].pct_change()

# Lag positions (can't trade on same-day signal)
backtest_df['position_lag'] = backtest_df['position'].shift(1).fillna(0)

# Spread return calculation
backtest_df['spread_return'] = (backtest_df['googl_ret'] -
                                backtest_df['beta'] * backtest_df['nflx_ret'])

# Gross P&L (before costs)
backtest_df['pnl_gross'] = (backtest_df['position_lag'] *
                            backtest_df['spread_return']).fillna(0)

# Transaction costs
backtest_df['position_change'] = backtest_df['position'].diff().abs().fillna(0)
backtest_df['transaction_costs'] = backtest_df['position_change'] * cost_per_round_trip

# Net P&L (after costs)
backtest_df['pnl_net'] = backtest_df['pnl_gross'] - backtest_df['transaction_costs']


#PERFORMANCE METRICS

def calculate_metrics(returns, label):
    """Calculate comprehensive performance metrics"""
    total_return = (1 + returns).prod() - 1

    # PROPER annualization
    num_days = len(returns)
    num_years = num_days / 252
    annual_return = ((1 + total_return) ** (1 / num_years)) - 1 if num_years > 0 else 0

    annual_vol = returns.std(ddof=1) * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding().max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Win rate
    winning_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = winning_days / total_days if total_days > 0 else 0

    # Trade analysis
    trades = (backtest_df['position_change'] > 0).sum()
    total_costs = backtest_df['transaction_costs'].sum()

    print(f"\n{label} Performance Metrics:")
    print("-" * 50)
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Annualized Volatility: {annual_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Number of Trades: {trades}")
    print(f"Total Transaction Costs: {total_costs:.4f} ({total_costs:.2%})")

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': trades,
        'total_costs': total_costs
    }


# Calculate performance for both gross and net
gross_metrics = calculate_metrics(backtest_df['pnl_gross'], "GROSS (Before Costs)")
net_metrics = calculate_metrics(backtest_df['pnl_net'], "NET (After Costs)")

# Cost impact analysis
cost_impact = gross_metrics['annual_return'] - net_metrics['annual_return']
print(f"\nTransaction Cost Impact:")
print(f"Annual return reduction: {cost_impact:.2%}")
print(f"Cost as % of gross return: {cost_impact / gross_metrics['annual_return'] * 100:.1f}%" if gross_metrics[
                                                                                                     'annual_return'] != 0 else "N/A")

# VISUALIZATION

fig, ((ax1, ax2, ax4)) = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Z-score and positions
ax1.plot(backtest_df.index, backtest_df['zscore'],
         label='Z-score', alpha=0.7, linewidth=1)
ax1.axhline(entry_threshold, linestyle='--', color='red',
            label=f'Short Entry (+{entry_threshold})')
ax1.axhline(-entry_threshold, linestyle='--', color='green',
            label=f'Long Entry (-{entry_threshold})')
ax1.axhline(0, linestyle=':', color='black', alpha=0.5)

# Color background by position
for pos_val, color, alpha in [(-1, 'red', 0.1), (1, 'green', 0.1)]:
    mask = backtest_df['position'] == pos_val
    ax1.fill_between(backtest_df.index, ax1.get_ylim()[0], ax1.get_ylim()[1],
                     where=mask, alpha=alpha, color=color)

ax1.set_title('GOOGL-NFLX Z-Score and Trading Positions')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('Z-Score')

# Plot 2: Cumulative returns comparison
cum_gross = (1 + backtest_df['pnl_gross']).cumprod()
cum_net = (1 + backtest_df['pnl_net']).cumprod()

ax2.plot(backtest_df.index, cum_gross,
         label=f'Gross Return ({gross_metrics["total_return"]:.1%})',
         linewidth=2)
ax2.plot(backtest_df.index, cum_net,
         label=f'Net Return ({net_metrics["total_return"]:.1%})',
         linewidth=2)
ax2.set_title('Cumulative Returns: Gross vs Net')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylabel('Cumulative Return')


# Plot 4: Spread with trading signals
ax4.plot(backtest_df.index, backtest_df['spread'],
         label='Spread (GOOGL - β×NFLX)', alpha=0.7)
ax4.axhline(backtest_df['mu'].iloc[-1], color='black', linestyle='--',
            alpha=0.5, label='Long-term Mean')

# Mark entry/exit points
entries = backtest_df[backtest_df['position_change'] > 0]
if len(entries) > 0:
    ax4.scatter(entries.index, entries['spread'],
                c='red', s=30, alpha=0.7, label='Trade Entries')

ax4.set_title('Spread with Trade Entry Points')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylabel('Spread ($)')

plt.tight_layout()
plt.show()

# ---------------- FINAL SUMMARY ----------------
print("\n" + "=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

print(f"\nStrategy Performance:")
print(f"• Net Sharpe Ratio: {net_metrics['sharpe']:.2f}")
print(f"• Annual Return (after costs): {net_metrics['annual_return']:.1%}")
print(f"• Maximum Drawdown: {net_metrics['max_drawdown']:.1%}")
print(f"• Win Rate: {net_metrics['win_rate']:.1%}")

print(f"\nTransaction Cost Analysis:")
print(f"• Total trades: {net_metrics['num_trades']}")
print(f"• Cost per trade: {cost_per_round_trip:.3%}")
print(f"• Annual cost impact: -{cost_impact:.1%}")
print(f"• Trading frequency: {net_metrics['num_trades'] / (len(backtest_df) / 252):.0f} trades/year")



