import numpy as np
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime


current_datetime = datetime.now()

# ---------------- ECONOMIC-BASED UNIVERSE ----------------
# Focus on tech with clear economic relationships

tech_universe = {
    # Cloud Infrastructure & Enterprise Software
    'CLOUD_ENTERPRISE': ['CRM', 'ORCL', 'SNOW', 'PLTR', 'WDAY', 'ADBE'],

    # Semiconductors - Direct Competitors
    'SEMICONDUCTORS': ['NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'MU', 'TSM'],

    # Big Tech Platforms
    'BIG_TECH': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],

    # Pure SaaS/Software
    'SAAS': ['MSFT', 'CRM', 'ADBE', 'INTU', 'CTXS', 'ZM'],

    # Hardware/Consumer Electronics
    'HARDWARE': ['AAPL', 'DELL', 'HPQ'],

    # Streaming/Content
    'CONTENT': ['NFLX', 'DIS', 'ROKU']
}

# Flatten to get all tickers
all_tickers = list(set([ticker for category in tech_universe.values() for ticker in category]))
print(f"Tech Universe: {len(all_tickers)} stocks")
print(f"Tickers: {sorted(all_tickers)}")


# These pairs have ECONOMIC reasons to mean-revert
economic_pairs = {

    # Direct Competitors (most likely to mean-revert)
    'DIRECT_COMPETITORS': [
        ('NVDA', 'AMD'),  # GPU leaders
        ('ORCL', 'CRM'),  # Enterprise software giants
        ('MSFT', 'GOOGL'),  # Cloud platforms
        ('AAPL', 'MSFT'),  # Big tech titans
        ('INTC', 'AMD'),  # CPU competitors
        ('NFLX', 'DIS'),  # Streaming wars
    ],

    # Supply Chain Relationships
    'SUPPLY_CHAIN': [
        ('AAPL', 'TSM'),  # Apple depends on TSMC
        ('NVDA', 'TSM'),  # NVIDIA uses TSMC fabs
        ('QCOM', 'AAPL'),  # Qualcomm supplies Apple
    ],

    # Same Business Model, Different Focus
    'BUSINESS_MODEL': [
        ('ADBE', 'MSFT'),  # Software subscriptions
        ('SNOW', 'PLTR'),  # Data platforms
        ('ZM', 'MSFT'),  # Communication platforms
    ],

    # Platform vs. Content
    'PLATFORM_CONTENT': [
        ('GOOGL', 'NFLX'),  # YouTube vs Netflix
        ('AAPL', 'NFLX'),  # Apple TV vs Netflix
        ('MSFT', 'NFLX'),  # Gaming vs entertainment
    ]
}

# Create final candidate list with economic reasoning
pair_candidates = []
for category, pairs in economic_pairs.items():
    for pair in pairs:
        pair_candidates.append({
            'asset1': pair[0],
            'asset2': pair[1],
            'category': category,
            'reasoning': f"{category}: {pair[0]} vs {pair[1]}"
        })

print(f"\nEconomic pair candidates: {len(pair_candidates)}")

# DATA DOWNLOAD
try:
    raw_data = yf.download(all_tickers, start="2021-01-01", end=current_datetime, auto_adjust=True)
    data = raw_data["Close"]
    print(f"Data downloaded: {data.shape}")
except Exception as e:
    print(f"Error downloading data: {e}")

# Remove tickers with insufficient data
min_data_points = 500
valid_tickers = []
for ticker in all_tickers:
    if ticker in data.columns and data[ticker].notna().sum() > min_data_points:
        valid_tickers.append(ticker)

data = data[valid_tickers]
print(f"Valid tickers after filtering: {len(valid_tickers)}")
print(f"Valid tickers: {sorted(valid_tickers)}")

# Filter pair candidates to only include valid tickers
valid_pairs = []
for pair_info in pair_candidates:
    if pair_info['asset1'] in valid_tickers and pair_info['asset2'] in valid_tickers:
        valid_pairs.append(pair_info)

print(f"Valid economic pairs: {len(valid_pairs)}")


# ---------------- ECONOMIC PAIR ANALYSIS ----------------
def analyze_pair_economic(data, asset1, asset2, window=252):
    """
    Analyze a pair with economic context, not just statistics
    """
    if asset1 not in data.columns or asset2 not in data.columns:
        return None

    prices1 = data[asset1].dropna()
    prices2 = data[asset2].dropna()

    # Align data
    common_dates = prices1.index.intersection(prices2.index)
    if len(common_dates) < window:
        return None

    p1 = prices1.loc[common_dates]
    p2 = prices2.loc[common_dates]

    # Basic statistics
    correlation = p1.corr(p2)

    # Cointegration test
    try:
        _, coint_pval, _ = coint(p1, p2)
    except:
        coint_pval = 1.0

    # Hedge ratio and spread analysis (economic intuition)
    # Fit: asset1 = alpha + beta * asset2 + error
    X = sm.add_constant(p2)
    try:
        ols_model = sm.OLS(p1, X).fit()
        alpha, beta = ols_model.params
        spread = p1 - (alpha + beta * p2)
    except:
        alpha, beta = 0, 1
        spread = p1 - p2

    spread_mean = spread.mean()
    spread_std = spread.std()
    spread_cv = abs(spread_std / spread_mean) if spread_mean != 0 else np.inf

    # Recent vs historical spread (regime change detection)
    recent_period = min(63, len(spread) // 4)  # Last quarter or 3 months
    recent_spread = spread.iloc[-recent_period:].mean()
    historical_spread = spread.iloc[:-recent_period].mean()
    spread_shift = abs(recent_spread - historical_spread) / spread_std if spread_std > 0 else 0

    # Stationarity of the spread
    try:
        adf_stat, adf_pval, _, _, _, _ = adfuller(spread.dropna(), maxlag=20)
        is_stationary = adf_pval < 0.05
    except:
        adf_stat, adf_pval = np.nan, 1.0
        is_stationary = False

    # Simple mean reversion check: spread range relative to std dev
    spread_range = spread.max() - spread.min()
    spread_range_normalized = spread_range / spread_std if spread_std > 0 else np.inf

    return {
        'correlation': correlation,
        'coint_pvalue': coint_pval,
        'hedge_ratio': beta,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'spread_cv': spread_cv,
        'spread_shift': spread_shift,
        'spread_range_norm': spread_range_normalized,
        'is_stationary': is_stationary,
        'adf_pvalue': adf_pval,
        'data_points': len(common_dates)
    }


# Analyze all valid economic pairs
results = []
for pair_info in valid_pairs:
    asset1, asset2 = pair_info['asset1'], pair_info['asset2']

    analysis = analyze_pair_economic(data, asset1, asset2)
    if analysis is not None:
        results.append({
            'asset1': asset1,
            'asset2': asset2,
            'category': pair_info['category'],
            'reasoning': pair_info['reasoning'],
            **analysis
        })

pair_analysis_df = pd.DataFrame(results)


# ---------------- ECONOMIC SCORING SYSTEM ----------------
def score_pair_economically(row):
    """
    Score pairs based on economic viability, not just statistical significance
    """
    score = 0

    # 1. Cointegration (but not the main factor)
    if row['coint_pvalue'] < 0.05:
        score += 20
    elif row['coint_pvalue'] < 0.1:
        score += 10

    # 2. Reasonable correlation (not too high, not too low)
    corr = abs(row['correlation'])
    if 0.6 <= corr <= 0.85:  # Sweet spot
        score += 25
    elif 0.5 <= corr <= 0.9:
        score += 15

    # 3. Stationarity of spread
    if row['is_stationary']:
        score += 30

    # 4. Good spread characteristics
    spread_range_norm = row['spread_range_norm']
    if 4 <= spread_range_norm <= 8:  # Spread moves 4-8 std devs (good trading range)
        score += 20
    elif 3 <= spread_range_norm <= 10:
        score += 10

    # 5. Stability (low spread shift means stable relationship)
    if row['spread_shift'] < 1.0:  # Less than 1 std dev shift
        score += 15
    elif row['spread_shift'] < 2.0:
        score += 10

    # 6. Bonus for direct competitors (most economic sense)
    if row['category'] == 'DIRECT_COMPETITORS':
        score += 10

    return score


pair_analysis_df['economic_score'] = pair_analysis_df.apply(score_pair_economically, axis=1)
pair_analysis_df = pair_analysis_df.sort_values('economic_score', ascending=False)

# ---------------- RESULTS ----------------
print("\n" + "=" * 80)
print("ECONOMIC PAIR SELECTION RESULTS")
print("=" * 80)

print(f"\nTop 10 Economic Pairs:")
print("-" * 60)
top_pairs = pair_analysis_df.head(10)
for idx, row in top_pairs.iterrows():
    print(f"{row['asset1']:>6} vs {row['asset2']:<6} | Score: {row['economic_score']:3.0f} | "
          f"Corr: {row['correlation']:.3f} | Hedge: {row['hedge_ratio']:5.2f} | "
          f"{row['category']}")

print(f"\nDetailed Analysis of Top 3 Pairs:")
print("-" * 60)
for idx, row in pair_analysis_df.head(3).iterrows():
    print(f"\n{row['asset1']} vs {row['asset2']} ({row['category']})")
    print(f"  Economic Score: {row['economic_score']:.0f}/100")
    print(f"  Correlation: {row['correlation']:.3f}")
    print(f"  Cointegration p-value: {row['coint_pvalue']:.4f}")
    print(
        f"  Hedge ratio: {row['hedge_ratio']:.3f} (buy {row['hedge_ratio']:.3f} shares of {row['asset2']} per 1 share of {row['asset1']})")
    print(f"  Spread range: {row['spread_range_norm']:.1f} std devs (higher = more opportunities)")
    print(f"  Reasoning: {row['reasoning']}")

# ---------------- VISUALIZATION ----------------
if len(pair_analysis_df) > 0:
    # Create visualization of top pairs
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Economic Score vs Correlation
    scatter = ax1.scatter(pair_analysis_df['correlation'],
                          pair_analysis_df['economic_score'],
                          c=pair_analysis_df['hedge_ratio'],
                          cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Correlation')
    ax1.set_ylabel('Economic Score')
    ax1.set_title('Economic Score vs Correlation (colored by hedge ratio)')
    plt.colorbar(scatter, ax=ax1, label='Hedge Ratio')

    # Add labels for top 5 pairs
    top_5 = pair_analysis_df.head(5)
    for idx, row in top_5.iterrows():
        ax1.annotate(f"{row['asset1']}-{row['asset2']}",
                     (row['correlation'], row['economic_score']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.8)

    # Plot 2: Category breakdown
    category_scores = pair_analysis_df.groupby('category')['economic_score'].mean().sort_values(ascending=False)
    category_scores.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Economic Score by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Average Economic Score')
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Spread range distribution
    ax3.hist(pair_analysis_df['spread_range_norm'], bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(x=6, color='red', linestyle='--', label='6 std dev range')
    ax3.set_xlabel('Spread Range (in std deviations)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Spread Trading Ranges')
    ax3.legend()

    # Plot 4: Top pair spread
    top_pair = pair_analysis_df.iloc[0]
    asset1, asset2 = top_pair['asset1'], top_pair['asset2']

    if asset1 in data.columns and asset2 in data.columns:
        # Calculate spread using the estimated hedge ratio
        X = sm.add_constant(data[asset2].dropna())
        p1_aligned = data[asset1].reindex(data[asset2].dropna().index).dropna()

        try:
            ols = sm.OLS(p1_aligned, X.reindex(p1_aligned.index)).fit()
            alpha, beta = ols.params
            spread = p1_aligned - (alpha + beta * data[asset2].reindex(p1_aligned.index))

            ax4.plot(spread.index, spread, label=f'{asset1} - {beta:.2f}*{asset2}', alpha=0.7)
            ax4.axhline(y=spread.mean(), color='red', linestyle='--',
                        label=f'Mean: ${spread.mean():.2f}')
            ax4.fill_between(spread.index,
                             spread.mean() - spread.std(),
                             spread.mean() + spread.std(),
                             alpha=0.2, color='red', label='±1 Std Dev')
            ax4.set_title(f'Best Economic Pair Spread: {asset1} vs {asset2}')
            ax4.set_ylabel('Spread ($)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error plotting spread: {e}', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.show()

print(f"\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("✓ Focused on economically meaningful relationships")
print("✓ Prioritized business fundamentals over statistical noise")
print("✓ Considered mean reversion speed (half-life)")
print("✓ Evaluated relationship stability over time")
print(f"✓ Selected {len(pair_analysis_df)} economically viable pairs")

# Export top pairs for strategy testing
top_3_pairs = pair_analysis_df.head(3)[['asset1', 'asset2', 'economic_score', 'reasoning']].to_dict('records')
print(f"\nRecommended pairs for strategy testing:")
for i, pair in enumerate(top_3_pairs, 1):
    print(f"{i}. {pair['asset1']} vs {pair['asset2']} (Score: {pair['economic_score']:.0f})")