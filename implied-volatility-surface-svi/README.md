# 📈 TSLA Volatility Surface & SVI Smile Fitting

This project builds a fully reproducible framework in Python and JAX to estimate the implied volatility (IV) surface of TSLA options 
and fit the Stochastic Volatility Inspired (SVI) model to a chosen volatility smile.

---

## 🧠 Key Features

- ✅ **IV Estimation** using JAX + Newton-Raphson with autodifferentiation  
- ✅ **Data Source:** Live options data fetched from Yahoo Finance via `yfinance`  
- ✅ **Surface Construction**: Moneyness vs. Maturity vs. Implied Volatility  
- ✅ **SVI Calibration**: Fitting a smooth parametric curve to a selected volatility smile  
- ✅ **Data Cleaning & Filtering**: Ensures liquidity, low spreads, and reliable smile shapes  
- ✅ **Reproducibility**: Code organized as a single Jupyter Notebook

---

## 🔧 Tools & Libraries

- `Python`
- `JAX`
- `SciPy`
- `Pandas`
- `Matplotlib`
- `yfinance`

---

## 📊 Visual Output

### 1. Implied Volatility Surface

A 3D plot of implied volatility across time-to-maturity and log-moneyness.

### 2. Raw Volatility Smile

A 2D scatterplot of strikes vs. IV for options with 20 to 60 day maturity.

### 3. SVI Fit

The final smile is calibrated using the SVI raw parameterization:

\[
w(x) = a + b\left[\rho(x - m) + \sqrt{(x - m)^2 + \sigma^2}\right]
\]

---

## 🧠 Key Learnings

- The volatility surface reveals rich structure — upward sloping smiles for OTM calls in TSLA suggest strong demand for upside exposure.
- SVI fits allow compression of IV data into 5 interpretable parameters — useful for risk analysis, pricing, and arbitrage detection.
- JAX’s automatic differentiation enables fast and precise gradient-based estimation of IVs.


