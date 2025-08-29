# Pairs Trading

This project implements and analyzes a **pairs trading strategy**, a classic example of statistical arbitrage in quantitative finance. The strategy identifies two historically correlated assets, monitors their price spread, and executes long/short positions when the spread deviates from its historical mean.

---

## 📌 Project Overview
- **Objective:** Explore the profitability and risk characteristics of pairs trading using real financial data.  
- **Approach:**  
  1. Select pairs of correlated stocks (e.g., based on cointegration tests).  
  2. Monitor their spread using z-scores.  
  3. Enter trades when the spread diverges beyond a threshold.  
  4. Close trades when the spread reverts to the mean.  
- **Tech Stack:** Python, Pandas, NumPy, Matplotlib, Statsmodels.  

---

## ⚙️ Features
- Data collection and preprocessing.  
- Correlation and cointegration testing for pair selection.  
- Z-score calculation for spread monitoring.  
- Backtesting engine for historical performance.  
- Visualization of price series, spreads, and trading signals.  



