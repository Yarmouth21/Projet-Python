import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf

# PART A
# 1. Data Downloading
try:
    print("Downloading EUR/USD Data")
    df_a = yf.download("EURUSD=X", start="2010-01-01", end="2025-11-30", progress=False)
    
    # Fix yfinance MultiIndex issue if present
    if isinstance(df_a.columns, pd.MultiIndex):
        df_a.columns = df_a.columns.get_level_values(0)
    
    prices_A = df_a['Close'].dropna()
    print(f"Data loaded: {len(prices_A)} trading days")

except Exception as e:
    print(f"Download failed: {e}")
    exit()

# 2. Theoretical Test
print("Question 1: Full Sample Mean Reversion Test")
diff_full = prices_A.diff().dropna()
lag_full = prices_A.shift(1).dropna()
# Align indices for regression
idx_common = diff_full.index.intersection(lag_full.index)

# Run OLS regression on the entire history
model_full = sm.OLS(diff_full.loc[idx_common], sm.add_constant(lag_full.loc[idx_common])).fit()
beta_full = model_full.params.iloc[1]
t_stat_full = model_full.tvalues.iloc[1]

print(f"Beta: {beta_full:.5f}")
print(f"T-stat: {t_stat_full:.4f}")

if t_stat_full < -1:
    print("Conclusion: Evidence of Mean Reversion detected on full sample.")
else:
    print("Conclusion: No significant evidence on full sample.")

# 3. Strategy Simulation (Main Loop
print("Running Part A Simulation")

# Parameters
window = 80 # From instructions
capital = 1.0
holdings_A = [capital]
dates_A = []
vals_A = prices_A.values
dates_index_A = prices_A.index

# Main Loop Part A
for t in range(window, len(vals_A) - 1):
    
    # a. Rolling Window Preparation
    win_vals = vals_A[t-window:t]
    win_diff = np.diff(win_vals)
    win_lag = win_vals[:-1]
    
    try:
        # Rolling OLS
        X = sm.add_constant(win_lag)
        model = sm.OLS(win_diff, X).fit()
        alpha, beta = model.params[0], model.params[1]
        t_stat = model.tvalues[1]
    except:
        alpha, beta, t_stat = 0, 0, 0

    # b. Trading Decision
    curr_price = vals_A[t]
    pred_change = alpha + beta * curr_price
    market_ret = (vals_A[t+1] - curr_price) / curr_price
    daily_pnl = 0
    
    # Entry Condition (Statistical Significance)
    if alpha > 0 and beta < 0 and t_stat < -1:
        if pred_change > 0:
            daily_pnl = market_ret   # Long
        elif pred_change < 0:
            daily_pnl = -market_ret  # Short

    # c. Update Capital
    capital = capital * (1 + daily_pnl)
    holdings_A.append(capital)
    dates_A.append(dates_index_A[t+1])

# 4. Results & Metrics Part A
equity_curve_A = pd.Series(holdings_A, index=[dates_index_A[window]] + dates_A)

# Calculate Metrics
rets_A = equity_curve_A.pct_change().dropna()
# Annualized Sharpe
sharpe_A = np.sqrt(252) * rets_A.mean() / rets_A.std() if rets_A.std() != 0 else 0
# Max Drawdown
dd_A = (equity_curve_A - equity_curve_A.cummax()) / equity_curve_A.cummax()
max_dd_A = dd_A.min()

print(f"Final Capital : {capital:.4f} $")
print(f"Sharpe Ratio  : {sharpe_A:.4f}")
print(f"Max Drawdown  : {max_dd_A:.2%}")

# Plot Part A
plt.figure(figsize=(10, 5))
plt.plot(equity_curve_A, label='EUR/USD Strategy (Base)', color='blue')
plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)
plt.title("Part A: Simple Mean Reversion Strategy")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.show()


# PART B

pairs = ["EURUSD=X", "GBPUSD=X", "EURGBP=X", "JPYUSD=X"]
resultats_B = {}

for paire in pairs:
    print(f"\nProcessing: {paire} ...")
    
    # 1. Download
    try:
        df_b = yf.download(paire, start="2010-01-01", end="2025-11-30", progress=False)
        if isinstance(df_b.columns, pd.MultiIndex):
            df_b.columns = df_b.columns.get_level_values(0)
        prices_B = df_b['Close'].dropna()
        
        # Check if enough data
        if len(prices_B) < window + 60:
            print("Not enough data points")
            continue
    except:
        print("Error")
        continue

    # 2. Simulation with Unique Filter
    vals_B = prices_B.values
    dates_B = prices_B.index
    capital_B = 1.0
    holdings_B = [capital_B]
    dates_strat_B = []
    
    # Pre-calculate returns for volatility
    all_returns = prices_B.pct_change().fillna(0).values
    vol_tracking = [] # To store historical volatility for percentile calc
    
    # Main Loop Part B
    for t in range(window, len(vals_B) - 1):
        
        # a. Regression
        win_vals = vals_B[t-window:t]
        win_diff = np.diff(win_vals)
        win_lag = win_vals[:-1]
        try:
            model = sm.OLS(win_diff, sm.add_constant(win_lag)).fit()
            alpha, beta, t_stat = model.params[0], model.params[1], model.tvalues[1]
        except:
            alpha, beta, t_stat = 0, 0, 0

        # b. Safety filter
        # Parameter 1 : 15-day window for volatility
        # Parameter 2 : 80th Percentile threshold
        is_risky = False
        
        recent_vol = np.std(all_returns[t-15:t])
        vol_tracking.append(recent_vol)
        
        # We need some history (60 days) to calculate a reliable percentile
        if len(vol_tracking) > 60:
            limit_vol = np.percentile(vol_tracking, 80)
            if recent_vol > limit_vol:
                is_risky = True

        # c. Decision
        curr_price = vals_B[t]
        pred = alpha + beta * curr_price
        ret_mkt = (vals_B[t+1] - curr_price) / curr_price
        pnl = 0

        if alpha > 0 and beta < 0 and t_stat < -1:
            if is_risky:
                pnl = 0 # Safety Mode Activated (Cash)
            else:
                if pred > 0:
                    pnl = ret_mkt
                elif pred < 0:
                    pnl = -ret_mkt

        capital_B = capital_B * (1 + pnl)
        holdings_B.append(capital_B)
        dates_strat_B.append(dates_B[t+1])

    # 3. Metrics & Storage
    curve_B = pd.Series(holdings_B, index=[dates_B[window]] + dates_strat_B)
    
    # Quick Metrics
    rets_B = curve_B.pct_change().dropna()
    sharpe_B = np.sqrt(252) * rets_B.mean() / rets_B.std() if rets_B.std() != 0 else 0
    dd_B = (curve_B - curve_B.cummax()) / curve_B.cummax()
    
    print(f" -> Final Capital : {capital_B:.2f} $")
    print(f" -> Sharpe Ratio  : {sharpe_B:.2f}")
    print(f" -> Max Drawdown  : {dd_B.min():.2%}")
    
    resultats_B[paire] = curve_B

# 4. Final Plot Part B
plt.figure(figsize=(12, 6))

for paire, courbe in resultats_B.items():
    plt.plot(courbe, label=f"{paire} (End: {courbe.iloc[-1]:.2f}$)")

plt.axhline(y=1, color='black', linestyle='--', alpha=0.3)
plt.title("Part B: Strategy with Dynamic Volatility Filter (80th Percentile)")
plt.xlabel("Year")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.show()

print("Exporting Data to CSV")

# Export Part A
try:
    equity_curve_A.name = "EURUSD_Base_Strategy"
    equity_curve_A.to_csv("results_part_A.csv", header=True)
    print("Part A saved to 'results_part_A.csv'")
except Exception as e:
    print(f"Error saving Part A: {e}")

# Export Part B 
try:
    df_results_B = pd.DataFrame(resultats_B)
    df_results_B.to_csv("results_part_B.csv")
    print("Part B saved to 'results_part_B.csv'")
except Exception as e:
    print(f"Error saving Part B: {e}")

print("Done!")