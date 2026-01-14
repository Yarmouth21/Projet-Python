import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

window = 80
data = pd.read_csv("csv/EURUSD.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

S = data["Close"].astype(float)  # closing exchange rate
dates = data["Date"]

positions = []
strategy_ret = []

for t in range(window, len(S) - 1):
    # estimation window: [t-window, ..., t-1]
    X_win = S.iloc[t-window:t]

    # model uses X and ΔX (many do this in log; keep in level if your course expects level)
    X = np.log(X_win)
    dX = X.diff().dropna()
    X_lag = X.shift(1).dropna()

    df = pd.concat([dX, X_lag], axis=1).dropna()
    df.columns = ["dX", "X_lag"]

    Xreg = sm.add_constant(df["X_lag"])
    y = df["dX"]
    res = sm.OLS(y, Xreg).fit()

    alpha_hat = res.params["const"]
    beta_hat = res.params["X_lag"]
    t_beta = res.tvalues["X_lag"]

    # expected change based on today's closing rate X_t
    X_t = np.log(S.iloc[t])
    pred_dX = alpha_hat + beta_hat * X_t

    # trading rule
    if (alpha_hat > 0) and (beta_hat < 0) and (t_beta < -1):
        pos = 1 if pred_dX > 0 else -1
    else:
        pos = 0

    # daily return (portfolio in USD), ignore transaction costs
    r = pos * (S.iloc[t+1] / S.iloc[t] - 1)

    positions.append(pos)
    strategy_ret.append(r)

strategy_ret = pd.Series(strategy_ret, index=dates.iloc[window:len(S)-1], name="ret")
equity = (1 + strategy_ret).cumprod()  # start with 1 dollar

# performance metrics
sharpe = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(252) if strategy_ret.std() != 0 else np.nan
running_max = equity.cummax()
drawdown = equity / running_max - 1
max_dd = drawdown.min()

print(f"Final value from $1: {equity.iloc[-1]:.4f}")
print(f"Sharpe (ann.): {sharpe:.3f}")
print(f"Max drawdown: {max_dd:.2%}")
print(f"% days invested: {(pd.Series(positions)!=0).mean():.2%}")

# plot
plt.figure(figsize=(12, 6))
plt.plot(equity.index, equity.values, label="Strategy equity (start=1)")
plt.title("Cumulative performance of mean-reversion strategy")
plt.xlabel("Date")
plt.ylabel("Portfolio value ($)")
plt.legend()
plt.tight_layout()
plt.show()
