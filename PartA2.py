import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

window = 80 # Fenêtre pour la moyenne mobile
jour_depart = 300 # Jour de départ pour la stratégie
data = pd.read_csv('../csv/EURUSD.csv')
pnl_list = []

for t in range(jour_depart, len(data)-1):
    X_window = data['Open'][t-window:t]
    dX = X_window.diff()[1:].reset_index(drop=True)
    X_lag = X_window.shift(1)[1:].reset_index(drop=True)
    X_lag_const = sm.add_constant(X_lag)
    model = sm.OLS(dX, X_lag_const)
    results = model.fit()
    beta_hat = results.params.iloc[1]
    t_beta = results.tvalues.iloc[1]
    alpha_hat = results.params.iloc[0]

    X_t = data['Open'].iloc[t]
    pred = alpha_hat + beta_hat * X_t
    if alpha_hat > 0 and beta_hat < 0 and t_beta < -1:
        if pred > 0:
            pnl = data['Open'].iloc[t+1] - data['Open'].iloc[t]
        else:
            pnl = data['Open'].iloc[t] - data['Open'].iloc[t+1]
    else:
        pnl = 0
    pnl = pnl*sum(pnl_list) if pnl_list else 1  
    pnl_list.append(pnl)
    
print(f"P&L total sur la période : {sum(pnl_list)}")

# Convertir les dates en datetime si ce n'est pas déjà fait
data['Date'] = pd.to_datetime(data['Date'])

# Utiliser les dates correspondantes pour le plot
dates = data['Date'].iloc[jour_depart:len(data)-1].reset_index(drop=True)
pnl_cumsum = pd.Series(pnl_list, index=dates).cumsum()

plt.figure(figsize=(12, 6))
plt.plot(pnl_cumsum.index, pnl_cumsum.values)
plt.plot(pnl_cumsum.index, data['Open'].iloc[jour_depart:len(data)-1], alpha=0.3, label='Prix EUR/USD')
plt.title("PnL cumulatif de la stratégie")
plt.xlabel("Date")
plt.ylabel("PnL")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
