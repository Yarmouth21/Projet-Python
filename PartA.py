#import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

""" #test = yf.download('EURUSD=X', start='2010-01-01', end='2025-11-30')

print(test.tail())
print(test.head())

test.rename(columns={'Price': 'Date'}, inplace=True)
test = test.iloc[2:] # Suppression des deux lignes inutiles
test = test.drop(columns=['Volume']) # La colonne n'est pas nécessaire


print(test.head())

test.to_csv('csv/EURUSD.csv') #Données de 2010 à 2025 stockées pour éviter de re-télécharger à chaque fois""" 

data = pd.read_csv('../csv/EURUSD.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['MA_15'] = data['Open'].rolling(window=15).mean()

plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Open'], label='Taux de change EUR/USD')
plt.plot(data['Date'], data['MA_15'], label='Moyenne mobile 15 jours', color='red')
plt.title('Taux de change entre 2010 et 2025')
plt.xlabel('Date')
plt.ylabel('EUR/USD')
plt.grid(True)
plt.xlim(data['Date'].min(), data['Date'].max())
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

data['Delta'] = data['Open'].diff() # Création d'une colonne indiquant la variation journalière
data = data.dropna() # Suppression des lignes avec des valeurs NaN (notamment la première)

X = sm.add_constant(data['Open'].shift(1).dropna())  # Variable indépendante avec constante
y = data['Delta'][1:]  # La première valeur de Delta est vide

model = sm.OLS(y,X).fit()
print(model.summary())