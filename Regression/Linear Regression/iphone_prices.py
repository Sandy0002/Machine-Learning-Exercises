# In this program we are going to build a linear model which predicts the Iphone price for a certain year

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Loading data
data = pd.read_csv('Datasets\\iphone_prices.csv')
data = data.dropna()
# Defining variables
x = data['Version']
y = data['Prices']

# Building model
results = sm.OLS(y,x).fit()

# Generating model summary
print(results.summary())

# Plotting the orginial and predicted values
# for this we need our label and feature variables
plt.scatter(data['Version'],data['Prices'])

# Equating the intercept and slope we get in summary
eq =  77.2727*x - 205.5455

# Plotting the above eq value as regression line
plt.plot(y,'r')

plt.show()
