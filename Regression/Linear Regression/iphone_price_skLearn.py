# In this program we are going to build a linear regression model using Sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Loading data
data = pd.read_csv('Datasets\\iphone_prices.csv')
data = data.dropna()

# When we are splitting data we need to pass a 2d array as sklearn expects a 2d array as input
trainData = data[['Version']]
testData = data[['Prices']]

# Model creation
model = LinearRegression()

# model fitting
model.fit(trainData,testData)

# Slope(coef_) and intercept(intercept_)
print(model.intercept_, model.coef_)

# Getting model accuracy
print(r2_score(testData,predictions))

# Plotting the points and model predictions line
plt.scatter(trainData,testData)
predictions = model.predict(trainData)

plt.plot(predictions,color='black')
plt.show()
