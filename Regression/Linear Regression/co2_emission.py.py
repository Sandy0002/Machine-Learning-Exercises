# In this program we will estimtate the CO2 emissions from the vehicle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# Loading dataset
data = pd.read_csv('fuel_consumption.csv')
print(data.head())

# Checking missing values
print(data.isna().sum())

# Plotting pair plots to get relation
sns.pairplot(data)
plt.show()

# As we can see there is no relation of model year with other variables so we drop it
data.drop('MODELYEAR',axis=1,inplace=True)

# Getting data information
print(data.info())


# Treating categorical data
# data = pd.get_dummies(data=data,drop_first=True)
print(data['MAKE'].unique())
print(data['MODEL'].unique())
print(data['VEHICLECLASS'].unique())
print(data['TRANSMISSION'].unique())
print(data['FUELTYPE'].unique())

# Here instead of get_dummies we will use encoding as there are lots of label values
encoder = LabelEncoder()
# Converting categorical data into numerical data
data['MAKE'] = encoder.fit_transform(data['MAKE'])
data['MODEL'] = encoder.fit_transform(data['MODEL'])
data['VEHICLECLASS'] = encoder.fit_transform(data['VEHICLECLASS'])
data['TRANSMISSION'] = encoder.fit_transform(data['TRANSMISSION'])
data['FUELTYPE'] = encoder.fit_transform(data['FUELTYPE'])


# Feature selection
features= list(data.drop('CO2EMISSIONS',axis=1))
target = 'CO2EMISSIONS'
size = len(features)
xTrain,xTest,yTrain,yTest = train_test_split(data[features],data[target],test_size=0.3)
print(xTest)
# Features selection
K=0
mini=10**9
selected_feature=[]
for k in range(3,size+1):
  selector = SelectKBest(f_regression, k=k)
  xTrainNew = selector.fit_transform(xTrain, yTrain)
  xTestNew = selector.transform(xTest)
  model = LinearRegression()
  model.fit(xTrainNew, yTrain)
  yPred = model.predict(xTestNew)
  error = mean_squared_error(yTest,yPred)
  # here if our mini > erorr then it means our current error is greater than computed so we take k value store futher process and features
  if mini > error:
    mini = error
    K = k
    selected_features = [features[i] for i in selector.get_support(indices=True)]

# Model building
model = LinearRegression()
model.fit(data[selected_features],data[target])

# Model Accuracy
targetPred = model.predict(data[selected_features])
print(r2_score(data[target],targetPred))