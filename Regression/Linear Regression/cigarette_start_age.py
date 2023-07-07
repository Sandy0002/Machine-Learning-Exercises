# In this program we will estimate the intiation age of cigratte of Indian Youth
#  Youth Tobacco Usage Data (India) dataset
import pandas as pd
from sklearn.linear_model import LinearRegression,RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Loading dataset
data = pd.read_csv('tobacco.csv')
print(data.head())

# Getting column names
print(data.columns)

# Setting target
target =  'Median age of initiation of Cigarette (in years)'

#                             DATA CLEANING
# Checking null value count
print(data.isna().sum())

# Getting data information
print(data.info())

# We find that numbers are written as string so we correct their type

# Target column treatement
print(data[target].unique())
data.loc[data[target]=='<7', target] = "6"
data.loc[data[target]=='--', target] = "10"
data[target] = pd.to_numeric(data[target])

'''Median age of initiation of Bidi (in years)                                                                                 
  Median age of initiation of smokeless tobacco (in years)     
  Students who saw anyone using tobacco on mass media in past 30 days  (%)
  Brought Cigarette as individual sticks in past 30 days    (%)'''

# Fixing remaining columns whose values are numerical but stored as string

col = 'Median age of initiation of Bidi (in years)'
print(data[col].unique())
data.loc[data[col]=='<7.0',col]="6"
data[col] = pd.to_numeric(data[col])

col = 'Median age of initiation of smokeless tobacco (in years)'
print(data[col].unique())
data.loc[data[col]=='<7',col]='6'
data.loc[data[col]=='<7.0',col]='6'
data.loc[data[col]=='--',col]='10.1'
data[col] = pd.to_numeric(data[col])

col = 'Students who saw anyone using tobacco on mass media in past 30 days  (%)'
print(data[col].unique())
data.loc[data[col]=='63..4',col]='63.4'
data[col] = pd.to_numeric(data[col])

col = 'Bought Cigarette as individual sticks in past 30 days    (%)'
print(data[col].unique())
data.loc[data[col]=='35,2',col]='35.2'
data[col] = pd.to_numeric(data[col])

# Converting categorical features to numerical which is state/ut column
data = pd.get_dummies(data,drop_first=True)

# Build a ridge model for getting the best model parameters
x = data.drop(target,axis=1)
y = data[target]
ridgeModel = RidgeCV(cv=10)
ridgeModel.fit(x,y)

coefficients = ridgeModel.coef_
selected_features = x.columns[coefficients!=0]
print(selected_features)

# Model building using the best features chosen using Ridge regression
x = data[list(selected_features)]
y = y
model = LinearRegression()
model.fit(x,y)

# Generating Model Predictions
yPred = model.predict(x)

# Assessing model error
print(mean_squared_error(y,yPred))

# Getting Model Accuracy
print(r2_score(y,yPred))