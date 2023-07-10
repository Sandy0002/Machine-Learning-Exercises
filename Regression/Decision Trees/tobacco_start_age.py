# Estimating the initiation age of tobacco usage.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score


data = pd.read_csv('tobacco.csv')
print(data.head())

# Checking for missing values
print(data.isna().sum())

# Converting categorical data into numerical
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=='object':
    data[col] = encoder.fit_transform(data[col])

# Split of data
features = data.drop('Median age of initiation of smokeless tobacco (in years)',axis=1)
target = data['Median age of initiation of smokeless tobacco (in years)']

# Getting correlation
correlation_matrix = data.corr()
print(correlation_matrix['Median age of initiation of smokeless tobacco (in years)'])

y = 'Median age of initiation of smokeless tobacco (in years)'

# Feature removal who have less correlation strength
for col in features.columns:
  if correlation_matrix[y][col]>0 and correlation_matrix[y][col]<0.1:
    del features[col]

  elif correlation_matrix[y][col]<0 and correlation_matrix[y][col]>-0.1:
    del features[col]

# from sklearn.preprocessing import GridSearchCV

xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=0.2)
model = DecisionTreeRegressor()

params = {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
          'splitter':['best','random'],
          'max_depth':range(1,6),
          'min_samples_split': range(3,7),
          'min_samples_leaf': range(2,10),
          }

search = GridSearchCV(model,param_grid=params,cv=10,scoring='neg_mean_squared_error')
search.fit(xTrain,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Model Creation
model = DecisionTreeRegressor(criterion="friedman_mse",max_depth=5,min_samples_leaf=3,min_samples_split=6)
model.fit(xTrain,yTrain)

yPred = model.predict(xTest)
print(mean_squared_error(yTest,yPred))
print(r2_score(yTest,yPred))