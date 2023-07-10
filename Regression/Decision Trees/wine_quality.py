# In this program we will determine the quality of wine
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Loading dataset
data = pd.read_csv('wine-quality.csv')
print(data.head())

# Missing value check
print(data.isna().sum())

# Correlation analysis
correlation_matrix = data.corr()
print(correlation_matrix['quality'])

# As residual sugar and pH are not strongly correlated we remove it
for col in data:
  correlation = correlation_matrix['quality'][col]
  if (correlation>0 and correlation<0.1) or (correlation<0 and correlation>-0.1):
    del data[col]

# Data split
features = data.drop('quality',axis=1)
target  = data['quality']

model = RandomForestRegressor()
xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=0.2)
params = {'n_estimators' : range(25,300,25),
          'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
          'max_depth':range(1,6),
          'min_samples_split': range(3,7),
          'min_samples_leaf': range(2,10),
          }

search = RandomizedSearchCV(model,param_distributions=params,cv=10,scoring='neg_mean_squared_error')
search.fit(xTrain,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

yPred = model.predict(xTest)
print(mean_squared_error(yTest,yPred))
print(r2_score(yTest,yPred))