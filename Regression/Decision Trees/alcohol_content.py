# In this program we will try to determine alcohol content in the wine.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import BaggingRegressor

data = pd.read_csv('wine-quality.csv')
print(data.head())

# As there are no missing values we will check for outliers
print(data.describe())

# Upon description we find that there are chances of outliers in
# residual_sugar, chlorides, free sulfur dioxide ,total sulfur dioxide


sns.scatterplot(data['residual sugar'])
plt.show()

# Outlier treatment for residual sugar
 # Getting values that fall under 99 percentile value
maxPercentile = np.percentile(data['residual sugar'], [99])

 # Getting values that fall under 1 percentile value
upperValue = np.percentile(data['residual sugar'], [99])[0]
data.loc[data['residual sugar']>9,'residual sugar']=upperValue

sns.scatterplot(data['chlorides'])
plt.show()

# Outlier treatment of chlorides
maxPercentile = np.percentile(data['chlorides'], [99])

 # Getting values that fall under 1 percentile value
upperValue = np.percentile(data['chlorides'], [99])[0]
data.loc[data['chlorides']>0.5,'chlorides']=upperValue

sns.scatterplot(data['free sulfur dioxide'])

# Outlier treatment
maxPercentile = np.percentile(data['free sulfur dioxide'], [99])

 # Getting values that fall under 1 percentile value
upperValue = np.percentile(data['free sulfur dioxide'], [99])[0]
data.loc[data['free sulfur dioxide']>60,'free sulfur dioxide']=upperValue

sns.scatterplot(data['total sulfur dioxide'])

# Outlier treatment for total sulfur dioxide
maxPercentile = np.percentile(data['total sulfur dioxide'], [99])

 # Getting values that fall under 1 percentile value
upperValue = np.percentile(data['total sulfur dioxide'], [99])[0]
data.loc[data['total sulfur dioxide']>200,'total sulfur dioxide']=upperValue

# Generating correlation
correlation_matrix = data.corr()
print(correlation_matrix['alcohol'])

# Removing unecessary columns
for col in data.columns:
  val = correlation_matrix['alcohol'][col]
  if (val >0 and val<0.1) or (val<0 and val>-0.1):
    del data[col]

# Split data
features = data.drop('alcohol',axis=1)
target =data['alcohol']

# Searching for optimal parameters
size = len(features.columns)
model = BaggingRegressor()
xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=0.2)
params = {'n_estimators' : range(10,300,15),
          'max_samples': range(5,100,15),
          'max_features' : range(1,size+1)
          }

search = RandomizedSearchCV(model,param_distributions=params,cv=10,scoring='neg_mean_squared_error')
search.fit(xTrain,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Building best model
model = BaggingRegressor(n_estimators=235,max_samples=95,max_features=5)
model.fit(xTrain,yTrain)

yPred = model.predict(xTest)
print(mean_squared_error(yTest,yPred))
print(r2_score(yTest,yPred))