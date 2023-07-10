# Estimating the productivity of the employees
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('garments_worker_productivity.csv')
print(data.head())

# As date won't be useful we remove it
del data['date']

print(data[data['wip'].isna()])

# We have 506 missing values only at wip
sns.scatterplot(data['wip'])

# Upon plotting we find that there certain outliers so we treat them

# here we get 99 percentile value as nan value so we calculate the 99 percentile by creating another dataset
# as data1 here
data1 = data.dropna()
maxPercentile = np.percentile(data1['wip'], [99])
# Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['wip']>5000,'wip']=upperValue

# Missing value imputation
data['wip'].fillna(upperValue,inplace=True)

# Checking for outliers using description of data
print(data.describe())

# Data that might have outliers are :
# over_time, incentive, idle_time, idle_men
sns.scatterplot(data['over_time'])

# Outlier treatment
maxPercentile = np.percentile(data['over_time'], [99])
# Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['over_time']>15000,'over_time']=upperValue

# Incentive outliers treatment
maxPercentile = np.percentile(data['incentive'], [99])
# Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['incentive']>150,'incentive']=upperValue

# idle_time, idle_men
sns.scatterplot(data['idle_men'])

# Idle men outliers treatment
maxPercentile = np.percentile(data['idle_men'], [99])
# Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['incentive']>10,'idle_men']=upperValue

sns.scatterplot(data['idle_time'])

# Idle time outliers treatment
maxPercentile = np.percentile(data['idle_time'], [99])
# Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['idle_time']>10,'idle_time']=upperValue

# Correlation analysis
correlation_matrix = data.corr()
print(correlation_matrix['actual_productivity'])

# Very weak relation of idle_time and idle_men
del data['idle_time'], data['idle_men'], data['no_of_workers']

# Categorical to numerical conversion
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=='object':
    data[col] = encoder.fit_transform(data[col])

# Searching for best parameters

# Split data
features = data.drop('actual_productivity',axis=1)
target =data['actual_productivity']

# Searching for optimal parameters
size = len(features.columns)
model = GradientBoostingRegressor()
xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=0.2)
params = {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'], 
        'learning_rate' : [0.0001,0.001,0.01,0.1,1,10,5,10,20],
        'criterion':['friedman_mse', 'squared_error'],
        'n_estimators' : range(10,300,15),
          'min_samples_split': range(2,31,2),
          'min_samples_leaf' : range(1,31,2),
          'max_depth':range(1,5)
          }

search = RandomizedSearchCV(model,param_distributions=params,cv=30,scoring='neg_mean_squared_error')
search.fit(xTrain,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Model creation
model = GradientBoostingRegressor(criterion='squared_error', loss='huber', max_depth=2,
                          min_samples_split=16, n_estimators=190)
model.fit(xTrain,yTrain)

yPred = model.predict(xTest)
# Model Evaluation
print(mean_squared_error(yTest,yPred))
print(r2_score(yTest,yPred))