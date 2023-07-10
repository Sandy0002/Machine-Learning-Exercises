# In this program we will estimate the insurance premium that would be applicable for a customer.
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR

data = pd.read_csv('insurance.csv')
print(data.head())

# Checking for missing values
print(data.isna().sum())

# Getting data information
print(data.info())

# Data description
# Categorical to numerical conversion of data
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=='object':
    data[col] = encoder.fit_transform(data[col])

# Getting relation of target variable with other variables
correlation_matrix = data.corr()
print(correlation_matrix['charges'])

# Here region have negligble impact so we remove it
del data['region']

# Splitting data
features = data.drop('charges',axis=1)
target = data['charges']

# train test split
xTrain,xTest, yTrain,yTest = train_test_split(features,target,test_size=0.3)

# Scale the data
scaler = StandardScaler()
xTrainScale = scaler.fit_transform(xTrain)
xTestScale = scaler.transform(xTest)

# We will use grid search to find optimal parameters
model = SVR()

params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
          'gamma':['scale','auto'],
          'C':range(25,1001,25)}

search = RandomizedSearchCV(model,param_distributions=params,cv=10,scoring='neg_mean_squared_error')
search.fit(xTrainScale,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Model creation using parameters
model = SVR(kernel='poly',C=475,gamma='auto')
model.fit(xTrainScale,yTrain)

# Model Evaluation
yPred = model.predict(xTestScale)
print(mean_squared_error(yTest,yPred))
print(r2_score(yTest,yPred))
