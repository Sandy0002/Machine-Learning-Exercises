# In this program we will estimate the category of the coffee
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier

data= pd.read_csv('coffee.csv')
print(data.head())

# We can remove id, farm name country of origin
del data['Unnamed: 0'],data['ID']

# Checking for missing values
# Getting list of missing values columns
ls=[]
for col in data.columns:
  if data[col].dtypes=='object':
    data[col].fillna('Unknown',inplace=True)
  else:
    data[col].fillna(np.mean(data[col]),inplace=True)

# Converting categorical to numerical
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=='object':
    data[col] = encoder.fit_transform(data[col])

# Generating correlation 
correlation_matrix=data.corr()
print(correlation_matrix['Variety'])

# here status, Clean Cup, Sweetness, Defects have Nan value as they single value for whole data
# we remove them 
del data['Status'], data['Clean Cup'],data['Sweetness'], data['Defects']

# Removing less correlated data
for col in data.columns:
  corr = correlation_matrix['Variety'][col]
  if (corr>0 and corr<0.1) or (corr<0 and corr>-0.1):
    del data[col]

# Splitting data
features = data.drop('Variety',axis=1)
target = data['Variety']

xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=0.2)
model = BaggingClassifier()

params = {'n_estimators' : range(25,500,25),
          'max_samples' : range(1,30),
          'max_features': range(1,len(data.columns)+1),
}

search = RandomizedSearchCV(model,param_distributions=params,cv=20,scoring='accuracy')
search.fit(xTrain,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Building best model
model = BaggingClassifier(max_features=12, max_samples=28, n_estimators=425)
model.fit(xTrain,yTrain)

yPred = model.predict(xTest)
# Model Evaluation
print(confusion_matrix(yTest,yPred))
print(accuracy_score(yTest,yPred))