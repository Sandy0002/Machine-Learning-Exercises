# In this program we will predict whether loan will be approved or not
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('loan_sanction.csv')
print(data.head())

# We can remove loan_id as its unecessary
del data['Loan_ID']

# Checking for missing values
print(data.isna().sum())

# Missing value imputation
for col in data.columns:
  if data[col].isna().any():
    if data[col].dtypes=='object':
      # Most repeated value
      colMode = data[col].value_counts().index[0]
      data[col].fillna(colMode,inplace=True)
    else:
      colMean = np.mean(data[col])
      data[col].fillna(colMean,inplace=True)

print(data.info())

# In Dependents table we have values like '3+' we need to change them
data.loc[data['Dependents']=='3+','Dependents']='3.5'

# Encoding categorical data
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=="object":
    data[col] = encoder.fit_transform(data[col])

# Correlation Analysis
correlation_matrix = data.corr()
print(correlation_matrix['Loan_Status'])

# We can remove Self_Employed, ApplicantIncome
del data['Self_Employed'], data['ApplicantIncome']

# Split data
features = data.drop('Loan_Status',axis=1)
target = data['Loan_Status']

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(features, target, test_size=0.2)

# Feature Scaling
scaler = StandardScaler()
xTrainScale = scaler.fit_transform(xTrain)
xTestScale = scaler.transform(xTest)

model = SVC()

params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
          'gamma':['scale','auto'],
          'C':range(25,1001,25)}

# Searching best parameters
search = RandomizedSearchCV(model,param_distributions=params,cv=10,scoring='accuracy')
search.fit(xTrainScale,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Model Building using best parameters
model = SVC(C=50,kernel='linear',gamma='scale')
model.fit(xTrainScale,yTrain)

yPred = model.predict(xTestScale)
# Model Evaluation
print(confusion_matrix(yTest,yPred))
print(accuracy_score(yTest,yPred))
