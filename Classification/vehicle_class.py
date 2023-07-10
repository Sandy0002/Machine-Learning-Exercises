# In this program we will identify the class of the vehicle
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('fuel_consumption.csv')
print(data.head())

# As model year, fuel consumption_city and fuel consumption_hwy are not useful we remove it
# As for fuel_consumption city and highway there is combined value so we remove it
del data['MODELYEAR'], data['FUELCONSUMPTION_CITY'], data['FUELCONSUMPTION_HWY']

# Checking for missing values
print(data.isna().sum())

# Converting categorical data to numerical
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=='object':
    data[col] = encoder.fit_transform(data[col])

# Getting the relation of vehicle class with other vehicles
correlation_matrix = data.corr()
print(correlation_matrix['VEHICLECLASS'])

#  As fuel type and make doesn't have much impact we can remove them
del data['MAKE'], data['FUELTYPE']

# Splitting labels and target from data
features = data.drop('VEHICLECLASS',axis=1)
target = data['VEHICLECLASS']

model = KNeighborsClassifier()

# Using grid search cv to find best model parameters
# Train-test split
xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=0.2)

# Feature Scaling
scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled = scaler.transform(xTest)


param_grid = {'n_neighbors':range(3,9),
              'weights': ['uniform', 'distance'],
              'p': [1, 2]}

grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
grid_search.fit(xTrainScaled, yTrain)

# Get the best hyperparameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Model generation from above parameters
model = KNeighborsClassifier(n_neighbors = 6, p=1,weights='distance')
model.fit(xTrainScaled,yTrain)

# predictions
yPred = model.predict(xTestScaled)

# Model Evaluation
print(confusion_matrix(yTest,yPred))
print(accuracy_score(yTest,yPred))