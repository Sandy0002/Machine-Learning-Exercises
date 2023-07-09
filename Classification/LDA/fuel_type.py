# In this program we will try to find the type of the fuel used in the vehicle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,accuracy_score

data = pd.read_csv('fuel_consumption.csv')
print(data.head())

# As we have only one model year we delete it
del data['MODELYEAR']

# As we have FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY as combined value so we remove them
del data['FUELCONSUMPTION_CITY']
del data['FUELCONSUMPTION_HWY']

# Checking for missing values
print(data.isna().sum())

# Conversion of categorical to numerical values
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=='object':
    data[col] = encoder.fit_transform(data[col])

# Getting relation of target value with other variables
correlation_matrix = data.corr()
print(correlation_matrix['FUELTYPE'])

# Splitting data
features = data.drop('FUELTYPE',axis=1)
target = data['FUELTYPE']

# Train Test split
xTrain, xTest, yTrain, yTest = train_test_split(features,target,test_size=0.2)

# Model creation
model = LinearDiscriminantAnalysis()
model.fit(xTrain,yTrain)

# Generating model predictions
yPred = model.predict(xTest)

# Model evaluation
print(confusion_matrix(yTest,yPred))
print(accuracy_score(yTest,yPred))
