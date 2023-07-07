# In this program we are going to predict the age of the possum
# This is a multiple linear regression example as here there are multiple predictor variables

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Loading data
data = pd.read_csv('Datasets\\possum.csv')
print(data.head())

# Upon observation we find that case represents enumeration so we delete it
del data['case']

# EDA
print(data.info())

# Statistics of numerical data
print(data.describe())

# Upon execution we find that foot length have one missing value and age have 2
# We impute them with their mean values
ageMean = 3.83
footlgthMean = 68.46

# Filling missing values
data['age'].fillna(ageMean,inplace=True)
data['footlgth'].fillna(footlgthMean,inplace=True)

# We can check the relation between the columns using a pairplot
sns.pairplot(data)
plt.show()

# Checking the distribution of the data in cateogrical variables
sns.countplot(x='Pop',data=data)
sns.countplot(x='sex',data=data)
plt.show()

# Converting cateogorical data into numerical
data1 = pd.get_dummies(data)

# delete useless columns
del data1['Pop_other'], data1['sex_f']

# Getting correlation matrix
correlation  = data1.corr()
print(correlation)

trainData = data1[['site', 'hdlngth', 'skullw', 'totlngth', 'taill', 'footlgth','earconch', 'eye', 'chest', 'belly', 'Pop_Vic',]]
testData = data1['age']

# Train test split
xTrain,xTest,yTrain,yTest = train_test_split(trainData,testData,test_size=0.2)

# Model creation 
model = LinearRegression()

# Model fitting
model.fit(xTrain,yTrain)

# Generating predictions
yPred = model.predict(xTest)

# Getting the model accuracy
print(r2_score(yTest,yPred))

# Prediction errors
print(mean_squared_error(yTest,yPred))
