# IN this program we will predict from the features if the person survived on the titanic incident 1912
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier


data = pd.read_csv('titanic.csv')
print(data.head())

# Unamed :0, and PassengerId can be removed
del data['Unnamed: 0'], data['PassengerId']

# Checking for missing values
print(data.info())

# Getting correlation
print(data.describe())

# Checking fare correlation
sns.scatterplot(data['Fare'])

# Getting values that fall under 99 percentile value
maxPercentile = np.percentile(data['Fare'], [99])

 # Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['Fare']>0.2,'Fare']=0.2

sns.scatterplot(data['Age'])

# Correlation matrix
correlation_matrix =data.corr()
print(correlation_matrix['Survived'])

# we can remove Age, Pclass_2, Family_size, Title_4, Emb_2
del data['Age'], data['Pclass_2'],data['Family_size'],data['Title_4'],data['Emb_2']

# Data split
features = data.drop('Survived',axis=1)
target = data['Survived']

# Searching for best parameters
xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=0.2)
model = AdaBoostClassifier()

params = {'n_estimators' : range(25,500,25),
          'learning_rate': [0.0001,0.001,0.01,0.1,1,2,5,10,20],
          'algorithm':['SAMME','SAMME.R']  }

search = RandomizedSearchCV(model, param_distributions=params,cv=25,scoring='accuracy')
search.fit(xTrain,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# Model creation
model =  AdaBoostClassifier(algorithm='SAMME', learning_rate=0.01)
model.fit(xTrain,yTrain)

yPred = model.predict(xTest)
print(confusion_matrix(yTest,yPred))
print(accuracy_score(yTest,yPred))