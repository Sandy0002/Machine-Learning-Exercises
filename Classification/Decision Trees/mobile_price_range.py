# In this program we will identify the variant of the mobile using price range column
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Loading dataset
data = pd.read_csv('mobile_phone.csv')
print(data.head())

# Checking for missing values
print(data.isna().sum())

# Outlier detection
print(data.describe())

# Potential outliers px_ht
sns.scatterplot(data['px_height'])
plt.show()

# Correlation analysis
correlation_matrix =data.corr()
print(correlation_matrix['price_range'])

# Dropping non correlated data
for col in data.columns:
  val = correlation_matrix['price_range'][col]
  if (val>0 and val<0.1) or (val<0 and val>-0.1):
    del data[col]

# Data split
features = data.drop('price_range',axis=1)
target = data['price_range']

# Search for best parameters
model = DecisionTreeClassifier()
xTrain,xTest,yTrain,yTest = train_test_split(features,target,test_size=0.3)
size = len(features.columns)
params = {
    'criterion':['gini', 'entropy', 'log_loss'],
    'splitter' : ['best','random'],
    'max_depth': range(1,6)
}

search = GridSearchCV(model, param_grid =params,cv=20,scoring='accuracy')
search.fit(xTrain,yTrain)

# Get the best hyperparameters and best estimator
best_params = search.best_params_
best_estimator = search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)

# model creation
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, splitter='best')
model.fit(xTrain,yTrain)

yPred = model.predict(xTest)
print(confusion_matrix(yTest,yPred))
print(accuracy_score(yTest,yPred))