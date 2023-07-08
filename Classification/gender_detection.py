# In this program we will predict the gender of the possum
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading dataset
data = pd.read_csv('possum.csv')
print(data.head())

# Since case is just enumeration we drop it
del data['case']

# Check for missing values
print(data.isna().sum())

# Observations
# There are 2 missing values in age column and 1 in footlgth

# We will get the mean of the data using description of the data
print(data.describe())

ageMean = 3.83
footLgthMean = 68.46
data['age'].fillna(ageMean,inplace=True)
data['footlgth'].fillna(footLgthMean,inplace=True)

# We find that Pop and sex are categorical data 
# Now we will convert the categorical data into numerical in features
data = pd.get_dummies(data,drop_first=True)

# Splitting data
features = data.drop('sex_m',axis=1)
target = data['sex_m']

# Constant addition
features = sm.add_constant(features)

# Model creation
model = sm.Logit(target,features).fit()

# Model predictions
targetPred = model.predict(features)

# Rounding up the vaues as this returns decimal values
targetPred = list(map(round, targetPred))

# Model Performance
confuseMatrix = confusion_matrix(target,targetPred)
accuracy = accuracy_score(target,targetPred)
print(confuseMatrix)
print(accuracy)
