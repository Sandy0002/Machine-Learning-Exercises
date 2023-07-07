import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.metrics import r2_score,mean_squared_error


# Loading data
data = pd.read_csv('possum.csv')
print(data.head())

# Upon observation we find that case represents enumeration so we delete it
del data['case']

# EDA
print(data.info())

# Statistics of numerical data
print(data.describe())

#                             Cleaning data
# Missing value imputation using mean
# below values are inferred upon looking at data description
ageMean = 3.83
footlgthMean = 68.46
data['age'].fillna(ageMean,inplace=True)
data['footlgth'].fillna(footlgthMean,inplace=True)

# Converting qualitative to numerical data
data = pd.get_dummies(data=data,drop_first=True)

# train test split
x = data.drop(['age'],axis=1)
y=data['age']

# Creating Linear Model
model = LinearRegression()
model.fit(x,y)

# Checking for multi collienarity
vif = pd.DataFrame()
vif['Predictor'] = x.columns
vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(vif)

# Since all the columns except 1 are highly collinear we use lasso
lassoModel = LassoCV(cv=5)
lassoModel.fit(x,y)

# Get the coefficients and selected features
coefficients = lassoModel.coef_
selected_features = x.columns[coefficients != 0]

# Print the selected features
print("Selected Features:")
print(selected_features)

# Generating features from above output
x = data[['hdlngth', 'skullw', 'totlngth', 'belly']]
model = LinearRegression()
model.fit(x,y)

# Assessing model score
yPred  = model.predict(x)
print(r2_score(y,yPred))
print(mean_squared_error(y,yPred))