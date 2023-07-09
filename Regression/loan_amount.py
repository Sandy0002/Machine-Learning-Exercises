# In this program we will try to estimate the loan amount that will be aproved
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('loan_sanction.csv')
print(data.head())

# As loan id is not useful so we remove it
del data['Loan_ID']

# Checking for missing values
print(data.isna().sum())

# Gender, Married, Dependents, Self_Employed, Loan_Amount_Term, LoanAmount, Credit_History have missing values
gotLoanMean = data.loc[data['Loan_Status']=='Y']['LoanAmount'].mean()
notGotLoanMean = data.loc[data['Loan_Status']=='N']['LoanAmount'].mean()

# impute the loanAmount with mean values based on their loan statuses
data.loc[data['Loan_Status']=='Y','LoanAmount'] = data.loc[data['Loan_Status']=='Y','LoanAmount'].fillna(gotLoanMean)
data.loc[data['Loan_Status']=='N','LoanAmount'] = data.loc[data['Loan_Status']=='N','LoanAmount'].fillna(notGotLoanMean)

# Imputing  loan amount term based on their loan approval status
loanTermMean = data.loc[data['Loan_Status']=='Y']['Loan_Amount_Term'].mean()
noLoanTermMean = data.loc[data['Loan_Status']=='N']['Loan_Amount_Term'].mean()

# impute the loanAmount with mean values based on their loan statuses
data.loc[data['Loan_Status']=='Y','Loan_Amount_Term'] = data.loc[data['Loan_Status']=='Y','Loan_Amount_Term'].fillna(loanTermMean)
data.loc[data['Loan_Status']=='N','Loan_Amount_Term'] = data.loc[data['Loan_Status']=='N','Loan_Amount_Term'].fillna(noLoanTermMean)

# Imputing credit history
# Here values are in form of 0s and 1s so we check for the count of with loan statuses
haveCreditHistory = data.loc[(data['Credit_History']==1) & (data['Loan_Status']=='Y')]['Credit_History'].count()
notHaveCreditHistory= data.loc[(data['Credit_History']==0) & (data['Loan_Status']=='Y')]['Credit_History'].count()
print(haveCreditHistory, notHaveCreditHistory)

# impute the loanAmount with mean values based on their loan statuses
data.loc[data['Loan_Status']=='Y','Credit_History'] = data.loc[data['Loan_Status']=='Y','Credit_History'].fillna(1)
data.loc[data['Loan_Status']=='N','Credit_History'] = data.loc[data['Loan_Status']=='N','Credit_History'].fillna(0)

# Count of married and got loan
marriedCount = data[(data['Married']=="Yes") &
            (data['Loan_Status']=="Y")]['Married'].count()
unmarriedCount = data[(data['Married']=="No") &
            (data['Loan_Status']=="Y")]['Married'].count()
print(marriedCount,unmarriedCount)

#  From above counts we find that people who are married have more chances to get loans so we impute yes
# so we impute married = yes if loan_status is yes else unmarried
# Imputing nan values with their loan and marital status
data.loc[data['Loan_Status']=='Y','Married'] = data.loc[data['Loan_Status']=='Y','Married'].fillna("Married")
data.loc[data['Loan_Status']=='N','Married'] = data.loc[data['Loan_Status']=='N','Married'].fillna("Unmarried")

# Same we check for gender category
maleCount = data[(data['Gender']=="Male") &
            (data['Loan_Status']=="Y")]['Gender'].count()
femaleCount = data[(data['Gender']=="Female") &
            (data['Loan_Status']=="Y")]['Gender'].count()
print(maleCount,femaleCount)

# As chances of a male getting the loan are very high
# so a loan status with yes will get male and no with female
data.loc[data['Loan_Status']=='Y', 'Gender'] = data.loc[data['Loan_Status']=='Y', 'Gender'].fillna("Male")
data.loc[data['Loan_Status']=='N', 'Gender']  = data.loc[data['Loan_Status']=='N', 'Gender'].fillna("Female")

# checking distribution
sns.countplot(x='Dependents',data=data)
plt.show()

# Here we find that numbers are in form of strings
data.loc[data['Dependents']=='3+','Dependents']='3'
# Converting string to numeric
data['Dependents']=pd.to_numeric(data['Dependents'])

# Check the peple who are having more chances of loan
zeroDependentCount = data[(data['Dependents']==0) & (data['Loan_Status']=="Y")]['Gender'].count()
oneDependentCount = data[(data['Dependents']==1) &  (data['Loan_Status']=="Y")]['Gender'].count()
twoDependentCount = data[(data['Dependents']==2) &  (data['Loan_Status']=="Y")]['Gender'].count()
multiDependentCount = data[(data['Dependents']==3.5) &  (data['Loan_Status']=="Y")]['Gender'].count()
print(zeroDependentCount,oneDependentCount,twoDependentCount,multiDependentCount)

# Its clear with people with no dependence have high chances of loan
# So we impute the people who got loans with 0 else 1
data.loc[data['Loan_Status']=='Y','Dependents'] = data.loc[data['Loan_Status']=='Y','Dependents'].fillna(0)
data.loc[data['Loan_Status']=='N','Dependents'] = data.loc[data['Loan_Status']=='N','Dependents'].fillna(1)

# Checking people who are self employed will get loan or people who are not
selfCount = data[(data['Self_Employed']=="Yes") & (data['Loan_Status']=="Y")]['Gender'].count()
notSelfCount = data[(data['Self_Employed']=="No") & (data['Loan_Status']=="Y")]['Gender'].count()
print(selfCount,notSelfCount)

# People who are not self employed have high chance
# So if loan_status is yes we apply self employed no else viceversa
data.loc[data['Loan_Status']=='Y','Self_Employed'] = data.loc[data['Loan_Status']=='Y','Self_Employed'].fillna("No")
data.loc[data['Loan_Status']=='N','Self_Employed'] = data.loc[data['Loan_Status']=='N','Self_Employed'].fillna("Yes")

# Encoding categorical data
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtypes=="object":
    data[col] = encoder.fit_transform(data[col])

# Generating Correlation matrix
correlation_matrix = data.corr()

# Getting relation of the target variable with other variables
print(correlation_matrix['LoanAmount'])

# As there is no significance of Credit_History, Property_Area, Loan_Status
del data['Credit_History'], data['Property_Area'],data['Loan_Status']

# Splitting variables
features = data.drop('LoanAmount',axis=1)
target = data['LoanAmount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model creation
knn_regressor = KNeighborsRegressor()

# Define the parameter grid for grid search
param_grid = {'n_neighbors': range(2,8),
              'weights': ['uniform', 'distance'],
              'p': [1, 2]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(knn_regressor, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Evaluate the best estimator on the test set
y_pred = best_estimator.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)
print("Mean Squared Error:", mse)

# Using best estimators
model = KNeighborsRegressor(n_neighbors=7,p=1,weights='uniform')
model.fit(X_train,y_train)

yPred = model.predict(X_test)

# Model evaluation
print(r2_score(y_test,yPred))