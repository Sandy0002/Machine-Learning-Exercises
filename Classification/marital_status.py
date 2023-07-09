# In this program we will try to predict property area of loan sanction
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('loan_sanction_train.csv')
print(data.head())

# As Loan_id is not needed we remove it
del data['Loan_ID']

# Check for missing values
print(data.isna().sum())

'''We find that Gender, Dependents, Self_Employed, LoanAmount, LoanAmountTerm, CreditHistory
have missing values'''
# We will impute numerical data based on their loan approval status

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
goodCreditHistoryMean = data.loc[data['Loan_Status']=='Y']['Credit_History'].mean()
badCreditHistoryMean= data.loc[data['Loan_Status']=='N']['Credit_History'].mean()

# impute the loanAmount with mean values based on their loan statuses
data.loc[data['Loan_Status']=='Y','Credit_History'] = data.loc[data['Loan_Status']=='Y','Credit_History'].fillna(goodCreditHistoryMean)
data.loc[data['Loan_Status']=='N','Credit_History'] = data.loc[data['Loan_Status']=='N','Credit_History'].fillna(badCreditHistoryMean)

# Count of married and got loan
marriedCount = data[(data['Married']=="Yes") &
            (data['Loan_Status']=="Y")]['Married'].count()
unmarriedCount = data[(data['Married']=="No") &
            (data['Loan_Status']=="Y")]['Married'].count()
print(marriedCount,unmarriedCount)

# From above counts we find that people who are married have more chances to get loans so we impute yes
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

print(data.describe())
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

objectColumns =  ['Gender', 'Married', 'Education','Self_Employed', 'Property_Area', 'Loan_Status'] 

encoder = LabelEncoder()
# Label Encoding above columns
for col in objectColumns:
  data[col] = encoder.fit_transform(data[col])

  # Generating Correlation matrix
correlation_matrix = data.corr()

# Getting relation of the target variable with other variables
print(correlation_matrix['Married'])

# We will use heat map to get the information
plt.figure(figsize=(10, 6))
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

# There is some relation between Marital status, gender and dependents
# Splitting data
features = data.drop('Married',axis=1)
target = data['Married']
xTrain,xTest, yTrain, yTest = train_test_split(features,target,test_size=0.2,random_state=1)

# Model Building 
model = LogisticRegression(max_iter=600)

model.fit(xTrain,yTrain)
yPred = model.predict(xTest)

# Model Evaluation
print(confusion_matrix(yTest,yPred))
print(accuracy_score(yTest,yPred))