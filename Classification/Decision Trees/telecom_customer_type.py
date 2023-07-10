import pandas as pd
import sklearn as sk
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Loading the data
data = pd.read_csv('customer_service.csv',encoding='utf-8')

''' Analyzing the given data'''
# print(data.info())
# print(data['custcat'].value_counts()) #important to get the counts
# print(data.describe())
# print(data.isnull().sum()) no null values
# print(data.columns)

x = data.drop('custcat', axis=1)
y = data['custcat']

plt.style.use('ggplot') # makes somewhat good

# sns.FacetGrid(data, hue="custcat").map(plt.scatter, "address", "region").add_legend()  # can be useful
# plt.show()

# finding correlation between between parameters
plt.figure(figsize=(5, 5))
cor = data.corr()
top_corr = cor.index
g = sns.heatmap(data[top_corr].corr(), annot=True, cmap='RdYlGn')
plt.show()

# from the heatmap we take following parameters
obs = data[['age', 'address','employ','custcat']]
another_obs = data[['tenure', 'age', 'address', 'employ', 'income', 'custcat']]

sns.set(style='ticks')
sns.pairplot(another_obs,hue='custcat')
plt.show()

# Model Building
scalr = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.2,random_state=0)

model = RandomForestClassifier(n_estimators=525, criterion='entropy', max_depth=9, random_state=0)
model.fit(x, y)

# metrics to evaluate model
print(accuracy_score(y_test, model.predict(x_test)))
print(confusion_matrix(y_test, model.predict(x_test)))

# plotting importances
(pd.Series(model.feature_importances_, index=x.columns).nlargest(x.shape[1]).plot(kind='bar', color='green'))
plt.show()
