# In this program we will create segments of customers of a mall
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import get_scorer_names

data = pd.read_csv('Mall_Customers.csv')
print(data.head())

# CustomerID can be removed
del data['CustomerID']

data['Gender']=data['Genre']
del data['Genre']

# Missing value check
print(data.isna().sum())

# Categorical data conversion
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])

dbscan = DBSCAN(eps=0.5,min_samples=6)
dbscan.fit(X_pca)
labels = dbscan.labels_

silhouette = silhouette_score(data, labels)
calinski_harabasz = calinski_harabasz_score(data, labels)
davies_bouldin = davies_bouldin_score(data, labels)

print("Silhouette Score:", silhouette)
print("Calinski-Harabasz Score:", calinski_harabasz)
print("Davies-Bouldin Score:", davies_bouldin)
