# In this program we will determine customer categories by credit card usage
# We will create clusters of customer types
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


data = pd.read_csv('CC GENERAL.csv')
print(data.head())

# Customer_id is unnecessary
del data['CUST_ID']

# Checking for missing values
print(data.isna().sum())

# MINIMUM_PAYMENTS having missing values
ls = data[data['MINIMUM_PAYMENTS'].isna()]
print(ls.head())

# Trying to find relation of MINIMUM_PAYMENTS with others
data1 = data.dropna()
correlation_matrix = data1.corr()
print(correlation_matrix['MINIMUM_PAYMENTS'])

# There is no strong correlation with any other columns so we can go remove those rows
data =data.dropna()

# Checking for outliers
print(data.describe())

# Potential outlier columns
# BALANCE, PURCHASES, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES, CASH_ADVANCE
# CASH_ADVANCE_TRX, PURCHASES_TRX, CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS
sns.scatterplot(data['BALANCE'])

sns.scatterplot(data['PURCHASES'])

sns.scatterplot(data['ONEOFF_PURCHASES'])

sns.scatterplot(data['INSTALLMENTS_PURCHASES'])
maxPercentile = np.percentile(data['INSTALLMENTS_PURCHASES'], [99])

 # Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['INSTALLMENTS_PURCHASES']>25000,'INSTALLMENTS_PURCHASES']=upperValue

sns.scatterplot(data['CASH_ADVANCE'])
maxPercentile = np.percentile(data['CASH_ADVANCE'], [99])

 # Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['CASH_ADVANCE']>25000,'CASH_ADVANCE']=upperValue

sns.scatterplot(data['CASH_ADVANCE_TRX'])

sns.scatterplot(data['PURCHASES_TRX'])

sns.scatterplot(data['CREDIT_LIMIT'])

sns.scatterplot(data['PAYMENTS'])

sns.scatterplot(data['MINIMUM_PAYMENTS'])

maxPercentile = np.percentile(data['MINIMUM_PAYMENTS'], [99])

 # Getting values that fall under 1 percentile value
upperValue = maxPercentile[0]
data.loc[data['MINIMUM_PAYMENTS']>60000,'MINIMUM_PAYMENTS']=upperValue

# Scaling data
data = StandardScaler().fit_transform(data)

# Clusters creations
k_values = range(2, 10)
wcss = []  # Within-cluster sum of squares
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Plot the within-cluster sum of squares for each k value
plt.plot(k_values, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method: Optimal k')
plt.show()

# As there is steep decline at k=3 we will pass 3 as value for k
kmeans = KMeans(n_clusters=3, n_init=125)
kmeans.fit(data)

# Get the cluster labels
labels = kmeans.labels_
# Clusters evaluations
# Evaluate the segmentation using different metrics
silhouette = silhouette_score(data, labels)
calinski_harabasz = calinski_harabasz_score(data, labels)
davies_bouldin = davies_bouldin_score(data, labels)

print("Silhouette Score:", silhouette)
print("Calinski-Harabasz Score:", calinski_harabasz)
print("Davies-Bouldin Score:", davies_bouldin)