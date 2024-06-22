import pandas as pd

data = pd.read_csv("Mall_Customers.csv") #Loading the dataset

print(data.head()) #Prints first few rows of the dataset
print(data.info()) #Prints information about the dataset

#Data Exploration and Preprocessing.

print(data.isnull().sum()) #Check for missing values
print(data.columns) #Check for column names

import matplotlib.pyplot as plt
import seaborn as sns

#Selecting features for Clustering.
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

plt.scatter(features['Annual Income (k$)'], features['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()

from sklearn.cluster import KMeans

#Elbow method used to determine the best value for K.
sse = []

for i in range(1, 11):
  km = KMeans(n_clusters = i)
  km.fit(features)
  sse.append(km.inertia_)

plt.plot(range(1,11), sse)
plt.xlabel('Number of Clusters')
plt.ylabel("SSE")
plt.show()

#Now we apply the K-means algorithm using the optimal value of k determined from the elbow method.
#Based on the elbow plot the optimal value of k is 5.

km= KMeans(n_clusters=5)
y_predicted = km.fit_predict(features)
data['Cluster'] = y_predicted
print(data.head())

km.cluster_centers_ #The centroids of the clusters

#Visualizing the clusters.

df1 = data[data.Cluster == 0]
df2 = data[data.Cluster == 1]
df3 = data[data.Cluster == 2]
df4 = data[data.Cluster == 3]
df5 = data[data.Cluster == 4]

# plt.scatter(df1['Annual Income (k$)'], df1['Spending Score (1-100)'], color = 'green')
# plt.scatter(df2['Annual Income (k$)'], df2['Spending Score (1-100)'], color = 'red')
# plt.scatter(df3['Annual Income (k$)'], df3['Spending Score (1-100)'], color = 'yellow')
# plt.scatter(df4['Annual Income (k$)'], df4['Spending Score (1-100)'], color = 'blue')
# plt.scatter(df5['Annual Income (k$)'], df5['Spending Score (1-100)'], color = 'black')

sns.scatterplot(x = "Annual Income (k$)", y = "Spending Score (1-100)", data = data, hue="Cluster")

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color="purple", marker="+")

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

