# PRODIGY_ML_02
# K-Means Clustering of Retail Store Customers

This project uses the K-means clustering algorithm to group retail store customers based on their purchase history. The primary goal is to segment the customers into different groups to understand their behaviour better and potentially tailor marketing strategies accordingly.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Theory](#theory)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Introduction

K-means clustering is an unsupervised machine learning algorithm used to partition data into K distinct clusters based on feature similarity. In this project, we use K-means to group customers based on their annual income and spending score.

## Dataset

The dataset used for this project is `Mall_Customers.csv`, which contains the following columns:

- **CustomerID**: Unique ID for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income (k$)**: Annual income of the customer in thousands of dollars
- **Spending Score (1-100)**: Score assigned by the store based on customer behavior and spending nature

## Project Structure

The project is organized as follows:

- `Mall_Customers.csv`: Dataset file
- `kmeans_clustering.py`: Main script for performing K-means clustering
- `README.md`: Project documentation (this file)

## Requirements

- Python 3.6 or higher
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/nabin-ai/PRODIGY_ML_02.git
    cd PRODIGY_ML_02
    ```

2. Install the required packages:
    ```sh
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3. Place the `Mall_Customers.csv` file in the project directory.

## Theory: K-means Clustering
K-means clustering aims to partition data into K clusters such that each data point belongs to the cluster with the nearest mean. 
The algorithm follows these steps:
1. Initialization: Select K initial centroids randomly.
2. Assignment: Assign each data point to the nearest centroid.
3. Update: Calculate new centroids as the mean of all data points in each cluster.
4. Repeat: Repeat the assignment and update steps until convergence.

## Usage
Run the following script to train the model and evaluate its performance:
```python
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
```

## Conclusion
After running the K-means clustering algorithm, we visualize the customer segments and analyze the clusters. Each cluster represents a group of customers with similar annual incomes and spending scores. These insights can be used to tailor marketing strategies and improve customer satisfaction.

## Acknowledgements
The dataset used in this project is publicly available and provided by the retail store for educational purposes. Special thanks to the creators of the dataset.


