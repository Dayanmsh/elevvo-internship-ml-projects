"""
Task 2: Customer Segmentation

Goal: Cluster customers into segments based on income and spending score using unsupervised learning.
Enhancements: Includes optimal cluster selection, DBSCAN, spending analysis, and detailed explanations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Load real dataset
import os
csv_path = os.path.join(os.path.dirname(__file__), 'Mall_Customers.csv')
data = pd.read_csv(csv_path)

# Rename columns for convenience
data = data.rename(columns={
    'Annual Income (k$)': 'income',
    'Spending Score (1-100)': 'spending_score'
})

# Data exploration
print("Data Head:\n", data.head())
print("Data Description:\n", data.describe())
print("\nColumns:", data.columns.tolist())

# Visualize relationship between income and spending score
plt.scatter(data['income'], data['spending_score'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Income vs Spending Score')
plt.show()

# Visualize distribution of age and gender
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
data['Age'].hist(bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.subplot(1,2,2)
data['Gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Feature selection and scaling
features = ['income', 'spending_score']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optimal number of clusters (Elbow method)
inertia = []
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.subplot(1,2,2)
plt.plot(K_range, silhouette_scores, marker='o', color='orange')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by k')
plt.tight_layout()
plt.show()

# Choose optimal k (e.g., 5 based on elbow/silhouette)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
score = silhouette_score(X_scaled, labels)
print(f"KMeans Silhouette Score: {score:.2f}")

# Visualize clusters
plt.scatter(data['income'], data['spending_score'], c=labels, cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title(f'KMeans Clusters (k={optimal_k})')
plt.show()

# Bonus: DBSCAN clustering
# DBSCAN is good for arbitrary-shaped clusters and outlier detection
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

plt.scatter(data['income'], data['spending_score'], c=dbscan_labels, cmap='plasma')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clusters')
plt.show()

# Bonus: Analyze average spending per cluster
cluster_df = data.copy()
cluster_df['cluster'] = labels
avg_spending = cluster_df.groupby('cluster')['spending_score'].mean()
print("Average Spending per Cluster:\n", avg_spending)

# Save results and explanations
with open('results.txt', 'w') as f:
    f.write(f"KMeans Silhouette Score: {score:.2f}\n")
    f.write(f"Optimal k: {optimal_k}\n")
    f.write("Average Spending per Cluster:\n")
    f.write(str(avg_spending))
    f.write("\nExplanations:\n")
    f.write("- KMeans clusters customers based on similarity in income and spending score.\n")
    f.write("- The Elbow and Silhouette methods help select the optimal number of clusters.\n")
    f.write("- DBSCAN can find clusters of arbitrary shape and detect outliers.\n")
    f.write("- Analyzing average spending per cluster helps understand customer segments.\n")
