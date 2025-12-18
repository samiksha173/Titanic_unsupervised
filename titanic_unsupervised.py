# ==========================================
# Titanic Unsupervised Learning
# K-Means Clustering + PCA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------
df = pd.read_csv("Titanic_data.csv")

print("Initial Dataset Shape:", df.shape)

# ------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------
df = df.drop_duplicates()

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop non-numeric / irrelevant columns
df = df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ------------------------------------------
# 3. FEATURE ENGINEERING & ENCODING
# ------------------------------------------
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = df.drop(columns=['SibSp', 'Parch'])

print("\nDataset After Encoding:")
print(df.head())

# ------------------------------------------
# 4. FEATURE SCALING
# ------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ------------------------------------------
# 5. K-MEANS CLUSTERING
# ------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

print("\nCluster Distribution:")
print(df['Cluster'].value_counts())

# ------------------------------------------
# 6. PCA (DIMENSIONALITY REDUCTION)
# ------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio (PCA):")
print(pca.explained_variance_ratio_)

# ------------------------------------------
# 7. PCA VISUALIZATION
# ------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters on Titanic Dataset (PCA Reduced)")
plt.show()

# ------------------------------------------
# 8. SAVE FINAL OUTPUT
# ------------------------------------------
df.to_csv("titanic_unsupervised_clusters.csv", index=False)
print("\nClustered dataset saved as titanic_unsupervised_clusters.csv")
