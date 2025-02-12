import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Use all features for PCA
y = iris.target

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# Explained variance by each component
explained_variance = pca.explained_variance_ratio_

print("Explained variance by each component:")
for i, variance in enumerate(explained_variance, start=1):
    print(f"Principal Component {i}: {variance:.4f}")

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Iris dataset has 3 classes
kmeans.fit(X_pca)

# Get cluster assignments
assignments = kmeans.labels_
centers = kmeans.cluster_centers_

# Plotting
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']

for i in range(3):  # Since there are 3 clusters
    plt.scatter(X_pca[assignments == i, 0], X_pca[assignments == i, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.6)

# Plot cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='yellow', marker='X', s=200, label='Centroids')

plt.title('KMeans Clustering with PCA on Iris Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()
