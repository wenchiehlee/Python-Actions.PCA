import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score

# Load the built-in Iris dataset
iris = load_iris()
X = iris.data  # Feature data (sepal length, sepal width, petal length, petal width)
y_true = iris.target  # Species labels
feature_names = iris.feature_names

# Create a DataFrame to store the original data and clustering results later
data_df = pd.DataFrame(X, columns=feature_names)
data_df['Species'] = [iris.target_names[label] for label in y_true]

# Standardize the data to avoid scaling issues
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check command-line arguments
if len(sys.argv) < 2 or sys.argv[1] not in ['1', '2']:
    print("Usage: python PCA.py [1|2] [Y|N]")
    print("1 - Use PCA for clustering and visualization with Species labels")
    print("2 - Perform K-means clustering with PCA")
    print("Optional 'Y' or 'N' for mode 1 to toggle data standardization")
    sys.exit(1)

mode = sys.argv[1]
standardize = sys.argv[2].lower() if len(sys.argv) == 3 else 'n'

if mode == '1':
    # Determine if data should be standardized based on user input
    if standardize == 'y':
        data_for_pca = X_scaled
        print("Data has been standardized.")
    else:
        data_for_pca = X
        print("Data has NOT been standardized.")

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data_for_pca)

    # Display the explained variance of each principal component
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by Principal Component 1: {explained_variance[0]:.4%}")
    print(f"Explained variance by Principal Component 2: {explained_variance[1]:.4%}")

    # Visualize the actual species labels
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue']
    species_names = iris.target_names

    for i, species in enumerate(species_names):
        plt.scatter(X_pca[y_true == i, 0], X_pca[y_true == i, 1],
                    alpha=0.7, color=colors[i], label=species)

    # Add titles and labels
    plt.title('PCA-based Visualization of Iris Dataset with Species Labels')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
    plt.legend()
    plt.grid(True)
    plt.show()

elif mode == '2':
    # Determine if data should be standardized based on the new parameter
    if standardize == 'y':
        data_for_pca = X_scaled
        print("Data has been standardized.")
    else:
        data_for_pca = X
        print("Data has NOT been standardized.")

    # Apply PCA to reduce the data to 2 dimensions for clustering
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data_for_pca)

    # Apply K-means clustering on the PCA-reduced data
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    # Add clustering results to DataFrame
    data_df['Cluster'] = clusters
    
    # Save the DataFrame with clustering results to a CSV file
    data_df.to_csv('iris_clustering_results.csv', index=False)
    print("Clustering results saved to 'iris_clustering_results.csv'")

    # Calculate the silhouette score to evaluate clustering quality
    silhouette_avg = silhouette_score(X_pca, clusters)
    print(f'Silhouette Score with PCA: {silhouette_avg:.3f}')

    # Compare clustering with true species labels
    ari_score = adjusted_rand_score(y_true, clusters)
    homogeneity = homogeneity_score(y_true, clusters)
    print(f'Adjusted Rand Index (ARI): {ari_score:.3f}')
    print(f'Homogeneity Score: {homogeneity:.3f}')

    # Display the explained variance of each principal component
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by Principal Component 1: {explained_variance[0]:.2%}")
    print(f"Explained variance by Principal Component 2: {explained_variance[1]:.2%}")

    # Visualize the K-means clustering results
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue']

    for i in range(3):
        plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1],
                    alpha=0.7, color=colors[i], label=f'Cluster {i+1}')

    # Plot the cluster centroids
    centers_pca = kmeans.cluster_centers_
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='yellow', marker='X', s=200, label='Centroids')

    # Add titles and labels
    plt.title('Unsupervised K-means Clustering with PCA on Iris Dataset')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
    plt.legend()
    plt.grid(True)
    plt.show()