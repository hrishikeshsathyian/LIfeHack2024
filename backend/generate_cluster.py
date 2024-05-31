from sklearn.cluster import KMeans
import pandas as pd
from datetime import datetime
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def apply_kmeans_clustering(grid_coords, heatmap_values, k):
    # Flatten the heatmap values for use as sample weights
    sample_weights = heatmap_values.ravel()

    # Apply K-means clustering to the grid coordinates with sample weights
    kmeans = KMeans(n_clusters=k, random_state=0).fit(grid_coords, sample_weight=sample_weights)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Plot the clusters on the heat map
    plt.imshow(heatmap_values, origin='lower', extent=[grid_coords[:, 0].min(), grid_coords[:, 0].max(), grid_coords[:, 1].min(), grid_coords[:, 1].max()], cmap='hot', alpha=0.5)
    plt.scatter(grid_coords[:, 0], grid_coords[:, 1], c=cluster_labels, s=1, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, marker='x')
    plt.title(f'K-means Clustering with {k} Clusters')
    plt.colorbar()
    plt.show()

    return cluster_labels, cluster_centers

# Define rectangles based on cluster labels
def define_rectangles(grid_coords, cluster_labels, k):
    rectangles = []
    for i in range(k):
        cluster_points = grid_coords[cluster_labels == i]
        lat_min, lon_min = cluster_points.min(axis=0)
        lat_max, lon_max = cluster_points.max(axis=0)
        upper_left = (lat_max, lon_min)
        bottom_right = (lat_min, lon_max)
        rectangles.append((upper_left, bottom_right))
        
    
    return rectangles


