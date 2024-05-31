import pandas as pd
from datetime import datetime
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_heatmap(file_path, upper_right, bottom_left, severity_weight=1.5, time_weight=0.1):
    # Load the data from the CSV file
    crime_data = pd.read_csv(file_path, parse_dates=['time'])

    # Normalize the severity values and ensure time is in the correct format
    crime_data['severity'] = crime_data['severity'] / crime_data['severity'].max()
    crime_data['time'] = pd.to_datetime(crime_data['time'])

    # Weighting function to calculate weights based on the time difference between crime occurrences and the current time
    def time_weighting(crime_time, current_time, sigma=2.0):
        crime_hour = crime_time.hour
        time_diff_hours = (crime_hour - current_time.hour) % 24
        weight = np.exp(-time_diff_hours**2 / (2 * sigma**2))
        return weight

    # Get the current time
    current_time = datetime.now()

    # Apply the time-based weighting to each crime's time
    crime_data['time_weight'] = crime_data['time'].apply(lambda x: time_weighting(x, current_time) * time_weight)
    # Calculate the weighted intensity by multiplying severity with the time weight
    crime_data['weighted_intensity'] = crime_data['severity'] * severity_weight * crime_data['time_weight']

    # Filter the data within the given rectangular grid using upper right and bottom left coordinates
    crime_data = crime_data[
        (crime_data['latitude'] >= bottom_left[0]) & (crime_data['latitude'] <= upper_right[0]) &
        (crime_data['longitude'] >= bottom_left[1]) & (crime_data['longitude'] <= upper_right[1])
    ]

    # Remove any crimes with zero weights (no contribution to the weighted intensity)
    crime_data = crime_data[crime_data['weighted_intensity'] > 0]

    # Extract coordinates and weighted intensities for the entire data
    coords = crime_data[['latitude', 'longitude']].values
    weighted_intensity = crime_data['weighted_intensity'].values

    # Fit a Kernel Density Estimation (KDE) model on the data
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(coords, sample_weight=weighted_intensity)

    # Generate a grid for the heat map
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Evaluate the KDE model on the grid coordinates to get density values
    z = np.exp(kde.score_samples(grid_coords))
    z = z.reshape(x_grid.shape)

    return grid_coords, z

def apply_kmeans_clustering(grid_coords, heatmap_values, k):
    # Flatten the heatmap values for use as sample weights
    sample_weights = heatmap_values.ravel()

    # Filter out zero weights
    mask = sample_weights > 0
    filtered_coords = grid_coords[mask]
    filtered_weights = sample_weights[mask]

    # Apply K-means clustering to the grid coordinates with sample weights
    kmeans = KMeans(n_clusters=k, random_state=0).fit(filtered_coords, sample_weight=filtered_weights)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    return cluster_labels, cluster_centers

# Define rectangles based on cluster labels
def define_rectangles(grid_coords, cluster_labels, k):
    rectangles = []
    for i in range(k):
        cluster_points = grid_coords[cluster_labels == i]
        lon_min, lat_min = cluster_points.min(axis=0)  # Swap lat and lon
        lon_max, lat_max = cluster_points.max(axis=0)  # Swap lat and lon
        upper_left = (lat_max, lon_min)
        bottom_right = (lat_min, lon_max)
        rectangles.append((upper_left, bottom_right))
    
    return rectangles

# Plot heat map and rectangles on the map (optional for visualization)
def plot_heatmap_and_clusters(grid_coords, heatmap_values, cluster_labels, cluster_centers, rectangles):
    x_min, x_max = grid_coords[:, 0].min(), grid_coords[:, 0].max()
    y_min, y_max = grid_coords[:, 1].min(), grid_coords[:, 1].max()
    
    # Plot the heat map
    plt.imshow(heatmap_values.reshape(100, 100), origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='hot', alpha=0.5)
    plt.scatter(grid_coords[:, 0], grid_coords[:, 1], c=cluster_labels, s=1, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, marker='x')
    plt.title(f'K-means Clustering with Clusters')
    plt.colorbar()
    
    fig, ax = plt.subplots()
    for rect in rectangles:
        upper_left, bottom_right = rect
        width = bottom_right[1] - upper_left[1]
        height = upper_left[0] - bottom_right[0]
        rectangle = patches.Rectangle((upper_left[1], bottom_right[0]), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rectangle)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.title('Patrol Area Rectangles')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show()

def get_cluster_rectangles(file_path, upper_right, bottom_left, severity_weight=1.5, time_weight=0.1, n_clusters=2):
    grid_coords, heatmap_values = generate_heatmap(file_path, upper_right, bottom_left, severity_weight, time_weight)
    
    # Flatten the grid coordinates and intensities for clustering
    flat_coords = grid_coords
    flat_intensities = heatmap_values.flatten()

    # Filter out zero intensities for clustering
    mask = flat_intensities > 0
    flat_coords = flat_coords[mask]
    flat_intensities = flat_intensities[mask]

    cluster_labels, cluster_centers = apply_kmeans_clustering(flat_coords, flat_intensities, n_clusters)
    rectangles = define_rectangles(flat_coords, cluster_labels, n_clusters)
    
    return rectangles, grid_coords, heatmap_values, cluster_labels, cluster_centers

