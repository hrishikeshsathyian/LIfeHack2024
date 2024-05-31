import pandas as pd
from datetime import datetime
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import osmnx as ox
from shapely.geometry import box
import geopandas as gpd

def load_and_preprocess_data(file_path, severity_weight, time_weight):
    """
    Load and preprocess crime data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing crime data.
        severity_weight (float): Weight applied to the severity of crimes.
        time_weight (float): Weight applied to the time of crimes.
    
    Returns:
        pd.DataFrame: Preprocessed crime data with weighted intensities.
    """
    try:
        crime_data = pd.read_csv(file_path, parse_dates=['time'])
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    
    crime_data['severity'] = crime_data['severity'] / crime_data['severity'].max()
    crime_data['time'] = pd.to_datetime(crime_data['time'])
    
    current_time = datetime.now()
    crime_data['time_weight'] = crime_data['time'].apply(lambda x: time_weighting(x, current_time) * time_weight)
    crime_data['weighted_intensity'] = crime_data['severity'] * severity_weight * crime_data['time_weight']
    
    return crime_data

def time_weighting(crime_time, current_time, sigma=2.0):
    """
    Calculate the time-based weight for a crime occurrence.
    
    Args:
        crime_time (datetime): Time of the crime occurrence.
        current_time (datetime): Current time.
        sigma (float): Standard deviation for the Gaussian function.
    
    Returns:
        float: Calculated time-based weight.
    """
    crime_hour = crime_time.hour
    time_diff_hours = (crime_hour - current_time.hour) % 24
    weight = np.exp(-time_diff_hours**2 / (2 * sigma**2))
    return weight

def filter_data_within_grid(crime_data, upper_right, bottom_left):
    """
    Filter crime data within the given rectangular grid.
    
    Args:
        crime_data (pd.DataFrame): Crime data with latitude and longitude columns.
        upper_right (tuple): Coordinates of the upper right corner of the grid.
        bottom_left (tuple): Coordinates of the bottom left corner of the grid.
    
    Returns:
        pd.DataFrame: Filtered crime data within the grid.
    """
    return crime_data[
        (crime_data['latitude'] >= bottom_left[0]) & (crime_data['latitude'] <= upper_right[0]) &
        (crime_data['longitude'] >= bottom_left[1]) & (crime_data['longitude'] <= upper_right[1])
    ]

def generate_heatmap(crime_data, bandwidth=0.01):
    """
    Generate a heat map using Kernel Density Estimation (KDE).
    
    Args:
        crime_data (pd.DataFrame): Preprocessed crime data with weighted intensities.
        bandwidth (float): Bandwidth for the KDE model.
    
    Returns:
        np.ndarray: Grid coordinates.
        np.ndarray: Heat map values.
    """
    coords = crime_data[['latitude', 'longitude']].values
    weighted_intensity = crime_data['weighted_intensity'].values
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(coords, sample_weight=weighted_intensity)
    
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    
    z = np.exp(kde.score_samples(grid_coords))
    z = z.reshape(x_grid.shape)
    
    return grid_coords, z

def apply_kmeans_clustering(grid_coords, heatmap_values, k):
    """
    Apply K-means clustering to the grid coordinates with sample weights.
    
    Args:
        grid_coords (np.ndarray): Grid coordinates.
        heatmap_values (np.ndarray): Heat map values.
        k (int): Number of clusters.
    
    Returns:
        np.ndarray: Cluster labels.
        np.ndarray: Cluster centers.
    """
    sample_weights = heatmap_values.ravel()
    mask = sample_weights > 0
    filtered_coords = grid_coords[mask]
    filtered_weights = sample_weights[mask]
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(filtered_coords, sample_weight=filtered_weights)
    return kmeans.labels_, kmeans.cluster_centers_

def define_rectangles(grid_coords, cluster_labels, k):
    """
    Define rectangles based on cluster labels.
    
    Args:
        grid_coords (np.ndarray): Grid coordinates.
        cluster_labels (np.ndarray): Cluster labels.
        k (int): Number of clusters.
    
    Returns:
        list: List of rectangles defined by upper left and bottom right coordinates.
    """
    rectangles = []
    for i in range(k):
        cluster_points = grid_coords[cluster_labels == i]
        lon_min, lat_min = cluster_points.min(axis=0)
        lon_max, lat_max = cluster_points.max(axis=0)
        upper_left = (lat_max, lon_min)
        bottom_right = (lat_min, lon_max)
        rectangles.append((upper_left, bottom_right))
    
    return rectangles

def plot_heatmap_and_clusters(grid_coords, heatmap_values, cluster_labels, cluster_centers, rectangles):
    """
    Plot heat map and clusters on the map.
    
    Args:
        grid_coords (np.ndarray): Grid coordinates.
        heatmap_values (np.ndarray): Heat map values.
        cluster_labels (np.ndarray): Cluster labels.
        cluster_centers (np.ndarray): Cluster centers.
        rectangles (list): List of rectangles defined by clusters.
    """
    x_min, x_max = grid_coords[:, 0].min(), grid_coords[:, 0].max()
    y_min, y_max = grid_coords[:, 1].min(), grid_coords[:, 1].max()
    
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
    """
    Get rectangles defined by clustering on crime data.
    
    Args:
        file_path (str): Path to the CSV file containing crime data.
        upper_right (tuple): Coordinates of the upper right corner of the grid.
        bottom_left (tuple): Coordinates of the bottom left corner of the grid.
        severity_weight (float): Weight applied to the severity of crimes.
        time_weight (float): Weight applied to the time of crimes.
        n_clusters (int): Number of clusters.
    
    Returns:
        tuple: Rectangles, grid coordinates, heat map values, cluster labels, and cluster centers.
    """
    crime_data = load_and_preprocess_data(file_path, severity_weight, time_weight)
    crime_data = filter_data_within_grid(crime_data, upper_right, bottom_left)
    
    grid_coords, heatmap_values = generate_heatmap(crime_data)
    cluster_labels, cluster_centers = apply_kmeans_clustering(grid_coords, heatmap_values, n_clusters)
    rectangles = define_rectangles(grid_coords, cluster_labels, n_clusters)
    
    return rectangles, grid_coords, heatmap_values, cluster_labels, cluster_centers

def plot_osmnx_map_with_rectangles(grid_coords, rectangles):
    """
    Plot rectangles on a map using OSMnx.
    
    Args:
        grid_coords (np.ndarray): Grid coordinates.
        rectangles (list): List of rectangles defined by clusters.
    """
    y_min, y_max = grid_coords[:, 0].min(), grid_coords[:, 0].max()
    x_min, x_max = grid_coords[:, 1].min(), grid_coords[:, 1].max()
    
    G = ox.graph_from_bbox(y_max, y_min, x_max, x_min, network_type='drive')
    fig, ax = ox.plot_graph(G, node_size=0, show=False, close=False)
    
    for rect in rectangles:
        bottom_right, upper_left = rect
        poly = box(upper_left[0], bottom_right[1], bottom_right[0], upper_left[1])
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly])
        gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
        print(rect)
    
    plt.show()


