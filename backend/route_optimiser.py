import pandas as pd
from datetime import datetime
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import osmnx as ox
import networkx as nx
from shapely.geometry import box
import geopandas as gpd
import random
import os

def load_and_preprocess_data(file_path, severity_weight, time_weight):
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
    crime_hour = crime_time.hour
    time_diff_hours = (crime_hour - current_time.hour) % 24
    weight = np.exp(-time_diff_hours**2 / (2 * sigma**2))
    return weight

def filter_data_within_grid(crime_data, upper_right, bottom_left):
    return crime_data[
        (crime_data['latitude'] >= bottom_left[0]) & (crime_data['latitude'] <= upper_right[0]) &
        (crime_data['longitude'] >= bottom_left[1]) & (crime_data['longitude'] <= upper_right[1])
    ]

def generate_heatmap(crime_data, bandwidth=0.01):
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
    sample_weights = heatmap_values.ravel()
    mask = sample_weights > 0
    filtered_coords = grid_coords[mask]
    filtered_weights = sample_weights[mask]
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(filtered_coords, sample_weight=filtered_weights)
    return kmeans.labels_, kmeans.cluster_centers_

def define_rectangles(grid_coords, cluster_labels, k):
    rectangles = []
    for i in range(k):
        cluster_points = grid_coords[cluster_labels == i]
        lon_min, lat_min = cluster_points.min(axis=0)
        lon_max, lat_max = cluster_points.max(axis=0)
        upper_left = (lat_max, lon_min)
        bottom_right = (lat_min, lon_max)
        rectangles.append((upper_left, bottom_right))
    
    return rectangles

def plot_heatmap(grid_coords, heatmap_values, output_dir, plot_filename):
    x_min, x_max = grid_coords[:, 0].min(), grid_coords[:, 0].max()
    y_min, y_max = grid_coords[:, 1].min(), grid_coords[:, 1].max()
    
    plt.imshow(heatmap_values.reshape(100, 100), origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='hot', alpha=0.5)
    plt.colorbar()
    plt.title('Heatmap of Crime Intensity')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.savefig(os.path.join(output_dir, f"{plot_filename}_heatmap.png"))
    plt.close()

def plot_heatmap_and_clusters(grid_coords, heatmap_values, cluster_labels, cluster_centers, rectangles, output_dir, plot_filename):
    x_min, x_max = grid_coords[:, 0].min(), grid_coords[:, 0].max()
    y_min, y_max = grid_coords[:, 1].min(), grid_coords[:, 1].max()
    
    plt.imshow(heatmap_values.reshape(100, 100), origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='hot', alpha=0.5)
    plt.scatter(grid_coords[:, 0], grid_coords[:, 1], c=cluster_labels, s=1, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, marker='x')
    plt.title(f'K-means Clustering with Clusters')
    plt.colorbar()
    
    plt.savefig(os.path.join(output_dir, f"{plot_filename}_heatmap_clusters.png"))
    plt.close()
    
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
    plt.savefig(os.path.join(output_dir, f"{plot_filename}_rectangles.png"))
    plt.close()

def get_cluster_rectangles(file_path, upper_right, bottom_left, severity_weight=1.5, time_weight=0.1, n_clusters=2, output_dir="plot_images"):
    crime_data = load_and_preprocess_data(file_path, severity_weight, time_weight)
    crime_data = filter_data_within_grid(crime_data, upper_right, bottom_left)
    
    grid_coords, heatmap_values = generate_heatmap(crime_data)
    plot_heatmap(grid_coords, heatmap_values, output_dir, "crime_heatmap")  # Save heatmap

    cluster_labels, cluster_centers = apply_kmeans_clustering(grid_coords, heatmap_values, n_clusters)
    rectangles = define_rectangles(grid_coords, cluster_labels, n_clusters)
    
    plot_heatmap_and_clusters(grid_coords, heatmap_values, cluster_labels, cluster_centers, rectangles, output_dir, "cluster_heatmap")
    
    return rectangles, grid_coords, heatmap_values, cluster_labels, cluster_centers

def plot_osmnx_map_with_rectangles(grid_coords, rectangles, output_dir, plot_filename):
    y_min, y_max = grid_coords[:, 0].min(), grid_coords[:, 0].max()
    x_min, x_max = grid_coords[:, 1].min(), grid_coords[:, 1].max()
    
    G = ox.graph_from_bbox(north=y_max, south=y_min, east=x_max, west=x_min, network_type='drive')
    fig, ax = ox.plot_graph(G, node_size=0, show=False, close=False)
    
    for rect in rectangles:
        bottom_right, upper_left = rect
        poly = box(upper_left[0], bottom_right[1], bottom_right[0], upper_left[1])
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly])
        gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
    
    plt.savefig(os.path.join(output_dir, f"{plot_filename}_osmnx_map.png"))
    plt.close()
    
    return rectangles

def plot_osmnx_map_with_intensity_route(grid_coords, heatmap_values, rectangle, output_dir, plot_filename, min_nodes=10, max_nodes=50, intensity_threshold=0.6):
    (max_lon, min_lat), (min_lon, max_lat) = rectangle
    
    # Generate the map within the specified bounding box
    G = ox.graph_from_bbox(north=max_lat, south=min_lat, east=max_lon, west=min_lon, network_type='drive')
    
    # Extract the largest weakly connected component
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    
    fig, ax = ox.plot_graph(G, node_size=0, show=False, close=False)
    
    # Find nodes above the intensity threshold within the rectangle
    rect_mask = (
        (grid_coords[:, 0] >= min_lat) & (grid_coords[:, 0] <= max_lat) &
        (grid_coords[:, 1] >= min_lon) & (grid_coords[:, 1] <= max_lon)
    )
    rect_coords = grid_coords[rect_mask]
    rect_intensity = heatmap_values.reshape(-1)[rect_mask]
    
    intensity_threshold_value = np.percentile(rect_intensity, intensity_threshold * 100)
    high_intensity_mask = rect_intensity >= intensity_threshold_value
    high_intensity_coords = rect_coords[high_intensity_mask]
    
    # Limit the number of high-intensity nodes
    if len(high_intensity_coords) > max_nodes:
        high_intensity_indices = np.random.choice(len(high_intensity_coords), max_nodes, replace=False)
        high_intensity_coords = high_intensity_coords[high_intensity_indices]
    
    # Ensure we have a minimum number of nodes
    if len(high_intensity_coords) < min_nodes:
        additional_coords = rect_coords[~high_intensity_mask]
        additional_indices = np.random.choice(len(additional_coords), min_nodes - len(high_intensity_coords), replace=False)
        high_intensity_coords = np.vstack([high_intensity_coords, additional_coords[additional_indices]])
    
    # Find the nearest nodes in the graph to the selected high-intensity coordinates
    high_intensity_nodes = [ox.distance.nearest_nodes(G, xy[1], xy[0]) for xy in high_intensity_coords]
    
    # Ensure all selected nodes are connected
    manually_added_edges = []

    def ensure_connection(graph, nodes):
        for i in range(len(nodes) - 1):
            if not nx.has_path(graph, nodes[i], nodes[i + 1]):
                # If no path, add artificial connection and tag it
                graph.add_edge(nodes[i], nodes[i + 1], length=1, manual=True)
                manually_added_edges.append((nodes[i], nodes[i + 1]))
        return nodes

    high_intensity_nodes = ensure_connection(G, high_intensity_nodes)
    
    # Start from the first high-intensity node
    start_node = high_intensity_nodes[0]
    route = [start_node]
    
    def find_nearest_high_intensity_node(current_node, graph, candidate_nodes):
        current_coords = np.array([graph.nodes[current_node]['y'], graph.nodes[current_node]['x']])
        candidate_coords = np.array([[graph.nodes[node]['y'], graph.nodes[node]['x']] for node in candidate_nodes])
        dists = np.linalg.norm(candidate_coords - current_coords, axis=1)
        nearest_index = np.argmin(dists)
        return candidate_nodes[nearest_index], dists[nearest_index]
    
    # Create a loop route connecting all high-intensity nodes
    for next_node in high_intensity_nodes[1:]:
        if nx.has_path(G, route[-1], next_node):
            sub_route = nx.shortest_path(G, route[-1], next_node, weight='length')
            route.extend(sub_route[1:])
        else:
            # If no path, artificially connect the node and tag it
            G.add_edge(route[-1], next_node, length=1, manual=True)
            manually_added_edges.append((route[-1], next_node))
            route.append(next_node)
    
    # Attempt to close the loop by connecting back to the start node
    if nx.has_path(G, route[-1], start_node):
        final_leg = nx.shortest_path(G, route[-1], start_node, weight='length')
        route.extend(final_leg[1:])
    else:
        G.add_edge(route[-1], start_node, length=1, manual=True)
        manually_added_edges.append((route[-1], start_node))
        route.append(start_node)
    
    # Filter out manually added edges from the route
    filtered_route = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if 'manual' not in G[u][v] or not G[u][v]['manual']:
            filtered_route.append(u)
    filtered_route.append(route[-1])

    # Plot the route on the map excluding manually added edges
    if isinstance(G, nx.MultiDiGraph):
        edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if 'manual' not in d]
    else:
        edges = [(u, v) for u, v, d in G.edges(data=True) if 'manual' not in d]

    G_sub = G.edge_subgraph(edges).copy()  # Ensure to make a copy for plotting
    fig, ax = ox.plot_graph(G_sub, node_size=0, show=False, close=False)

    # Plot the filtered route
    ox.plot_graph_route(G, filtered_route, route_linewidth=6, node_size=0, bgcolor='k', ax=ax, route_color='red', show=False, close=False)
    
    # Plot manually added edges separately for verification
    manual_edge_coords = [(G.nodes[u]['y'], G.nodes[u]['x'], G.nodes[v]['y'], G.nodes[v]['x']) for u, v in manually_added_edges]
    for (y1, x1, y2, x2) in manual_edge_coords:
        ax.plot([x1, x2], [y1, y2], color='blue', linewidth=2, linestyle='--')  # Blue dashed lines for manual edges
    
    plt.savefig(os.path.join(output_dir, f"{plot_filename}_intensity_route.png"))
    plt.close()


# Define your file path where you want to save the plots
output_dir = 'plot_images'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
# Example usage
file_path = '/Users/hrishikeshsathyian/Documents/GitHub/LIfeHack2024/backend/stats/fake_crime_data_north_singapore_1000.csv'
upper_right = (1.4700, 103.9500)
bottom_left = (1.3500, 103.8200)
severity_weight = 1.5
time_weight = 0.1
n_clusters = 3
rectangles, grid_coords, heatmap_values, cluster_labels, cluster_centers = get_cluster_rectangles(file_path, upper_right, bottom_left, severity_weight, time_weight, n_clusters, output_dir)
rectangles = plot_osmnx_map_with_rectangles(grid_coords, rectangles, output_dir, "osmnx_map")

for i, rectangle in enumerate(rectangles):
    plot_osmnx_map_with_intensity_route(grid_coords, heatmap_values, rectangle, output_dir, f"osmnx_intensity_route_{i}")
