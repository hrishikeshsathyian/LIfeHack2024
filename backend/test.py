from route_optimiser import *

file_path = 'backend/stats/fake_crime_data_north_singapore_1000.csv'
upper_right = (1.4700, 103.9500)
bottom_left = (1.3500, 103.8200)
severity_weight = 1.5
time_weight = 0.1
n_clusters = 2

rectangles, grid_coords, heatmap_values, cluster_labels, cluster_centers = get_cluster_rectangles(file_path, upper_right, bottom_left, severity_weight, time_weight, n_clusters)

print("Cluster Rectangles Coordinates:")
for i, rect in enumerate(rectangles):
    print(f"Cluster {i + 1}: Upper Left {rect[0]}, Bottom Right {rect[1]}")
print(grid_coords)
plot_heatmap_and_clusters(grid_coords, heatmap_values, cluster_labels, cluster_centers, rectangles)
for rect in rectangles:
    print(rect)


import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import geopandas as gpd

import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import box
import geopandas as gpd


def plot_osmnx_map_with_rectangles(grid_coords, rectangles):
    # Define the bounding box for the map using the grid coordinates
    y_min, y_max = grid_coords[:, 0].min(), grid_coords[:, 0].max()  # latitudes
    x_min, x_max = grid_coords[:, 1].min(), grid_coords[:, 1].max()  # longitudes

    # Get the map data for the bounding box
    G = ox.graph_from_bbox(y_max, y_min, x_max, x_min, network_type='drive')

    # Create a plot of the map without nodes
    fig, ax = ox.plot_graph(G, node_size=0, show=False, close=False)

    # Plot each rectangle
    for rect in rectangles:
        bottom_right, upper_left = rect
        # Create a Polygon for the rectangle
        poly = box(upper_left[0], bottom_right[1], bottom_right[0], upper_left[1])  # box(minx, miny, maxx, maxy)
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[poly])
        # Plot the rectangle on the map
        gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)

    # Show the plot
    plt.show()



plot_osmnx_map_with_rectangles(grid_coords, rectangles)

