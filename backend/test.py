from route_optimiser import *

file_path = 'backend/stats/fake_crime_data_north_singapore_1000.csv'
upper_right = (1.4700, 103.9500)
bottom_left = (1.3500, 103.8200)
severity_weight = 1.5
time_weight = 0.1
n_clusters = 3

rectangles, grid_coords, heatmap_values, cluster_labels, cluster_centers = get_cluster_rectangles(file_path, upper_right, bottom_left, severity_weight, time_weight, n_clusters)

plot_heatmap_and_clusters(grid_coords, heatmap_values, cluster_labels, cluster_centers, rectangles)
plot_osmnx_map_with_rectangles(grid_coords, rectangles)