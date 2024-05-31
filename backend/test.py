import matplotlib.patches as patches
from heatmap import generate_heatmap
from generate_cluster import apply_kmeans_clustering, define_rectangles
import matplotlib.pyplot as plt



file_path = '/Users/hrishikeshsathyian/Documents/GitHub/LIfeHack2024/backend/stats/fake_crime_data_north_singapore_1000.csv'
upper_left = (1.4700, 103.8200)  # Example coordinates for the upper left corner
bottom_right = (1.3500, 103.9500)  # Example coordinates for the bottom right corner
# Generate the heatmap
grid_coords, heatmap = generate_heatmap(file_path, upper_left, bottom_right)

# Apply K-means clustering and define patrol areas
k = 8  # Number of patrol areas
cluster_labels, cluster_centers = apply_kmeans_clustering(grid_coords, heatmap, k)
rectangles = define_rectangles(grid_coords, cluster_labels, k)

# Plot the heat map with rectangles
fig, ax = plt.subplots()
ax.imshow(heatmap, origin='lower', extent=[grid_coords[:, 0].min(), grid_coords[:, 0].max(), grid_coords[:, 1].min(), grid_coords[:, 1].max()], cmap='hot', alpha=0.5)


plt.scatter(grid_coords[:, 0], grid_coords[:, 1], c='blue', s=1)
plt.colorbar()
plt.show()
