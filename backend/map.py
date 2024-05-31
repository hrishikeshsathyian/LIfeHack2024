import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box

upper_right = (1.3719192174008703, 103.90082210571713)
bottom_left = (1.349596156159813, 103.84209426225685)
# Define the coordinates
# Define the coordinates
latitude_ur, longitude_ur = upper_right
latitude_bl, longitude_bl = bottom_left

# Create a bounding box
bounding_box = box(longitude_bl, latitude_bl, longitude_ur, latitude_ur)

# Get a graph within the bounding box
G = ox.graph_from_polygon(bounding_box, network_type='all')

# Plot the graph
fig, ax = ox.plot_graph(G, show=False, close=False)

# Create a rectangle patch
rectangle = plt.Polygon([(longitude_bl, latitude_bl), 
                         (longitude_ur, latitude_bl), 
                         (longitude_ur, latitude_ur), 
                         (longitude_bl, latitude_ur)], 
                         closed=True, edgecolor='r', facecolor='none')
ax.add_patch(rectangle)

# Show the plot
plt.show()

