import pandas as pd
from datetime import datetime
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

def generate_heatmap(file_path, upper_right, bottom_left, severity_weight=1.5, time_weight=0.1):
    # Load the data from the CSV file
    crime_data = pd.read_csv(file_path, parse_dates=['time'])

    # Normalize the severity values and ensure time is in the correct format
    crime_data['severity'] = crime_data['severity'] / crime_data['severity'].max()
    crime_data['time'] = pd.to_datetime(crime_data['time'])

    # WeightingFunction to calculate weights based on the time difference between crime occurrences and the current time
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

    # Plot the heat map
    plt.imshow(z, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='hot', alpha=0.5)
    # Overlay the data points on the heat map
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=10)
    plt.colorbar()
    plt.show()

    return grid_coords, z

# Example usage with adjusted weights
file_path = 'backend/stats/fake_crime_data_north_singapore_1000.csv'
upper_right = (1.4700, 103.9500)  # Example coordinates for the upper right corner
bottom_left = (1.3500, 103.8200)  # Example coordinates for the bottom left corner

# Adjust the weights for severity and time
severity_weight = 1.5
time_weight = 0.1

grid_coords, z = generate_heatmap(file_path, upper_right, bottom_left, severity_weight, time_weight)
