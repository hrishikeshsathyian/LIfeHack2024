import pandas as pd
from datetime import datetime
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def generate_heatmap(file_path, upper_right, bottom_left, severity_weight=1.0, time_weight=1.0):
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

    # Split the data into training (before 2024-01-01) and testing sets (from 2024-01-01 onwards)
    train_data = crime_data[crime_data['time'] < '2024-01-01']
    test_data = crime_data[crime_data['time'] >= '2024-01-01']

    # Extract coordinates and weighted intensities for the training data
    coords_train = train_data[['latitude', 'longitude']].values
    weighted_intensity_train = train_data['weighted_intensity'].values

    # Fit a Kernel Density Estimation (KDE) model on the training data
    kde_train = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(coords_train, sample_weight=weighted_intensity_train)

    # Extract coordinates for the testing data
    coords_test = test_data[['latitude', 'longitude']].values

    # Evaluate the KDE model on the testing data coordinates
    log_density_test = kde_train.score_samples(coords_test)
    density_test = np.exp(log_density_test)

    # Normalize the predicted densities to match the scale of actual weighted intensities
    density_test *= weighted_intensity_train.sum() / density_test.sum()

    # Calculate the Mean Squared Error (MSE) between the predicted density and actual weighted intensity
    mse = mean_squared_error(test_data['weighted_intensity'], density_test)
    print(f"Mean Squared Error: {mse}")

    # Generate a grid for the heat map
    x_min, x_max = coords_train[:, 0].min(), coords_train[:, 0].max()
    y_min, y_max = coords_train[:, 1].min(), coords_train[:, 1].max()
    x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Evaluate the KDE model on the grid coordinates to get density values
    z = np.exp(kde_train.score_samples(grid_coords))
    z = z.reshape(x_grid.shape)

    # Plot the heat map
    plt.imshow(z, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='hot', alpha=0.5)
    # Overlay the training data points on the heat map
    plt.scatter(coords_train[:, 0], coords_train[:, 1], c='blue', s=10)
    plt.colorbar()
    plt.show()

    # Baseline calculation for normalized severity
    average_severity_normalized = crime_data['severity'].mean()
    crime_data['baseline_prediction'] = average_severity_normalized
    baseline_mse_normalized = mean_squared_error(crime_data['severity'], crime_data['baseline_prediction'])

    print(f"Normalized Baseline Mean Squared Error: {baseline_mse_normalized}")

    return z, x_grid, y_grid

# Example usage with adjusted weights
file_path = 'backend/stats/fake_crime_data_north_singapore_1000.csv'
upper_right = (1.4700, 103.9500)  # Example coordinates for the upper right corner
bottom_left = (1.3500, 103.8200)  # Example coordinates for the bottom left corner

# Adjust the weights for severity and time
severity_weight = 1.5
time_weight = 0.1

heatmap, x_grid, y_grid = generate_heatmap(file_path, upper_right, bottom_left, severity_weight, time_weight)
