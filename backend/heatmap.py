import pandas as pd
from datetime import datetime
from sklearn.neighbors import KernelDensity
import numpy as np


# Load the data, normalize intensity and convert time to datetime
crime_data = pd.read_csv('backend/stats/fake_crime_data_north_singapore.csv')

crime_data['severity'] = crime_data['severity'] / 10
crime_data['time'] = pd.to_datetime(crime_data['time'])

# Extract latitude, longitude, and intensity
coords = crime_data[['latitude', 'longitude']].values
intensity = crime_data['severity'].values

# Fit KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(coords, sample_weight=intensity)


# Time weighting function with hour-based difference
def time_weighting(crime_time, current_time, sigma=1):
    crime_hour = crime_time.hour + crime_time.minute / 60
    current_hour = current_time.hour + current_time.minute / 60
    time_diff = abs(crime_hour - current_hour)
    print(f"Crime Hour: {crime_hour}, Current Hour: {current_hour}, Time Difference: {time_diff}")
    weight = np.exp(-time_diff**2 / (2 * sigma**2))
    print(f"Weight: {weight}")
    return weight

# Get the current time
current_time = datetime.now()

# Apply time weighting
crime_data['time_weight'] = crime_data['time'].apply(lambda x: time_weighting(x, current_time))
crime_data['weighted_intensity'] = crime_data['severity'] * crime_data['time_weight']

# Recalculate KDE with time-weighted intensities
coords = crime_data[['latitude', 'longitude']].values
weighted_intensity = crime_data['weighted_intensity'].values
# Remove any zero weights
crime_data = crime_data[crime_data['weighted_intensity'] > 0]

kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(coords, sample_weight=weighted_intensity)

# Generate heat map grid
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Evaluate KDE on grid
z = np.exp(kde.score_samples(grid_coords))
z = z.reshape(x_grid.shape)

# Plot heat map
import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='hot', alpha=0.5)
plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=10)
plt.colorbar()
plt.show()


