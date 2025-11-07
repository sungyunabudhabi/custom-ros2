import numpy as np
import matplotlib.pyplot as plt
import os

# Specify path to .npy file
# Call file name
npy_file_path = "reinvent_base.npy"
npy_custom_file_path = "reinvent_base-400-4-2019-10-11-161903.npy"

# Load waypoints as array
track_arr = np.load(npy_file_path)

# The array typically contains coordinates for the center, left, and right sides of the track
# The structure is usually [waypoint_index, x_coordinate, y_coordinate, ...], 
# but for visualization, we just need the x and y coordinates.
# The common structure for DeepRacer track npy files is a 2D array where 
# the first column is the waypoint index (sometimes omitted or implicit), 
# and subsequent columns are x, y coordinates.
# A common format is: [index, center_x, center_y, left_x, left_y, right_x, right_y, ...]. 
# Adjust indexing if your file format is different.

# Assuming a simple 2D array of [x, y] points for the center line:
# If the file has multiple columns (e.g., center, left, right), you would slice it differently.
# A typical DeepRacer track npy file will have columns for (center_x, center_y), (left_x, left_y), (right_x, right_y).

# Switch original waypoint to custom waypoint
track_arr[:, [0, 1]] = np.load(npy_custom_file_path)

# Example for plotting all three lines:
center_points = track_arr[:, [0, 1]]
left_points = track_arr[:, [2, 3]]
right_points = track_arr[:, [4, 5]]

# Specify filename for the .npy file
filename = "reinvent_base_track_custom.npy"

# Save the modified track array to a new .npy file
np.save(filename, track_arr)
print(f"Modified track waypoints saved to {filename}")

# Plot the waypoints
plt.figure(figsize=(10, 8))
plt.scatter(center_points[:, 0], center_points[:, 1], color='green', s=5, label='Center Line')
plt.scatter(left_points[:, 0], left_points[:, 1], color='red', s=2, label='Left Border')
plt.scatter(right_points[:, 0], right_points[:, 1], color='red', s=2, label='Right Border')
plt.title(f"DRFC Track Visualization: {os.path.basename(npy_file_path)}")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.gca().set_aspect('equal', adjustable='box') # Keep the aspect ratio equal so the track isn't skewed
plt.grid(True)
plt.show()

# You can also print the raw array to inspect the data in the terminal
# print(track_arr) 
# print(f"Data shape: {track_arr.shape}")
