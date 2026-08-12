import numpy as np

# Load the .npy file
data = np.load('results/refinement_times.npy')

# View the array's contents
print(data)

# Inspect the array's structure
print("Shape:", data.shape)
print("Data Type:", data.dtype)