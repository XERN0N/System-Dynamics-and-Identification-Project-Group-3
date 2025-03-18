import numpy as np
import pandas as pd
import os
import re
import tkinter as tk
from tkinter import filedialog

# Hide the root window
root = tk.Tk()
root.withdraw()

# Select directory containing the data files
data_folder_path = filedialog.askdirectory(title="Select Folder Containing Data Files")
file_list = os.listdir(data_folder_path)

# Initialize lists for storing data
plusX, minusX, plusY, minusY = [], [], [], []

for file in file_list:
    file_path = os.path.join(data_folder_path, file)
    print(f"Processing file: {file_path}")
    
    try:
        # Read file assuming no header
        data = pd.read_csv(file_path, header=None, dtype=str)
        
        # Extract first three rows (Description part)
        description = data.iloc[:3].fillna('').astype(str)
        
        # Remove first three rows and reset index
        data = data.iloc[3:].reset_index(drop=True)
        data = data.iloc[:-2]
        
        # Ensure valid rows (length 12)
        data = data[data[0].str.len() == 12].astype(str)
        
        # Split 12-digit numbers into 4 columns
        data = data[0].apply(lambda x: [x[i:i+3] for i in range(0, 12, 3)])
        data = pd.DataFrame(data.tolist(), columns=["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"])
        # Convert to numeric
        data = data.astype(float)
        
        #print(data) debug for data

        # Classify files based on names
        if "posX" in file:
            plusX.append(data.values)
        elif "negX" in file:
            minusX.append(data.values)
        elif "posY" in file:
            plusY.append(data.values)
        elif "negY" in file:
            minusY.append(data.values)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Convert lists to numpy arrays
plusX = np.vstack(plusX) if plusX else np.zeros((100, 4))
minusX = np.vstack(minusX) if minusX else np.zeros((100, 4))
plusY = np.vstack(plusY) if plusY else np.zeros((100, 4))
minusY = np.vstack(minusY) if minusY else np.zeros((100, 4))
plusZ = np.zeros_like(plusX)
minusZ = np.zeros_like(minusX)

# Define the starting index for data extraction
n0 = 10

# Extract relevant columns (assuming 2nd, 3rd, and 4th columns contain acceleration data)
a_plus = np.column_stack((plusX[n0:, 0], plusY[n0:, 0], plusZ[n0:, 0]))
a_minus = np.column_stack((minusX[n0:, 0], minusY[n0:, 0], minusZ[n0:, 0]))
#print(a_minus)
# Initialize arrays for sensitivity, bias, and noise standard deviation
sens = np.zeros(3)
bias = np.zeros(3)
noise_std = np.zeros(3)

# Gravity constant in m/s²
g = 9.81

# Compute sensitivity, bias, and noise standard deviation for each axis (X, Y, Z)
for i in range(3):
    plus_g = np.mean(a_plus[:, i])
    minus_g = np.mean(a_minus[:, i])
    
    # Sensitivity in (m/s²)/LSB
    sens[i] = 2 * g / (plus_g - minus_g) if (plus_g - minus_g) != 0 else 0
    
    # Bias computation
    bias[i] = 0.5 * (plus_g + minus_g)
    
    # Compute acceleration using sensitivity and bias
    ac_plus = (a_plus[:, i] - bias[i]) * sens[i]
    ac_minus = (a_minus[:, i] - bias[i]) * sens[i]
    
    # Compute standard deviation of noise
    noise_std[i] = 0.5 * (np.std(ac_plus) + np.std(ac_minus))

# Store results in a DataFrame
calibAcel = pd.DataFrame({
    "Sensitivity (m/s²/LSB)": sens,
    "Bias": bias,
    "Noise Std (m/s²)": noise_std
})

# Save to text file
calibAcel.to_csv("calibAcel.txt", sep=",", index=False, float_format="%.6f")

print("Calibration data saved to calibAcel.txt")
