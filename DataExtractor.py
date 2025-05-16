import pandas as pd
import numpy as np
import os as os
import re
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

Calibration_path = os.getcwd()
Calibration_path = os.path.join(Calibration_path, "calibAcel.txt")
if os.path.exists(Calibration_path) == False:
    print("Calibration file not found")
else:
    print("Calibration file found")

try:
    Calibration = pd.read_csv(Calibration_path, dtype=float, delimiter=",")
except Exception as e:
    print(f"Calibration file csv error at {Calibration_path}: \n {e}")

# Get data folder path
data_folder_path = filedialog.askdirectory()  # selecting the directory for the folders
file_list = os.listdir(data_folder_path)  # list of files in the directory
# Get export folder path
export_folder_path = filedialog.askdirectory()  # selecting the directory for the folders

# Choose the dimensionality (1D/2D) of the sensor readings
dimensionality = int(input("Please input dimensionality for folder files (1, 2 for 1D/2D):\n"))
if dimensionality in [1, 2]:
    print(f"Dimensionality is {dimensionality}")
else:
    print(f"Dimensionality is incorrect: {dimensionality}")
    raise ValueError

if dimensionality == 1:
    # Choose direction of the sensor readings
    Direction = int(input("Please input acc direction for folder files (0, 1, 2 for x, y, z):\n"))
    if Direction in [0, 1, 2]:
        print(f"Direction is {Direction}")
        #Extract bias and sensitivity from calibration file
        sensitivity = [Calibration.iloc[Direction]["Sensitivity (m/s²/LSB)"]]
        bias = [Calibration.iloc[Direction]["Bias"]]
    else:
        print(f"Direction is incorrect: {Direction}")

elif dimensionality == 2:
        #Extract bias and sensitivity from calibration file
        sensitivity = Calibration.iloc[0:2]["Sensitivity (m/s²/LSB)"].to_numpy()
        bias = Calibration.iloc[0:2]["Bias"].to_numpy()
else:
    print(f"Error in dimensionlity {dimensionality}")
    raise ValueError

for file in file_list:
    file_path = os.path.join(data_folder_path, file)
    print(f"Processing file: {file_path}")

    try:
        # Read file with no header (assuming no column names)
        Data = pd.read_csv(file_path, header=None, dtype=str)

        # Extract first three rows (Description part)
        Description = Data.iloc[:3].fillna('').astype(str)

        # Search for fs and dt in the second row
        fs_match = re.search(r'fs:\s*(\d+)', Description.iloc[1, 0])
        dt_match = re.search(r'dt:\s*(\d+)', Description.iloc[1, 0])

        # Extract numerical values or set None if not found
        fs_value = int(fs_match.group(1)) if fs_match else None
        dt_value = float(dt_match.group(1)) if dt_match else None

        print(f"Extracted fs: {fs_value}, dt: {dt_value}")

        # Remove first three rows and reset index
        Data = Data.iloc[3:].reset_index(drop=True)

        if dimensionality == 1:
            # Ensure valid rows (length 12)
            Data = Data[Data[0].str.len() == 12].astype(str)
        elif dimensionality == 2:   
            # Ensure valid rows (length 24)
            Data = Data[Data[0].str.len() == 24].astype(str)
        else: 
            print(f"Error in data valid rows. the data is {Data}")
            raise ValueError

        # Add extracted fs and dt as new columns

        if dimensionality == 2:
            sensor_column_end = 24
        elif dimensionality == 1:
            sensor_column_end = 12
        
        # Split 12/24-digit numbers into 4/8 columns
        Data = Data[0].apply(lambda x: [x[i:i+3] for i in range(0, len(x), 3)])
        Data = Data.apply(lambda row: row + [np.nan] * (8 - len(row)))
        Data = pd.DataFrame(Data.tolist(), columns=["Sensor 1 - X", "Sensor 2 - X", "Sensor 3 - X", "Sensor 4 - X", "Sensor 1 - Y", "Sensor 2 - Y", "Sensor 3 - Y", "Sensor 4 - Y"])
        for dimension in range(dimensionality):
            sensor_direction = ["X", "Y"]
            for sensor in range(1, 5):
                Data[f"Sensor {sensor} - {sensor_direction[dimension]}"] = (Data[f"Sensor {sensor} - {sensor_direction[dimension]}"].astype(float) - bias[dimension]) * sensitivity[dimension] #Actual convertion using calibAccel

        # Create cumulative time index
        time_index = np.arange(len(Data)) * dt_value / 1_000_000  # Convert μs to seconds
        print(time_index)
        # Assign time index
        Data.insert(0, "Time (us)", time_index)  # Add as a regular column instead of index
        Data.insert(1, "fs", fs_value)
        Data.insert(2, "dt", dt_value)

        print(Data.head())  # Preview data

        # Save new csv with timeseries

        export_file_path = os.path.join(export_folder_path, f"timeseries_{file}")
        Data.to_csv(export_file_path, index=False)
        print(f"Saved to: {export_file_path}")

    except Exception as e:
        print(f"Error processing {file}: {e}")