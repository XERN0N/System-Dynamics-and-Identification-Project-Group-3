import pandas as pd
import numpy as np
import os as os
import re
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import filedialog

tk.Tk().withdraw()

Data_folder_path = filedialog.askdirectory()  # selecting the directory for the folders
File_list = os.listdir(Data_folder_path)  # list of files in the directory

for file in File_list:
    file_path = os.path.join(Data_folder_path, file)
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
        dt_value = int(dt_match.group(1)) if dt_match else None

        print(f"Extracted fs: {fs_value}, dt: {dt_value}")

        # Remove first three rows and reset index
        Data = Data.iloc[3:].reset_index(drop=True)

        # Ensure valid rows (length 12)
        Data = Data[Data[0].str.len() == 12].astype(str)

        # Split 12-digit numbers into 4 columns
        Data = Data[0].apply(lambda x: [x[i:i+3] for i in range(0, 12, 3)])
        Data = pd.DataFrame(Data.tolist(), columns=["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"])

        # Add extracted fs and dt as new columns
        Data["fs"] = fs_value
        Data["dt"] = dt_value

        print(Data.head())  # Preview data

        # Save new file
        new_file_path = os.path.join(Data_folder_path, f"manipulated_{file}")
        Data.to_csv(new_file_path, index=False)
        print(f"Saved to: {new_file_path}")

    except Exception as e:
        print(f"Error processing {file}: {e}")