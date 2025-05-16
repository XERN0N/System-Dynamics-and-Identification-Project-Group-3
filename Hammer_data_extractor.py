import pandas as pd
import numpy as np
import os as os
import re
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

# Get hammer data folder path
hammer_data_folder_path = filedialog.askdirectory(title="Select hammer data folder")  # selecting the directory for the folders
hammer_file_list = os.listdir(hammer_data_folder_path)  # list of files in the directory
export_folder_path = filedialog.askdirectory(title="Select output folder for data export")  # selecting the directory for the folders

#Deprecated
#bias = 2.5982e-4
#sensitivity = 0.020472
dt_value = 6.0679e-4

for file in hammer_file_list:
    hammer_file_path = os.path.join(hammer_data_folder_path, file)
    print(f"Processing file: {hammer_file_path}")

    try:
        hammer_data = pd.read_csv(hammer_file_path, header=0, delimiter=";", decimal=",", engine="python", names=["Time [s]", "Hammer force [N]"])
        hammer_data["Hammer force [N]"] = pd.to_numeric(hammer_data["Hammer force [N]"], errors='coerce') #Converting to float
        hammer_data.drop(columns=["Time [s]"], inplace=True) #Removing the original date/time string
        time_index = np.arange(len(hammer_data)) * dt_value #Create time index from dt and length of file
        hammer_data.insert(0, "Time (s)", time_index)  # Add corrected time
        hammer_data = hammer_data.dropna(axis=0)

        #Deprecated ------ modify values with calibration data
        #hammer_data["Hammer force [N]"] = (hammer_data["Hammer force [N]"] - bias) / sensitivity
       

        export_file_path = os.path.join(export_folder_path, f"timeseries_{file}")
        hammer_data.to_csv(f"{export_file_path}.csv", index=False)
        print(f"Saved to: {export_file_path}.csv")

    except Exception as e:
        print(f"Error processing {file}: {e}")