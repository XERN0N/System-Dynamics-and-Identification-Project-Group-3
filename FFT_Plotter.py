import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os as os
import tkinter as tk
from tkinter import filedialog

tk.Tk().withdraw()
Data_folder_path = filedialog.askdirectory()  # selecting the directory for the folders
File_list = os.listdir(Data_folder_path)  # list of files in the directory

for file in File_list:
    file_path = os.path.join(Data_folder_path, file)
    print(f"Processing file: {file_path}")

    try:
        # Read CSV without a header (assumes no column names)
        Data = pd.read_csv(file_path, dtype=float, delimiter=",")

        if Data.empty or Data.shape[1] < 5:
            print(f"Skipping {file}: Insufficient data columns")
            continue

        print(Data.head())  # Preview data

        # Extract time (column 4)
        Time = Data.iloc[:, 4].values  # Convert to NumPy array
        Signal_length = len(Time)

        # Compute Delta_T (mean time step in seconds)
        Delta_T = np.mean(np.diff(Time))  # Convert microseconds to seconds

        # Extract sensor signals (columns 0-3)
        Sensors = [Data.iloc[:, i].values for i in range(4)]  # List of 4 sensor arrays

        # Compute FFT frequency bins in **Hz** (not rad/s)
        fft_freq_signal = fftfreq(Signal_length, d=Delta_T)[:Signal_length // 2]


        # Prepare the figure
        plt.figure(figsize=(12, 6))
        colors = ['b', 'r', 'g', 'm']  # Assign colors for 4 sensors

        for i, signal in enumerate(Sensors):
            # Compute FFT for each sensor
            fft_signal = fft(signal)
            fft_signal_mag = np.abs(fft_signal)

            # Extract the positive half
            fft_signal_mag = 2.0 / Signal_length * fft_signal_mag[:Signal_length // 2]

            # Find the top 4 peak frequencies
            num_max_values = 4
            largest_indices = np.argpartition(fft_signal_mag, -num_max_values)[-num_max_values:]
            largest_values = fft_signal_mag[largest_indices]
            max_values = largest_values
            max_loc = fft_freq_signal[largest_indices]  # Now in Hz

            # Identify peak frequency
            peakIdx = np.argmax(max_values)
            peakX = max_loc[peakIdx]  # Already in Hz
            peakY = max_values[peakIdx]

            print(f"Sensor {i}: Peak Frequency = {peakX:.2f} Hz, Peak Amplitude = {peakY:.3f}")

            # Plot FFT for this sensor
            plt.plot(fft_freq_signal, fft_signal_mag, label=f'Sensor {i}', color=colors[i])
            plt.plot(max_loc, max_values, 'o', color=colors[i])  # Highlight peaks

        # Plot formatting
        plt.xlabel('Frequency (Hz)')  # Now correctly in Hz
        plt.ylabel('Amplitude')
        plt.xlim(0,(1/Delta_T)/2)
        plt.title(f'FFT of the 4 Sensors - {file}')
        plt.legend()
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"Error processing {file}: {e}")


'''
for file in File_list:
    file_path = os.path.join(Data_folder_path, file)
    print(f"Processing file: {file_path}")

    try:
        # Read file with no header (assuming no column names)
        Data = pd.read_csv(file_path, dtype=float, delimiter=",")
        print(Data.head()) #Preview data

        # Extract first three rows (Description part)
        Description = Data.iloc[:3].fillna('').astype(str)

    except Exception as e:
        print(f"Error processing {file}: {e}")

Time = Data[:, 0]
Signal_length = Time.size
Delta_T = Time[-1]/Signal_length                                            #taking avg delta T and rounds
Signal = Data[:, 1]                                                                  #extract signaly
fft_signal = fft(Signal)                                                           #Fourier coefficients
fft_signal_mag = np.abs(fft_signal)
fft_freq_signal = fftfreq(Signal_length, Delta_T)[:Signal_length//2]
num_max_values = 4
largest_indices = np.argpartition(fft_signal_mag[:Signal_length//2], -num_max_values)[-num_max_values:]
largest_values = fft_signal_mag[largest_indices]
max = 2.0/Signal_length * largest_values
max_loc = fft_freq_signal[largest_indices] * 2 * np.pi

peakIdx = np.argmax(max)
peakX = max_loc[peakIdx]
peakY = max[peakIdx]
print(peakX)
print(peakY)

plt.plot(2*np.pi*fft_freq_signal, 2.0/Signal_length * fft_signal_mag[:Signal_length//2]) #Single-sided fourier
plt.plot(max_loc, max,'o')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
#plt.xlim(0, 1/Delta_T*1/2)
plt.xlim(0, 350)
plt.title('FFT of the Signal')
plt.show()'
'''