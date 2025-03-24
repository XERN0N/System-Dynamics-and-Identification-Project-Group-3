import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import filedialog


def select_data_folder():
    # Opens a folder selection dialog and returns the selected path.
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
    return folder_path


def load_csv(file_path):
    #Reads the csv files from the folder
    try:
        data = pd.read_csv(file_path, dtype=float, delimiter=",") # Read the CSV file into a DataFrame
        if data.empty or data.shape[1] < 5: # Check if the file has at least 5 columns
            raise ValueError("Insufficient data columns") #Error at too few columns
        return data 
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None


def compute_fft(signal, delta_t):
    #Computes the FFT of the signal and returns the frequencies and magnitudes
    n = len(signal)
    freqs = fftfreq(n, d=delta_t)[:n // 2]
    fft_vals = fft(signal)
    mags = 2.0 / n * np.abs(fft_vals[:n // 2])
    return freqs, mags


def find_peak_frequencies(freqs, mags, num_peaks=4):
    """Finds the top N peak frequencies from the FFT magnitude."""
    peak_indices = np.argpartition(mags, -num_peaks)[-num_peaks:]
    peak_freqs = freqs[peak_indices]
    peak_mags = mags[peak_indices]
    peak_idx = np.argmax(peak_mags)
    return peak_freqs, peak_mags, peak_freqs[peak_idx], peak_mags[peak_idx]


def plot_fft_results(file_name, fft_results, delta_t):
    """Plots the FFT results for each sensor."""
    plt.figure(figsize=(12, 6))
    colors = ['b', 'r', 'g', 'm']

    for i, (freqs, mags, peaks_f, peaks_m) in enumerate(fft_results):
        label = f"Sensor {i}"
        plt.plot(freqs, mags, label=label, color=colors[i])
        plt.plot(peaks_f, peaks_m, 'o', color=colors[i])  # Highlight peaks

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"FFT of the 4 Sensors - {file_name}")
    plt.xlim(0, 1 / (2 * delta_t))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def process_file(file_path):
    """Handles reading, processing, and plotting for one file."""
    print(f"\nProcessing: {file_path}")
    data = load_csv(file_path)
    if data is None:
        return

    time = data.iloc[:, 4].values
    delta_t = np.mean(np.diff(time))

    fft_results = []

    for i in range(4):
        signal = data.iloc[:, i].values
        freqs, mags = compute_fft(signal, delta_t)
        peak_freqs, peak_mags, peak_x, peak_y = find_peak_frequencies(freqs, mags)

        print(f"Sensor {i}: Peak Frequency = {peak_x:.2f} Hz, Peak Amplitude = {peak_y:.3f}")
        fft_results.append((freqs, mags, peak_freqs, peak_mags))

    plot_fft_results(os.path.basename(file_path), fft_results, delta_t)


def main():
    folder_path = select_data_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(".txt")]
    if not file_list:
        print("No CSV files found in the selected directory.")
        return

    for file in file_list:
        file_path = os.path.join(folder_path, file)
        process_file(file_path)


if __name__ == "__main__":
    main()





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
print(f"Error processing {file}: {e}")'''
