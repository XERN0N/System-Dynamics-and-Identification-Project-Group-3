import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Literal
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.integrate import cumulative_trapezoid
from scipy import signal as sig
import tkinter as tk
from tkinter import filedialog


def select_data_folder():
    # Opens a folder selection dialog and returns the selected path.
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
    
    if not folder_path:
        raise ValueError("No folder selected. Exiting.")
        return
    return folder_path

def load_csv(file_path):
    #Reads the csv files from the folder
    try:
        data = pd.read_csv(file_path, dtype=float, delimiter=",") # Read the CSV file into a DataFrame
        if data.empty or data.shape[1] < 8: # Check if the) file has at least 5 columns
            raise ValueError("Insufficient data columns") #Error at too few columns
        return data 
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None
    
def get_filter(lower_cutoff_frequency, higher_cutoff_frequency = None, filter_order = 2, filter_type = Literal["highpass", "bandpass"], sampling_frequency = 427):
    nyquist = sampling_frequency * 0.5
    normalized_lower = lower_cutoff_frequency / nyquist
    if higher_cutoff_frequency is not None:
        normalized_higher = higher_cutoff_frequency / nyquist
    if filter_type == "highpass":
        custom_filter = sig.butter(filter_order, normalized_lower, 'highpass', output='sos')
    elif filter_type == "bandpass":
        custom_filter = sig.butter(filter_order, np.array([normalized_lower, normalized_higher]), 'bandpass', output='sos')
    else:
        print(f"Wrong inputs: {lower_cutoff_frequency}, {higher_cutoff_frequency}, {filter_order}, {filter_type}")
    return custom_filter

def compute_fft(signal, delta_t, hann = False):
    #Computes the FFT of the signal and returns the frequencies and magnitudes
    signal_length = len(signal)
    if hann == True:
        hann_window = np.hanning(signal_length)
        signal *= hann_window
        norm_factor = np.sum(hann_window)
    else:
        norm_factor = signal_length
    freqs = rfftfreq(signal_length, delta_t)
    mags = 2.0 / norm_factor * rfft(signal)

    return freqs, mags

def Signal_integration(Desired_signal, Provided_signal, delta_t):
    #Define the desired signal and integrate accordingly
    if Desired_signal == "Velocity":
        Integrated_signal = cumulative_trapezoid(Provided_signal, dx=delta_t, initial=0) #integrating once
    elif Desired_signal == "Displacement":
        Integrated_signal = cumulative_trapezoid(Provided_signal, dx=delta_t, initial=0)  #integrating twice
    else:
        raise ValueError(f"Invalid signal type: {Desired_signal}")
    
    return Integrated_signal


def plot_fft_results(file_name, fft_bin, fft_results, peak_freqs, peak_mags, delta_t):
    #Plots the FFT results for each sensor.
    plt.figure(figsize=(12, 6))
    colors = ['b', 'r', 'g', 'm']
    #For-loop to take each sensor's FFT results and plot them
    for i in range(fft_results.shape[1]):
        plt.plot(fft_bin[:, i], fft_results[:, i], label=f"Sensor {i+1}", color=colors[i % len(colors)])
        plt.plot(peak_freqs[i], peak_mags[i], 'o', color=colors[i % len(colors)])


    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"FFT of the 4 Sensors - {file_name}")
    plt.xlim(0, 1 / (2 * delta_t))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_peak_frequencies(freqs, mags, peak_width = None, min_amplitude = None):
    """Finds the top N peak frequencies from the FFT magnitude."""
    if peak_width is not None:
        freq_resolution = freqs[1] - freqs[0]
        peak_width = int(peak_width / freq_resolution)

    if min_amplitude is not None:
        peaks, _ = sig.find_peaks(mags, distance=peak_width, height=min_amplitude)
    else:
        peaks, _ = sig.find_peaks(mags, distance=peak_width)

    peak_freqs = freqs[peaks]
    peak_mags = mags[peaks]

    return peak_freqs, peak_mags

def plot_signal(file_name, Acceleration_signal, Velocity_signal, Displacement_signal, time):
    
    signal_data = [Acceleration_signal, Velocity_signal, Displacement_signal]
    signal_labels = ['Acceleration', 'Velocity', 'Displacement']
    y_labels = ['Acceleration (m/s²)', 'Velocity (m/s)', 'Displacement (m)']
    
    cmap =  cm.get_cmap('tab10', 8)
    colors = [cmap(i) for i in range(8)]
    
    #colors = ['b', 'r', 'g', 'm', 'b', 'r', 'g', 'm']
    print(signal_data[1][:5, :])
    fig, axs = plt.subplots(3, 2, figsize=(16, 10), sharex=True)

    fig.suptitle(f"Sensor signals over time from file: {file_name}")    
    for i, signal_label in enumerate(signal_labels):
        axs[i, 0].set_title(f"{signal_labels[i]} - Sensor - X-direction")
        axs[i, 1].set_title(f"{signal_labels[i]} - Sensor - Y-direction")

        for sensor in range(4):
            axs[i, 0].plot(time, signal_data[i][:, sensor], label=f'Sensor {sensor+1}', color = colors[sensor])
        for sensor in range(4, 8):
            axs[i, 1].plot(time, signal_data[i][:, sensor], label=f'Sensor {sensor+4}', color = colors[sensor])
        axs[i, 0].set_ylabel(y_labels[i])
        axs[i, 0].grid(True)
        axs[i, 0].legend()
        axs[i, 1].set_ylabel(y_labels[i])
        axs[i, 1].grid(True)
        axs[i, 1].legend()

    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    #plt.savefig(file_name)
    

def process_file(file_path, sos_filter = None, hann = None, peak_width = None, min_amplitude = None):
    """Handles reading, processing, and plotting for one file."""
    print(f"\nProcessing: {file_path}")
    data = load_csv(file_path)
    if data is None:
        return

    signal_length = len(data)
    last_sensor_direction = 11

    #initialize numpy arrays for processed values
    Acceleration_signal = np.empty((signal_length, last_sensor_direction-3))
    Displacement_signal = np.empty((signal_length, last_sensor_direction-3))
    Velocity_signal = np.empty((signal_length, last_sensor_direction-3))
    n_fft = signal_length//2 + 1
    fft_results = np.empty((n_fft, last_sensor_direction-3))
    fft_bins = np.empty((n_fft, last_sensor_direction-3))
    peak_freqs = []
    peak_mags = []

    time = data.iloc[:, 0].values
    delta_t = data.iloc[0, 2]/1000000
    signal = data.iloc[:, 3:last_sensor_direction]

    for i, col_name in enumerate(signal.columns):
        sig_data = signal[col_name].values
        sig_data -= np.mean(sig_data)
        if sos_filter is not None:
            sig_data = sig.sosfilt(sos_filter, sig_data)
        fft_bins[:, i], fft_results[:, i] = compute_fft(sig_data, delta_t, hann=hann)

        peaks_f, peaks_m = find_peak_frequencies(fft_bins[:, i], fft_results[:, i], peak_width=peak_width, min_amplitude=min_amplitude)
        peak_freqs.append(peaks_f)
        peak_mags.append(peaks_m)
        
        #Calulate the integrated signals displacement and velocities
        Acceleration_signal[:, i] = sig_data
        Velocity_signal[:, i] = Signal_integration("Velocity", sig_data, delta_t)
        Velocity_signal[:, i] = Velocity_signal[:, i] - np.mean(Velocity_signal[:, i])
        Displacement_signal[:, i] = Signal_integration("Displacement", Velocity_signal[:, i], delta_t)
        Displacement_signal[:, i] = Displacement_signal[:, i] - np.mean(Displacement_signal[:, i])

    plot_signal(os.path.basename(file_path), Acceleration_signal, Velocity_signal, Displacement_signal, time=time)
    plot_fft_results(os.path.basename(file_path), fft_bins, fft_results, peak_freqs, peak_mags, delta_t)

def main():

    folder_path = select_data_folder()

    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(".txt") or f.lower().endswith(".csv")]
    if not file_list:
        print("No CSV files found in the selected directory.")
        return

    for file in file_list:
        file_path = os.path.join(folder_path, file)
        #test_filter = sig.butter(2, 0.01, 'highpass', output='sos')
        test_filter = get_filter(0.5, 220, 2, "bandpass", 470)
        process_file(file_path, sos_filter=test_filter, hann=False, peak_width=5, min_amplitude=0.05)




if __name__ == "__main__":
    main()

