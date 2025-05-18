import numpy as np
from collections import deque 
from scipy.linalg import svd,logm
import matplotlib.pyplot as plt
from SystemModels import Beam_Lattice
from scipy.signal import find_peaks
from OriginalModel import *
from typing import Literal

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=10000, suppress=True, precision=3)

def system_identifier(order_number,number_of_block_rows,output_data,time_step):
    """
    This function calculates the modal parameters based on measured output.
    
    Inputs:
    order_number (int): The order number of the system (ns).
    number_of_block_rows (int): The number of block rows in the system (a).
    output_data (array): The measured output data (displacement, velocity or acceleration).
    time_step (float): The time step of the measured data in seconds.
    
    Returns:
    tuple: A tuple containing the modal parameters (eigen frequency, damping ratio, and mode shape).
    """
    if number_of_block_rows < order_number:
        raise ValueError(f"Number of block rows (a={number_of_block_rows}) must be greater than (ns={order_number}).")
    
    #Datapoints 
    y = np.array(output_data)
    a = number_of_block_rows
    ns = order_number
    dt = time_step
    #Checks the number of measurements
    number_of_datapoints = max(y.shape[0],y.shape[1])

    #Number of outputs
    m = min(y.shape[0],y.shape[1])
    
    b = number_of_datapoints - 2*number_of_block_rows + 1

    #Initializes the deques
    Yp = deque(maxlen=b)
    Yf = deque(maxlen=b)
    #Builds the columns
    for i in range(b-1):
        Yp.append(y[i, :].reshape(-1, 1))
        Yf.append(y[a+i, :].reshape(-1, 1))

    #Initializes the stacked matrices
    stacked_Yp = np.hstack(Yp)
    stacked_Yf = np.hstack(Yf)
    
    #Builds the block Hankel matrix
    for i in range(a-1):
        Yp.append(y[b+i, :].reshape(-1, 1))
        Yp.popleft()
        Yf.append(y[b+a+i, :].reshape(-1, 1))
        Yf.popleft()
        stacked_Yp = np.vstack((stacked_Yp,np.hstack(Yp)))
        stacked_Yf = np.vstack((stacked_Yf,np.hstack(Yf)))          
    block_hankel_matrix = np.vstack((stacked_Yp, stacked_Yf))
    
    #Defining Ka matrix
    K_a = stacked_Yf @ stacked_Yp.T @ np.linalg.pinv(stacked_Yp @ stacked_Yp.T) @ stacked_Yp

    #Calculating the thin SVD of Ka matrix
    U, SS, _ = np.linalg.svd(K_a, full_matrices=False)
    #r1 = min(SS.shape)
    S = np.diag(SS)
    #sig = np.diag(S)

    #Extended observability matrix 
    observability_matrix = U[:,:ns] @ S[:ns,:ns]**0.5

    #Calculating output matrix
    output_matrix_discrete = observability_matrix[:m,:]
    #Observability matrix without last row
    observability_matrix_under = observability_matrix[:-m, :]

    #Observability matrix without first row
    observability_matrix_over = observability_matrix[m:, :]
    #Calculating state matrix
    state_matrix_discrete = np.linalg.pinv(observability_matrix_under) @ observability_matrix_over

    #Transforming into continuous time
    state_matrix_continuos = logm(state_matrix_discrete) / dt
    eigenvalues, eigenvectors = np.linalg.eig(state_matrix_continuos)
    eigen_frequencies = np.abs(np.imag(eigenvalues))
    pos_imag = np.imag(eigenvalues) > 0
    mask = (eigen_frequencies > 1e-8) & pos_imag
    f_Hz = eigen_frequencies[mask] / (2 * np.pi)
    sorted_indices = np.argsort(f_Hz)
    f_Hz = f_Hz[sorted_indices]
    damping_ratios = -np.real(eigenvalues[mask]) / eigen_frequencies[mask]
    damping_ratios = damping_ratios[sorted_indices]
    


    return f_Hz, damping_ratios, eigenvectors



def model_order_selector(n_min,n_max,number_of_block_rows,output_data,time_step):
    orders, freqs = [], []

    for i in range(n_min,n_max+1):
        omega , _ , _ = system_identifier(i, number_of_block_rows, output_data, time_step)
    
        orders.extend([i] * len(omega))
        freqs.extend(omega)

    orders = np.asarray(orders)
    freqs  = np.asarray(freqs)
    fig, ax = plt.subplots()
    ax.scatter(freqs, orders, marker='o')
    ax.set_xlabel("Eigen-frequency  [Hz]")
    ax.set_ylabel("Model order  $n_s$")
    ax.set_title("Stability diagram")
    ax.grid(True)
    plt.show()



A = np.array((np.linspace(1,50,50),np.linspace(1,50,50),np.linspace(1,50,50)))
A = np.transpose(A)



#bh = system_identifier(3, 9, A)
#print(min(bh.shape))

data1 = np.loadtxt('Sorted timeseries/Free/1D/Calibrated/weak_axis_data/timeseries_gr3_FirstSoft.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
data2 = np.loadtxt('Sorted timeseries/Free/1D/Calibrated/weak_axis_data/timeseries_gr3_SecondSoft_2nd.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
data3 = np.loadtxt('Sorted timeseries/Free/1D/Calibrated/strong_axis_data/timeseries_gr3_FirstStiff.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
data4 = np.loadtxt('Sorted timeseries/Free/1D/Calibrated/strong_axis_data/timeseries_gr3_SecondStiff_2nd.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
data5 = np.loadtxt('Sorted timeseries/Free/1D/Calibrated/Torsional_axis_data/X/timeseries_gr3_twisting_2nd.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
#data5 = np.loadtxt('timeseries_serial_output_gr3_SecondSoft_2d.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
#data6 = np.loadtxt('timeseries_serial_output_gr3_twisting_2d.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
#data7 = np.loadtxt('timeseries_serial_output_gr3_SecondStiff_2d.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))

#model_order_selector(1, 20, 30, data, 0.001079)
#ef, dr , phi = system_identifier(16, 40, data, 0.001079)
#print(ef)
#print(dr)
#print(phi)

def modalexpander(acceleration_data_path: str, estimated_accelerometer_number: int, axis: Literal['primary', 'secondary']) -> None:
    """
    Estimates the acceleration at a specified accelerometer location and plots the estimation vs actual acceleration over time.

    Parameters
    ----------
    acceleration_data_path : str
        The relative path to the acceleration data from this git repo.
    estimated_accelerometer_number : int
        The accelerameter number to estimate (from 0 to 3) where 0 is the bottom-most.
    axis : str
        The axis to to modal expansion on. Can be either 'primary' or 'secondary'.
    """
    acceleration_data = np.loadtxt(acceleration_data_path, delimiter=',', skiprows=1, usecols=(3, 4, 5, 6))

    system = generate_original_model()
    eigfreq, eigvec, damped = system.get_modal_param()
   
    print(eigvec[:, :15])
    print(eigfreq[:15])
    print(eigvec.shape)
    
    match axis:
        case 'primary':
            mu1 = list(Vertex_DOFs.output_primary_DOFs.value)
            col_pick = [1,3,5]
        case 'secondary':
            mu1 = list(Vertex_DOFs.output_secondary_DOFs.value)
            col_pick = [2,4,6]
        case _:
            raise ValueError(f"'axis' expected either 'primary' or 'secondary' but recieved '{axis}'.")
    
    mu2 = mu1[estimated_accelerometer_number]
    mu1.remove(mu2)
    print(mu1)
    
    phi_mu1 = eigvec[mu1].T[col_pick].T
    phi_mu2 = eigvec[mu2].T[col_pick].T
    print(phi_mu1)
    print(phi_mu2)
    measured_data = acceleration_data[:, [i for i in range(4) if i != estimated_accelerometer_number]].T
    least_squares = np.linalg.pinv(phi_mu1) @ measured_data
    
    estimated_data = phi_mu2 @ least_squares
    
    scaling_factor = np.linalg.lstsq(estimated_data.reshape(-1, 1), acceleration_data[:, estimated_accelerometer_number].reshape(-1, 1), rcond=None)[0]
    estimated_data_scaled = estimated_data * scaling_factor[0]  # scaling_factor is a 1-element array

    plt.plot(acceleration_data[:, estimated_accelerometer_number], '--r', label='Measured')
    plt.plot(estimated_data_scaled, 'k', label='Estimated', alpha=0.5)
    plt.xlabel("Time step")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.title(f"Acceleration at DOF {mu2}")
    plt.show() 

def modelFRF(accelerometer_location: int, hammer_location: int, axis: Literal['primary', 'secondary'], FRF_kinematic: Literal['receptence', 'mobility', 'accelerance']) -> None:
    """
    Plots the model FRF.

    Parameters
    ----------
    accelerometer_location : int
        The accelorameter number from 0 to 3 where 0 is the buttom-most accelerometer.
    hammer_location : int
        The hammer location number from 0 to 2 where 0 is the buttom-most hammer location.
    axis : str
        Which axis to get the FRF for. Can be either 'primary' or 'secondary'.
    """
    match axis:
        case 'primary':
            output_DOF = Vertex_DOFs.output_primary_DOFs.value[accelerometer_location]
            input_DOF = Vertex_DOFs.input_primary_DOFs.value[hammer_location]
        case 'secondary':
            output_DOF = Vertex_DOFs.output_secondary_DOFs.value[accelerometer_location]
            input_DOF = Vertex_DOFs.input_secondary_DOFs.value[hammer_location]
        case _:
            raise ValueError(f"'axis' expected either 'primary' or 'secondary' but recieved '{axis}'.")

    model = generate_original_model()
    eigfreq, eigvec,_ = model.get_modal_param()
    mass_matrix,_,_ = model.get_system_level_matrices()
    freq = np.linspace(0, 300, 1000)  # Frequency range for FRF calculation
    omega = 2 * np.pi * freq  # convert to rad/s
    omega_n = 2 * np.pi * eigfreq  # convert to rad/s
    n_modes = len(omega_n)
    h_kl = np.zeros(len(freq), dtype=complex)

    for j in range(n_modes):
        phi_j = eigvec[:, j].reshape(-1, 1)
        m_j = (phi_j.T @ mass_matrix @ phi_j).item()
        num = phi_j[output_DOF, 0] * phi_j[input_DOF, 0]
        denom = omega_n[j]**2 - omega**2 + 2j * 0.005 * omega_n[j] * omega
        h_kl += num / (m_j * denom)

    match FRF_kinematic:
        case 'receptence':
            Z_kl = h_kl
        case 'mobility':
            Z_kl = omega * h_kl
        case 'accelerance':
            Z_kl = -omega**2 * h_kl
        case _:
            raise ValueError(f"'FRF_kinematic' recieved an unexpected input. Expected either 'receptence', 'mobility', 'accelerance' but recieved '{FRF_kinematic}'.")

    frf_mag = np.abs(Z_kl)  # Magnitude of the FRF
    peaks, _ = find_peaks(frf_mag, height=0)
    
    plt.figure(figsize=(8, 4))
    plt.plot(freq, frf_mag, color='orange')

    for peak in peaks:
        f_peak = freq[peak]
        m_peak = frf_mag[peak]
        plt.plot(f_peak, m_peak, 'ro', markersize=6)  # red circle
        plt.text(f_peak, m_peak * 1.05, f'{f_peak:.1f} Hz', ha='center', va='bottom', fontsize=9)

    plt.title(f'{FRF_kinematic} FRF with input DOF = {input_DOF} and output DOF = {output_DOF}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('$|h|$')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #fre1, bla, tis = system_identifier(20,50,data4,0.001079)
    #[    4.218     5.817    36.564    50.426   112.377   154.981   230.919   318.463   392.557   410.525   503.7     541.382   597.501 ]
    model_f = [4.217,     5.816,    36.424,    50.279,   111.192,   153.74,    216.241,   225.925,   313.277,   377.456,   503.7,     525.906,   559.812]

    #modalexpander(data2,3,'x',15)
    #modelFRF(40,1,1)
    #fre1, _, _ = system_identifier(30, 50, data, 0.001079)

    #print(fre1)
    #print(bla)
    #print(tis)
    #Kig i lecture 10, for at lave FRF baseret på måle data i forced vibration
    #model_order_selector(10,30,50,data6,0.001079)


    datasets = [data1, data2, data3, data4, data5]
    dataset_labels = [f"data{i+1}" for i in range(len(datasets))]

    # Parameters for system_identifier

    dt = 0.001079

    # Storage for eigenfrequencies
    all_frequencies = []

    # Loop through datasets
    for i, data in enumerate(datasets):
        freqs, _, _ = system_identifier(30, 50, data, dt)
        all_frequencies.append(freqs)

    # Plotting
    plt.figure(figsize=(10, 6))
    for idx, freqs in enumerate(all_frequencies):
        y = np.full_like(freqs, idx + 1)  # assign y-position based on dataset number
        plt.plot(freqs, y, 'o', label=dataset_labels[idx])
    for f in model_f:
        plt.axvline(f, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.yticks(ticks=range(1, 6), labels=['weak-axis','weak-axis-hammer','strong-axis','strong-axis-hammer','twisting'])
    plt.xlabel("Eigen-frequency [Hz]")
    plt.ylabel("Dataset")
    plt.title("Eigenfrequencies per dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    '''
    a = 9
    N = A.shape[0]
    b = N-2*a+1
    m = A.shape[1]
    print(b)

    print(2*a)

    result = np.hstack([A[i, :].reshape(-1, 1) for i in range(b-1)])
    result2 = np.vstack((result, result))
    Yp = deque(maxlen=b)
    Yf = deque(maxlen=b)
    for i in range(b-1):
        Yp.append(A[i, :].reshape(-1, 1))
        Yf.append(A[a+i, :].reshape(-1, 1))

    stacked_Yp = np.hstack(Yp)
    stacked_Yf = np.hstack(Yf)

    print(stacked_Yp)
    print(stacked_Yf)    
    for i in range(a-1):
        Yp.append(A[b+i, :].reshape(-1, 1))
        Yp.popleft()
        Yf.append(A[b+a+i, :].reshape(-1, 1))
        Yf.popleft()
        stacked_Yp = np.vstack((stacked_Yp,np.hstack(Yp)))
        stacked_Yf = np.vstack((stacked_Yf,np.hstack(Yf)))          

    result3 = np.vstack((stacked_Yp, stacked_Yf))

    print(result3)
    '''




    
