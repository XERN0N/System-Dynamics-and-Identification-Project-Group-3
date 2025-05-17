import numpy as np
from collections import deque 
from scipy.linalg import svd,logm
import matplotlib.pyplot as plt
from SystemModels import Beam_Lattice
from scipy.signal import find_peaks
import numpy.typing as npt


np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=10000, suppress=True, precision=3)

def system_identifier(output_data: npt.ArrayLike,
                      order_number: int = 6,
                      number_of_block_rows: int = 24,
                      time_step: float = 1 / 470
                      ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
                      
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


if __name__ == "__main__":
    data = np.loadtxt('timeseries_gr3_FirstSoft.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
    data2 = np.loadtxt('timeseries_gr3_SecondSoft.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
    data3 = np.loadtxt('timeseries_gr3_FirstStiff.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
    data4 = np.loadtxt('timeseries_gr3_SecondStiff.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
#bh = system_identifier(3, 9, A)
#print(min(bh.shape))



#model_order_selector(1, 20, 30, data, 0.001079)
#ef, dr , phi = system_identifier(16, 40, data, 0.001079)
#print(ef)
#print(dr)
#print(phi)

def modalexpander(data, estimated_DOF, data_direction, number_of_elements):
    if number_of_elements % 5 != 0:
        raise ValueError("The number_of_elements must be divisible by 5.")
    system = Beam_Lattice()
    e = number_of_elements//5
    system.add_beam_edge(
        number_of_elements=number_of_elements, 
        E_modulus=2.1*10**11, 
        shear_modulus=7.9*10**10,
        primary_moment_of_area=2.157*10**-8,
        secondary_moment_of_area=1.113*10**-8,
        torsional_constant=3.7*10**-8,
        density=7850, 
        cross_sectional_area=1.737*10**-4, 
        coordinates=[[0, 0, 0], [0, 0, 1.7]],
        edge_polar_rotation=0,
        point_mass=1.31,
        point_mass_moment_of_inertias=(0,0,0),
        point_mass_location='end_vertex'
    )
    system.fix_vertices((0,))
    eigfreq, eigvec, damped = system.get_modal_param()
   
    print(eigvec[:, :15])
    print(eigfreq[:])
    print(eigvec.shape)
    if data_direction == 'x':
        mu1 = [6*e,12*e,18*e,24*e]
        col_pick = [0,2,4]
        #start = 0
        #end = 3
    if data_direction == 'y':
        mu1 = [7*e,13*e,19*e,25*e]
        col_pick = [1,3,5]
    mu2 = mu1[estimated_DOF]
    mu1.remove(mu2)
    
    phi_mu1 = eigvec[mu1].T[col_pick].T
    phi_mu2 = eigvec[mu2].T[col_pick].T
    print(phi_mu1)
    print(phi_mu2)
    measured_data = data[:, [i for i in range(4) if i != estimated_DOF]].T
    least_squares = np.linalg.pinv(phi_mu1) @ measured_data
    
    estimated_data = phi_mu2 @ least_squares

    plt.plot(data[:, estimated_DOF], '--r', label='Measured')
    plt.plot(estimated_data, 'k', label='Estimated', alpha=0.5)
    plt.xlabel("Time step")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.title(f"Acceleration at DOF {mu2}")
    plt.show() 

def modelFRF(number_of_elements,accelerometer_location, axis):
    def modelFRF(number_of_elements, accelerometer_location, axis):
        """
        Calculate and plot the Frequency Response Function (FRF) for a beam lattice system.
        Parameters:
        ----------
        number_of_elements : int
            This is the number of elements in the model.
        accelerometer_location : int
            1 is the topmost accelerometer.
        axis : int
            0 is along the x-axis, it follows as structured in BeamLattice.
        Returns:
        -------
        None
            This function does not return a value. It generates a plot of the FRF magnitude.
        Notes:
        -----
        - The function assumes a beam lattice system with specific material and geometric properties.
        - The FRF is calculated based on modal parameters and system matrices.
        - Peaks in the FRF magnitude are identified and annotated on the plot.
        """

    #The top most accelerometer is 1, the bottom most is 4

    
    f = accelerometer_location*round(0.34*number_of_elements/1.7)
    k = (number_of_elements+1-f)*6+axis
    system = Beam_Lattice()
    
    system.add_beam_edge(
        number_of_elements=number_of_elements, 
        E_modulus=2.1*10**11, 
        shear_modulus=7.9*10**10,
        primary_moment_of_area=2.157*10**-8,
        secondary_moment_of_area=1.113*10**-8,
        torsional_constant=3.7*10**-8,
        density=7850, 
        cross_sectional_area=1.737*10**-4, 
        coordinates=[[0, 0, 0], [0, 0, 1.7]],
        edge_polar_rotation=0,
        point_mass=1.31,
        point_mass_moment_of_inertias=(0,0,0),
        point_mass_location='end_vertex'
    )
    system.fix_vertices((0,))
    eigfreq, eigvec,_ = system.get_modal_param()
    mass_matrix,_,_ = system.get_system_level_matrices()
    freq = np.linspace(0, 300, 1000)  # Frequency range for FRF calculation
    omega = 2 * np.pi * freq  # convert to rad/s
    omega_n = 2 * np.pi * eigfreq  # convert to rad/s
    n_modes = len(omega_n)
    h_kl = np.zeros(len(freq), dtype=complex)

    for j in range(n_modes):
        phi_j = eigvec[:, j].reshape(-1, 1)
        m_j = (phi_j.T @ mass_matrix @ phi_j).item()
        num = phi_j[k, 0] * phi_j[k, 0]
        denom = omega_n[j]**2 - omega**2 + 2j * 0.005 * omega_n[j] * omega
        h_kl += num / (m_j * denom)

    Z_kl = -omega**2 * h_kl
    frf_mag = np.abs(Z_kl)  # Magnitude of the FRF
    peaks, _ = find_peaks(frf_mag, height=0)

    
    plt.figure(figsize=(8, 4))
    plt.plot(freq, frf_mag, color='orange')

    for peak in peaks:
        f_peak = freq[peak]
        m_peak = frf_mag[peak]
        plt.plot(f_peak, m_peak, 'ro', markersize=6)  # red circle
        plt.text(f_peak, m_peak * 1.05, f'{f_peak:.1f} Hz', ha='center', va='bottom', fontsize=9)

    plt.title(f'|Z_{{{k+1}{k+1}}}(ω)| Inertance FRF')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [m/s² per N]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    modalexpander(data2,3,'x',30)
#modelFRF(40,1,1)
#fre1, bla, tis = system_identifier(8, 12, data, 0.001079)
#print(fre1)
#print(bla)
#print(tis)


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




   
