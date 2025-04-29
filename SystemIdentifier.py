import numpy as np
from collections import deque 
from scipy.linalg import svd,logm
import matplotlib.pyplot as plt

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
    damping_ratios = -np.real(eigenvalues[mask]) / eigen_frequencies[mask]
    


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

data = np.loadtxt('timeseries_gr3_FirstSoft.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
data2 = np.loadtxt('timeseries_gr3_SecondSoft.txt', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3))
model_order_selector(1, 20, 30, data, 0.001079)
ef, dr , phi = system_identifier(16, 40, data, 0.001079)
print(ef)
print(dr)
print(phi)
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


   
