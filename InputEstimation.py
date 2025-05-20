from SystemModels import *
from SystemSolvers import *
from OriginalModel import generate_original_model
import matplotlib.pyplot as plt
import pandas as pd

#------------------
from scipy import linalg
from sklearn.linear_model import Lasso
from tqdm import tqdm
import time
from datetime import datetime
import os
#------------------

model = generate_original_model()

ouput_DOFs = np.linspace(19, 73, 4, dtype=int)

hammer_delay = 14150 #compensate for delay between signals
stacked_output_hammer = pd.read_csv(r"Sorted timeseries\Test gruppe 1\Hammer kalibreret\timeseries_hammer_gr1_y_dir_160_4hit.csv.csv", nrows=27000)
time_vector_hammer = stacked_output_hammer['Time (s)'].values
hammer_data = stacked_output_hammer["Hammer force [N]"].values

stacked_output_read = pd.read_csv(r"Sorted timeseries\Test gruppe 1\Accelerometer kalibreret\timeseries_gr1_y_dir_160_4hit.csv", nrows=3200)
time_vector = stacked_output_read['Time (us)'].values

#------------------- WINDOWING----------------

window_length = 3200
window_step = 3200

#alphas = np.logspace(-5, -3, 1000)

sensor_data = stacked_output_read.iloc[:, [3, 4, 5, 6,]].values
total_length = len(sensor_data)
window_start_indices = range(0, len(sensor_data) - window_length +1, window_step)

#Preallocate zeros (has to be zeros to avoid empty values)
input_reconstruction_array = np.zeros(total_length)
lasso_full_estimated_inputs = np.zeros(total_length)
lasso_estimated_inputs_truncated = np.zeros(total_length)
number_of_overlaps = np.zeros(total_length)

#Get toeplitz matrix (time invariant)
toeplitz_matrix = model.get_toeplitz_matrix('accelerance', (85,), ouput_DOFs, 1/927, window_length)
left_singular_values, singular_values, right_singular_values = linalg.svd(toeplitz_matrix, full_matrices=False, overwrite_a=False)

for start in tqdm(window_start_indices, desc="Processing input reconstruction windows"):
    #Get window
    window = sensor_data[start : start + window_length]
    #calculations from lecture 26 eq (9)
    #Window flattened to get 1D Ln vector assuming z0=0
    LN = np.empty(window_length*4)
    LN = window.flatten()

    #amount of singular values to include for regularization
    s = 3150 #from L-curve plot
    #s = int(len(singular_values) * 0.9) #relative version
    
    #Vectorized version of mu_i.T * LN 
    projected_values = left_singular_values.T @ LN
    #divide by sigma_i and multiply by v_i
    estimated_inputs = (projected_values[:s] / singular_values[:s]) @ right_singular_values[:s, :]
    toeplitz_matrix_svd_approx = (left_singular_values[:, :s] @ np.diag(singular_values[:s]) @ right_singular_values[:s, :])

    #collect windowed values and insert into final vector
    input_reconstruction_array[start : start + window_length] += estimated_inputs[:window_length]
    number_of_overlaps[start : start + window_length] += 1
    
    #Lasso regularization on toeplitz
    start_time = time.time()
    lasso_full = Lasso(1e-5, positive=True, tol=1e-2, max_iter=100000, precompute=True, warm_start=False)
    lasso_full.fit(toeplitz_matrix, LN)
    lasso_full_coeffs = lasso_full.coef_
    lasso_full_estimated_inputs[start : start + window_length] += lasso_full_coeffs[:window_length]
    end_time = time.time()
    print("\n", end_time-start_time, lasso_full.n_iter_, lasso_full.tol, lasso_full.alpha)
    print(np.linalg.cond(lasso_full_coeffs))

    #Lasso regularization on truncated toeplitz
    start_time = time.time()
    lasso_truncated = Lasso(2e-5, positive=True, tol=1e-2, max_iter=100000, precompute=True, warm_start=False)
    lasso_truncated.fit(toeplitz_matrix_svd_approx, LN)
    lasso_truncated_coeffs = lasso_truncated.coef_
    lasso_estimated_inputs_truncated[start : start + window_length] += lasso_truncated_coeffs[:window_length]
    end_time = time.time()
    print(end_time-start_time, lasso_truncated.n_iter_, lasso_truncated.tol, lasso_truncated.alpha)
    print(np.linalg.cond(lasso_full_coeffs))


#correct if no overlaps and take average
number_of_overlaps[number_of_overlaps == 0] = 1
input_reconstruction_svd_truncated = input_reconstruction_array / number_of_overlaps
input_reconstruction_lasso_full = lasso_full_estimated_inputs / number_of_overlaps
input_reconstruction_truncated = lasso_estimated_inputs_truncated / number_of_overlaps

#------------------PLOTS----------------------------
#Time
timestamp = datetime.now().strftime("%H%M%S")
# Create directory if it doesn't exist
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)
dpi = 800
figsize = (18, 12)

plt.figure(figsize=figsize)
plt.plot(time_vector_hammer[:-hammer_delay], hammer_data[hammer_delay:], label="Hammer data", color='black', alpha=0.4)
plt.plot(time_vector, stacked_output_read["Sensor 4 - X"], label="Sensor values for top sensor", alpha=0.3, color='red')
plt.plot(time_vector, input_reconstruction_svd_truncated, label="Input reconstruction truncated", alpha=0.4, color='cyan')
plt.plot(time_vector, input_reconstruction_lasso_full, label="Input reconstruction lasso full", color='blue')
plt.plot(time_vector, input_reconstruction_truncated, label="Input reconstruction lasso truncated", color='red', linestyle='--')

plt.yscale('linear')
plt.ylim([-50, 230])
plt.xlim([0, 3.5])
plt.legend()
plt.savefig(os.path.join(output_folder, f"Reconstruction_vs_Hammer_{timestamp}.png"), dpi=dpi)
plt.show()

# --- L-curve loglog plot for singular values ---
plt.figure(figsize=figsize)
plt.loglog(singular_values)
plt.title("L-curve (Singular Values, log-log scale)")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(output_folder, f"Lcurve_singular_values_{timestamp}.png"), dpi=dpi)
plt.show()