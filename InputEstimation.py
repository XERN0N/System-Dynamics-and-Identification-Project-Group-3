from SystemModels import *
from SystemSolvers import *
from OriginalModel import generate_original_model
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
from sklearn.linear_model import Lasso

#------------------
import time
#------------------

np.set_printoptions(linewidth=1000, precision=2)

model = generate_original_model()

ouput_DOFs = np.linspace(19, 73, 4, dtype=int)

stacked_output_hammer = pd.read_csv("Sorted timeseries/Forced/1D/Hammer/Calibrated/X/timeseries_gr3_1D_x_resonant2025-03-27-10-56-47.csv.csv")
time_vector_hammer = stacked_output_hammer['Time (s)']
hammer_data = stacked_output_hammer["Hammer force [N]"]
stacked_output_read = pd.read_csv("Sorted timeseries/Forced/1D/Accelerometer/Calibrated/gr3_1D_x_resonant.csv", nrows=2000, skiprows=range(1, 7001))

time_vector = stacked_output_read['Time (us)']
stacked_output = np.empty(len(time_vector)*4)
toeplitz_matrix = model.get_toeplitz_matrix('accelerance', (85,), ouput_DOFs, 1/927, len(stacked_output_read['Time (us)']))

for i, rows in enumerate(stacked_output_read["Time (us)"]):
    stacked_output[i*4:i*4+4] = stacked_output_read.iloc[[i], [3, 4, 5, 6,]].values


LN = stacked_output #assume z0 = 0

left_singular_values, singular_values, right_singular_values = np.linalg.svd(toeplitz_matrix)

from sklearn.linear_model import Lasso


for alpha in [1e-4, 9e-5, 8e-5, 7e-5]:
    model = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=10000, tol=1e-6)
    model.fit(toeplitz_matrix, LN)
    u_hat = model.coef_


    print(u_hat)

    """ # optional clean-up:
    thr = 0.05 * u_hat.max()
    u_hat[u_hat < thr] = 0 """



    """ s = int(len(singular_values) * 0.5)

    projected_values = left_singular_values.T @ stacked_output

    estimated_inputs = (projected_values[:s] / singular_values[:s]) @ right_singular_values[:s, :] """







    """
    reconstructed_input = np.empty(len(time_vector))
    reconstructed_input_estimate = np.empty(len(time_vector))

    inverted_toeplitz = linalg.pinv(toeplitz_matrix)
    reconstructed_input = inverted_toeplitz @ stacked_output

    inverted_toeplitz_estimate = linalg.pinv(toeplitz_matrix, atol=0.60)

    reconstructed_input_estimate = inverted_toeplitz_estimate @ stacked_output
    #condition_number_estimate = np.linalg.cond(inverted_toeplitz_estimate)

    Lasso = Lasso(alpha=0.01, max_iter=1000, positive=True)
    Lasso.fit(toeplitz_matrix, stacked_output)
    U_est = Lasso.coef_ 


    condition_number_estimate = np.linalg.cond(toeplitz_matrix)
    #print(condition_number)
    print(condition_number_estimate)



    left_singular_matrix, singular_values, right_singular_matrix = np.linalg.svd(toeplitz_matrix, full_matrices=False)

    condition_number = singular_values[0] / singular_values[-1]
    print(condition_number)
    """

    
    plt.plot(time_vector, u_hat, label="Estimated inputs")
plt.plot(time_vector, stacked_output_read["Sensor 4 - X"], label="stacked")
#plt.plot(time_vector, reconstructed_input, label="reconstructed")
#plt.plot(time_vector, reconstructed_input_estimate, label="reconstructed_estimate")
#plt.plot(time_vector, U_est, label="U_estimate")
plt.plot(time_vector_hammer, hammer_data, label="Hammer_data")





    #plt.scatter(range(len(singular_values)), singular_values)

plt.yscale('linear')
plt.legend()
plt.show()
