from SystemModels import *
from SystemSolvers import *
from OriginalModel import generate_original_model
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(linewidth=1000, precision=2)

model = generate_original_model()

ouput_DOFs = np.linspace(18, 72, 4, dtype=int)


stacked_output_read = pd.read_csv(r"Sorted timeseries\Forced\1D\Accelerometer\Calibrated\gr3_1D_x_resonant.csv", nrows=4000)

stacked_output = np.empty(len(stacked_output_read['Time (us)'])*4)
toeplitz_matrix = model.get_toeplitz_matrix('accelerance', (72,), ouput_DOFs, 1/470, len(stacked_output_read['Time (us)']))

for i, rows in enumerate(stacked_output_read["Time (us)"]):
    stacked_output[i*4:i*4+4] = stacked_output_read.iloc[[i], [3, 4, 5, 6,]].values

print(stacked_output.shape)
print(toeplitz_matrix.shape)

reconstructed_input = np.linalg.pinv(toeplitz_matrix) @ stacked_output

print(reconstructed_input)

left_singular_matrix, singular_values, right_singular_matrix = np.linalg.svd(toeplitz_matrix, full_matrices=False)

plt.scatter(range(len(singular_values)), singular_values)

plt.yscale('log')

plt.show()