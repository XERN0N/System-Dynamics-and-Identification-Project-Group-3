from SystemModels import *
from SystemSolvers import *
from OriginalModel import generate_original_model
import matplotlib.pyplot as plt

model = generate_original_model()

ouput_DOFs = np.linspace(18, 72, 4, dtype=int)

toeplitz_matrix = model.get_toeplitz_matrix('accelerance', (72,), ouput_DOFs, 1/427, 1000)

left_singular_matrix, singular_values, right_singular_matrix = np.linalg.svd(toeplitz_matrix, full_matrices=False)

plt.scatter(range(len(singular_values)), singular_values)

plt.yscale('log')

plt.show()