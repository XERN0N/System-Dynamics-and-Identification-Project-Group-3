from SystemModels import *
from SystemSolvers import *

system = Beam_Lattice()

primary_moment_of_area = 1.94e-8
secondary_moment_of_area = 1.02e-8
torsional_constant = 2.29e-8
cross_sectional_area = 1.74e-4

system.add_beam_edge(
    number_of_elements=5,
    E_modulus=2.1e11,
    shear_modulus=7.9e10,
    primary_moment_of_area=primary_moment_of_area,
    secondary_moment_of_area=secondary_moment_of_area,
    torsional_constant=torsional_constant,
    density=7850,
    cross_sectional_area=cross_sectional_area,
    coordinates=((0, 0, 0), (0, 0, 1.7)),
    edge_polar_rotation=0
)

system.fix_vertices((0,))

toeplitz_matrix = system.get_toeplitz_matrix('accelerance', (18,), (6, 12, 18, 24), 1/427, 10_000)

""" 
start_time = time() 
np.linalg.pinv(toplitz_matrix)
end_time = time()

print(end_time - start_time)
 """
print(toeplitz_matrix.shape)