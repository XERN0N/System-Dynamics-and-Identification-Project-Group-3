from SystemModels import Beam_Lattice

def generate_original_model(number_of_elements: int = 15) -> Beam_Lattice:
    """
    Generates the original model "before" running the SSI.
    Sources:
        [1] Sigurd Mousten Jager Nielsens filer - System Dynamics and Identification Project/Beam properties/RHS_30x20x2.pdf
    
    Parameters
    ----------
    number_of_elements : int, optional
        The number of elements in the vertical beam.

    Returns
    -------
    Beam_Lattice
        The model.
    """
    model = Beam_Lattice()

    model.add_beam_edge(
        number_of_elements=number_of_elements, 
        E_modulus=2.1e11, 
        shear_modulus=7.9e10,
        primary_moment_of_area=1.94e-8,
        secondary_moment_of_area=1.02e-8,
        torsional_constant=2.29e-8,
        density=7850, 
        cross_sectional_area=1.74e-4, 
        coordinates=[[0, 0, 0], [0, 0, 1.7]],
        edge_polar_rotation=0,
        point_mass=1.31,
        point_mass_moment_of_inertias=(0,0,0),
        point_mass_location='end_vertex'
    )
    model.fix_vertices((0,))
    model.set_damping_ratio(0.0015)
    
    return model

""" Determination of the default number of elements for the generate_original_model function. """
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    element_multipliers = np.arange(1, 6)

    norms = []
    for element_multiplier in element_multipliers:
        number_of_elements = 5 * element_multiplier

        model = generate_original_model(number_of_elements)

        output_DOFs = np.arange(1, 5) * element_multiplier * 6

        state_matrix, _, output_matrix, _, = model.get_state_space_matrices('receptence', output_DOFs=output_DOFs)

        eigen_values, eigen_vectors = np.linalg.eig(state_matrix)

        mode_shapes = output_matrix @ eigen_vectors

        norms.append(np.linalg.matrix_norm(mode_shapes[:, :30]))

    plt.plot(element_multipliers * 6 * 5, norms, marker='o')
    plt.title("Matrix norm of the first 30 modes as a function of DOF")
    plt.xlabel("DOF")
    plt.ylabel("$\\|C_c\\Psi_c\\|$")
    plt.grid()
    plt.show()