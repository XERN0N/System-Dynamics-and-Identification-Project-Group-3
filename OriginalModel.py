from SystemModels import Beam_Lattice
from enum import Enum

class Vertex_DOFs(Enum):
    output_primary_DOFs = (1, 7, 19, 31)
    output_secondary_DOFs = (0, 6, 18, 30)
    input_primary_DOFs = (13, 25, 37)
    input_secondary_DOFs = (12, 24, 36)

def generate_original_model(number_of_elements: int = 2, E_modulus: float = 2.1e11, density: float = 7850) -> Beam_Lattice:
    """
    Generates the original model "before" running the SSI.
    Sources:
        [1] Sigurd Mousten Jager Nielsens filer - System Dynamics and Identification Project/Beam properties/RHS_30x20x2.pdf
    
    Parameters
    ----------
    number_of_elements : int, optional
        The number of elements between each vertex where a vertex is placed at each input and output location.
    E_modulus : float, optional
        The modulus of elasticity for the beam [Pa]. Default 210 GPa.
    density : float, optional
        The density of the beam material [kg/m^3]. Default 7850 kg/m^3.

    Returns
    -------
    Beam_Lattice
        The model.
    """
    model = Beam_Lattice()

    vertex_heights = (0.34, 0.68, 0.81, 1.02, 1.24, 1.36, 1.60, 1.715)

    # Start beam.
    model.add_beam_edge(
        number_of_elements=number_of_elements, 
        E_modulus=E_modulus, 
        shear_modulus=7.9e10,
        primary_moment_of_area=1.94e-8,
        secondary_moment_of_area=1.02e-8,
        torsional_constant=2.29e-8,
        density=density, 
        cross_sectional_area=1.74e-4, 
        coordinates=((0, 0, 0), (0, 0, vertex_heights[0])),
        edge_polar_rotation=0
    )

    # Bulk beams.
    for i, vertex_height in enumerate(vertex_heights[1:-1]):
        model.add_beam_edge(
            number_of_elements=number_of_elements, 
            E_modulus=E_modulus, 
            shear_modulus=7.9e10,
            primary_moment_of_area=1.94e-8,
            secondary_moment_of_area=1.02e-8,
            torsional_constant=2.29e-8,
            density=density, 
            cross_sectional_area=1.74e-4, 
            coordinates=(0, 0, vertex_height),
            vertex_IDs=i+1,
            edge_polar_rotation=0
        )

    # End beam.
    model.add_beam_edge(
        number_of_elements=number_of_elements, 
        E_modulus=E_modulus, 
        shear_modulus=7.9e10,
        primary_moment_of_area=1.94e-8,
        secondary_moment_of_area=1.02e-8,
        torsional_constant=2.29e-8,
        density=density, 
        cross_sectional_area=1.74e-4, 
        coordinates=(0, 0, vertex_heights[-1]),
        vertex_IDs=7,
        edge_polar_rotation=0,
        point_mass=1.31,
        point_mass_moment_of_inertias=(0.0004384, 0.0011301, 0.000861),
        point_mass_location='end_vertex'
    )

    model.fix_vertices((0,))
    model.set_damping_ratio(0.0013)

    return model

""" Determination of the default number of elements for the generate_original_model function. """
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    number_of_elements = np.arange(1, 8)

    norms = []
    DOFs = []
    for number_of_element in number_of_elements:
        model_primary = generate_original_model(number_of_element)

        state_matrix, _, output_matrix, _, = model_primary.get_state_space_matrices('receptence', output_DOFs=np.concatenate((Vertex_DOFs.output_primary_DOFs.value, 
                                                                                                                      Vertex_DOFs.output_secondary_DOFs.value)))

        eigen_values, eigen_vectors = np.linalg.eig(state_matrix)

        mode_shapes = output_matrix @ eigen_vectors
        
        norms.append(np.linalg.matrix_norm(mode_shapes[:, :5]))
        DOFs.append(model_primary.system_DOF - len(model_primary.fixed_DOFs))

    plt.plot(DOFs, norms, marker='o')
    plt.title("Matrix norm of the first 5 modes as a function of DOF")
    plt.xlabel("DOF")
    plt.ylabel("$\\|C_c\\Psi_c\\|$")
    plt.grid()
    plt.show()