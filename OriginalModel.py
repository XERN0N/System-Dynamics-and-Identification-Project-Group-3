from SystemModels import Beam_Lattice
from enum import Enum

class Vertex_DOFs(Enum):
    output_primary_DOFs = (1, 7, 19, 31)
    output_secondary_DOFs = (0, 6, 18, 30)
    input_primary_DOFs = (13, 25, 37)
    input_secondary_DOFs = (12, 24, 36)

class Default_beam_edge_parameters(Enum):
    default_beam_edge_parameters = {
        'number_of_elements': 2, 
        'E_modulus': 2.1e11, 
        'shear_modulus': 7.9e10,
        'primary_moment_of_area': 1.94e-8,
        'secondary_moment_of_area': 1.02e-8,
        'torsional_constant': 2.29e-8,
        'density': 7850, 
        'cross_sectional_area': 1.74e-4, 
        'edge_polar_rotation': 0
    }

    default_point_mass_parameters = {
        'point_mass': 1.31,
        'point_mass_moment_of_inertias': (0, 0, 0),
        'point_mass_location': 'end_vertex'
    }

def generate_original_model(**beam_edge_parameters) -> Beam_Lattice:
    """
    Generates the original model "before" running the SSI.
    Sources:
        [1] Sigurd Mousten Jager Nielsens filer - System Dynamics and Identification Project/Beam properties/RHS_30x20x2.pdf
    
    Parameters
    ----------
    beam_edge_parameters : kwargs, optional
        Parameters for the Beam_Lattice method 'add_beam_edge'. If certain parameters are not specified, default values are used.
        Parameters given that are not in the method are ignored.

    Returns
    -------
    Beam_Lattice
        The model.
    """

    default_beam_edge_parameters = Default_beam_edge_parameters.default_beam_edge_parameters.value

    default_point_mass_parameters = Default_beam_edge_parameters.default_point_mass_parameters.value

    for key, value in beam_edge_parameters.items():
        if key in default_beam_edge_parameters.keys():
            default_beam_edge_parameters.update({key: value})
        elif key in default_point_mass_parameters.keys():
            default_point_mass_parameters.update({key: value})

    model = Beam_Lattice()

    vertex_heights = (0.34, 0.68, 0.81, 1.02, 1.24, 1.36, 1.60, 1.715)

    # Start beam.
    model.add_beam_edge( 
        coordinates=((0, 0, 0), (0, 0, vertex_heights[0])),
        **default_beam_edge_parameters
    )

    # Bulk beams.
    for i, vertex_height in enumerate(vertex_heights[1:-1]):
        model.add_beam_edge( 
            coordinates=(0, 0, vertex_height),
            vertex_IDs=i+1,
            **default_beam_edge_parameters
        )

    # End beam.
    model.add_beam_edge(
        coordinates=(0, 0, vertex_heights[-1]),
        vertex_IDs=7,
        **default_point_mass_parameters,
        **default_beam_edge_parameters
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