import numpy as np
import numpy.typing as npt
import igraph as ig
from matplotlib.axes import Axes
from typing import Literal
from scipy.linalg import block_diag

class Beam_Lattice:

    def __init__(self) -> None:
        self.graph = ig.Graph()

    def add_beam_edge(self,
                      number_of_elements: int,
                      E_modulus: float, 
                      moment_of_area: float, 
                      density: float, 
                      cross_sectional_area:float,
                      start_vertex_ID: int | None = None,
                      end_vertex_ID: int | None = None,
                      coordinates: npt.ArrayLike | None = None 
                      ) -> ig.Graph:

        # Creates the neccesary vertices.
        if start_vertex_ID is None:
            coordinates = np.asarray(coordinates)
            if coordinates.shape == (2,):
                start_vertex = self.graph.add_vertex(coordinates=coordinates)
            elif coordinates.shape == (2, 2):
                start_vertex = self.graph.add_vertex(coordinates=coordinates[0])
            else:
                raise ValueError("'coordinates' vector have wrong size.")
        else:
            start_vertex = self.graph.vs[start_vertex_ID]
                
        if end_vertex_ID is None:
            coordinates = np.asarray(coordinates)
            if coordinates.shape == (2,):
                end_vertex = self.graph.add_vertex(coordinates=coordinates)
            elif coordinates.shape == (2, 2):
                end_vertex = self.graph.add_vertex(coordinates=coordinates[1])
            else:
                raise ValueError("'coordinates' vector have wrong size.")
        else:
            end_vertex = self.graph.vs[end_vertex_ID]

        edge_vector = end_vertex['coordinates'] - start_vertex['coordinates']
        L = np.linalg.norm(edge_vector) / number_of_elements
        E = E_modulus
        I = moment_of_area
        A = cross_sectional_area

        # Determines the mass matrix per beam element.
        element_mass_matrix = np.array([[140,     0,       0,  70,    0,       0],
                                        [  0,   156,    22*L,   0,   54,   -13*L],
                                        [  0,  22*L,  4*L**2,   0, 13*L, -3*L**2], 
                                        [ 70,     0,       0, 140,    0,       0],
                                        [  0,    54,    13*L,   0,  156,   -22*L],
                                        [  0, -13*L, -3*L**2,   0, -22*L, 4*L**2]])
        element_mass_matrix *= density*A*L/420

        # Determines the stiffness matrix per beam element.
        element_stiffness_matrix = np.array([[ E*A/L,            0,           0, -E*A/L,            0,           0],
                                             [     0,  12*E*I/L**3,  6*E*I/L**2,      0, -12*E*I/L**3,  6*E*I/L**2],
                                             [     0,   6*E*I/L**2,     4*E*I/L,      0,  -6*E*I/L**2,     2*E*I/L],
                                             [-E*A/L,            0,           0,  E*A/L,            0,           0],
                                             [     0, -12*E*I/L**3, -6*E*I/L**2,      0,  12*E*I/L**3, -6*E*I/L**2],
                                             [     0,   6*E*I/L**2,     2*E*I/L,      0,  -6*E*I/L**2,     4*E*I/L]])
        
        # Determines the coordinates for the element vectors.
        edge_vertices_coordinates = np.array([start_vertex['coordinates'] + edge_vector / np.linalg.norm(edge_vector) * i*L for i in range(1, number_of_elements)])

        # Determines the combined stiffness and mass matrix for the entire edge.
        edge_DOF = 6*number_of_elements-(number_of_elements-1)*3
        edge_mass_matrix = np.zeros((edge_DOF, edge_DOF))
        edge_stiffness_matrix = edge_mass_matrix.copy()
        for i in range(number_of_elements):
            element_pickoff_operator = np.zeros((6, edge_DOF), dtype=np.int8)
            element_pickoff_operator[:, i*3:i*3+6] = np.eye(6, dtype=np.int8)
            edge_mass_matrix += element_pickoff_operator.T @ element_mass_matrix @ element_pickoff_operator
            edge_stiffness_matrix += element_pickoff_operator.T @ element_stiffness_matrix @ element_pickoff_operator
        
        # Rotates the mass and stiffness matrices to the global context.
        edge_angle_cos, edge_angle_sin = edge_vector / np.linalg.norm(edge_vector)
        rotational_matrix = np.array([[edge_angle_cos, -edge_angle_sin, 0], 
                                      [edge_angle_sin,  edge_angle_cos, 0],
                                      [             0,               0, 1]])
        transformation_matrix = block_diag(*(rotational_matrix,)*(number_of_elements+1))
        edge_mass_matrix = transformation_matrix.T @ edge_mass_matrix @ transformation_matrix
        edge_stiffness_matrix = transformation_matrix.T @ edge_stiffness_matrix @ transformation_matrix
        
        # Determines the shape function.
        shape_function = lambda x: transformation_matrix @ np.array([[1-x/L, 0, 0, x/L, 0, 0],
                                                                     [0, 1-3*(x/L)**2+2*(x/L)**3, 
                                                                      x-2*L*(x/L)**2+L*(x/L)**3, 0, 
                                                                      3*(x/L)**2-2*(x/L)**3, 
                                                                      -L*(x/L)**2+L*(x/L)**3]]) @ transformation_matrix.T
        
        # Adds the beam into the graph.
        self.graph.add_edge(end_vertex, start_vertex, 
                            edge_mass_matrix=edge_mass_matrix, 
                            edge_stiffness_matrix=edge_stiffness_matrix,
                            number_of_elements=number_of_elements,
                            shape_function=shape_function,
                            edge_vertices_coordinates=edge_vertices_coordinates)

    def get_system_DOF(self) -> int:
        return 3*(np.sum(self.graph.es['number_of_elements'], dtype=int) + self.graph.vcount() - self.graph.ecount())

    def get_system_level_matrices(self) -> tuple[npt.NDArray, npt.NDArray]:
        # Calculates the number of DOF in the entire system.
        system_DOF = self.get_system_DOF()
        
        # Initializes the system level mass and stiffness matrices.
        system_mass_matrix = np.zeros((system_DOF, system_DOF))
        system_stiffness_matrix = system_mass_matrix.copy()
        
        # Initializes the column index for the first edge in the edge pickoff operator.
        accumulative_edge_DOF = 3*self.graph.vcount()

        # Loops over all edges to add each edge contribution to the system level matrices.
        for edge in self.graph.es:
            # Calculates the number of DOF in each edge excluding the DOF's in its target and source vertices.
            edge_DOF = 3*(edge['number_of_elements']-1)

            # Initializes the edge pickoff operator where the first set of columns will represent the DOF's in the vertices
            # and the second set of columns will represent the DOF's in the edges alone.
            edge_pickoff_operator = np.zeros((edge_DOF + 6, system_DOF), dtype=np.int8)

            # Picking the correct indices for the edge DOF's.
            edge_pickoff_operator[3:-3, accumulative_edge_DOF:accumulative_edge_DOF+edge_DOF] = np.eye(edge_DOF, dtype=np.int8)
            
            # Picking the correct indices for the vertices DOF's.
            edge_pickoff_operator[ :3, 3*edge.source:3*edge.source + 3] = np.eye(3, dtype=np.int8)
            edge_pickoff_operator[-3:, 3*edge.target:3*edge.target + 3] = np.eye(3, dtype=np.int8)

            # Adding the edge contributions to the system level matrices.
            system_mass_matrix += edge_pickoff_operator.T @ edge['edge_mass_matrix'] @ edge_pickoff_operator
            system_stiffness_matrix += edge_pickoff_operator.T @ edge['edge_stiffness_matrix'] @ edge_pickoff_operator
            
            accumulative_edge_DOF += edge_DOF

        return system_mass_matrix, system_stiffness_matrix

    def displace(self, forces: dict[int, npt.ArrayLike], fixed_vertex_IDs: tuple[int, ...]) -> list[npt.NDArray]:
        _, stiffness_matrix =  self.get_system_level_matrices()

        # Applies boundary conditions.
        fixed_DOFs = np.ravel([(3*fixed_vertex_ID, 
                                3*fixed_vertex_ID+1, 
                                3*fixed_vertex_ID+2) for fixed_vertex_ID in fixed_vertex_IDs])
        stiffness_matrix = np.delete(stiffness_matrix, fixed_DOFs, axis=0)
        stiffness_matrix = np.delete(stiffness_matrix, fixed_DOFs, axis=1)

        # Constructs the force vector.
        force_vector = np.zeros(self.get_system_DOF())
        for vertex_ID, force in forces.items():
            if vertex_ID in fixed_vertex_IDs:
                raise ValueError(f"Vertex {vertex_ID} is fixed and cannot have a force applied to it.")
            force_vector[3*vertex_ID:3*vertex_ID + 3] = np.asarray(force)
        force_vector = np.delete(force_vector, fixed_DOFs)

        # Calculates the displacements.
        displacements = np.linalg.inv(stiffness_matrix) @ force_vector

        # Inserts the fixed DOf back again.
        for fixed_DOF in fixed_DOFs:
            displacements = np.insert(displacements, fixed_DOF, 0.0)
        
        vertex_displacements, edge_displacements = np.split(displacements, [3*self.graph.vcount()])
        
        new_coordinates = list()
        # Initializes the column index for the first edge in the edge pickoff operator.
        accumulative_edge_DOF = 0
        for edge in self.graph.es:
            # Calculates the number of DOF in each edge excluding the DOF's in its target and source vertices.
            edge_DOF = 3*(edge['number_of_elements']-1)
            source_vertex_new_coordinates = vertex_displacements[3*edge.source:3*edge.source+2] + self.graph.vs[edge.source]['coordinates']
            target_vertex_new_coordinates = vertex_displacements[3*edge.target:3*edge.target+2] + self.graph.vs[edge.target]['coordinates']
            edge_vertices_coordinates = edge_displacements[accumulative_edge_DOF:accumulative_edge_DOF+edge_DOF]
            edge_vertices_new_coordinates = np.array([edge_vertices_coordinates[::3], edge_vertices_coordinates[1::3]]).T + edge['edge_vertices_coordinates']
            new_coordinates.append(np.vstack((source_vertex_new_coordinates, edge_vertices_new_coordinates, target_vertex_new_coordinates)))
            accumulative_edge_DOF += edge_DOF

        return new_coordinates

    def plot_lattice(self, ax: Axes, **kwargs) -> None:
        ig.plot(self.graph, target=ax, layout=self.graph.vs['coordinates'], **kwargs)

def chain(masses: npt.ArrayLike, springs: npt.ArrayLike, dampers: npt.ArrayLike = None) -> tuple[npt.NDArray, npt.NDArray] | tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Creates the mass and spring matrix with size n and also the damper matrix if supplied.

    Parameters
    ----------
    masses : array_like with shape (n,)
        The mass of the lumped masses in the system.
    springs : array_like with shape (n,) | (n+1,)
        The spring constants between the lumped masses.
    dampers : array_like with shape (n,) | (n+1,), optional
        The damper constants of the dampers in between. Ignored by default.

    Returns
    -------
    tuple of np arrays with shape (n, n)
        Contains the mass matrix, spring matrix, and damper matrix if provided.
    """
    springs = np.asarray(springs)
    matrix_size = len(masses)
    
    # Creating mass matrix.
    mass_matrix = np.eye(matrix_size)
    np.fill_diagonal(mass_matrix, masses)
    
    def create_matrix(actuator: npt.NDArray) -> npt.NDArray:
        matrix = np.zeros((matrix_size, matrix_size))
        diag_indicies = np.diag_indices_from(matrix)
        # If there is no actuator on the end.
        if matrix_size == len(actuator):
            # Main diagonal.
            matrix[diag_indicies] = actuator + np.concatenate((actuator[1:], [0]))
            # Sub diagonals.
            matrix[1:, :-1][np.diag_indices(matrix_size - 1)] = -actuator[1:]
            matrix[:-1, 1:][np.diag_indices(matrix_size - 1)] = -actuator[1:]
        # If there is an actuator on the end.
        elif matrix_size + 1 == len(actuator):
            # Main diagonals.
            matrix[diag_indicies] = actuator[:-1] + actuator[1:]
            # Sub diagonals.
            matrix[1:, :-1][np.diag_indices(matrix_size - 1)] = -actuator[1:-1]
            matrix[:-1, 1:][np.diag_indices(matrix_size - 1)] = -actuator[1:-1]
        else:
            raise ValueError(f"Length of mass and spring vectors aren't compatible (len(masses) = {matrix_size}, len(springs) = {len(actuator)}).")
        return matrix
        
    # Creating the stiffness matrix.
    stiffness_matrix = create_matrix(springs)
    if dampers is not None:
        # Creating the damper matrix.
        dampers = np.asarray(dampers)
        damper_matrix = create_matrix(dampers)
        return mass_matrix, stiffness_matrix, damper_matrix
    
    return mass_matrix, stiffness_matrix

# Example.
if __name__ == "__main__":
    # Beam lattice example.
    import matplotlib.pyplot as plt
    beam_lattice = Beam_Lattice()
    beam_lattice.add_beam_edge(
        number_of_elements=1, 
        E_modulus=2e11, 
        moment_of_area=8.33e-10, 
        density=1, 
        cross_sectional_area=0.01**2, 
        coordinates=[[0, 0], [1, 0]]
    )

    new_coordinates = beam_lattice.displace({1: [0, 1, 0]}, (0,))

    ax = plt.subplot()
    ax.axis('equal')
    # beam_lattice.plot_lattice(ax, vertex_label=np.arange(beam_lattice.graph.vcount()))
    for new_coordinate in new_coordinates:
        ax.plot(new_coordinate[:, 0], new_coordinate[:, 1], color='red', linewidth=2.0, linestyle='--')
        beam_lattice.plot_lattice(ax, vertex_size=5)
    ax.grid()
    print(beam_lattice.get_system_DOF())
    plt.show()


    # chain example.
    """
    masses = [1, 2, 3]
    springs = [3, 5, 2]
    dampers = [5, 3, 7, 8]
    matrices = chain(masses, springs, dampers)
    print(f"Mass matrix:\n{matrices[0]}\n")
    print(f"Spring matrix:\n{matrices[1]}\n")
    print(f"Damper matrix:\n{matrices[2]}")
    """