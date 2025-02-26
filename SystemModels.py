import numpy as np
import numpy.typing as npt
import igraph as ig
from matplotlib.axes import Axes
from collections.abc import Collection
from typing import Callable
from scipy.linalg import block_diag

class Beam_Lattice:
    """
    A finite element solver for an arbitrary structure of beam elements.

    Attributes
    ----------
    graph : igraph.Graph
        The graph structure of the lattice where each edge represents a straight set of beam elements and the vertices represent which sets of
        beam elements are connected. The graph has the following attributes:
            edge_attributes:
                edge_mass_matrix : numpy array 
                    The mass matrix of the entire edge rotated to the global context where the rows are orded in the direction of the edge including
                    the end vertices. The direction is from the lowest to highest ID of the connected vertices.
                edge_stiffness_matrix : numpy array
                    The stiffness matrix of the entire edge rotated to the global context where the rows are orded in the direction of the edge including
                    the end vertices. The direction is from the lowest to highest ID of the connected vertices.
                number_of_elements : int
                    The number of beam elements in the given edge.
                shape_function : Callable with float argument and numpy array return
                    The shape function for any beam element in the edge rotated to the global context.
                edge_vertices_coordinates : numpy array
                    The set of coordinates for all the nodes in the edge including the end vertices. Shape = (number_of_elements + 1, 2).
            vertex_attributes:
                coordinates : numpy array
                    The coordinates of the vertex in the global context with shape (2,).
    system_DOF : int
        The total DOF of the system.
    """
    def __init__(self) -> None:
        self.graph = ig.Graph()

    def add_beam_edge(self, number_of_elements: int, E_modulus: npt.ArrayLike, moment_of_area: npt.ArrayLike, density: npt.ArrayLike, 
                      cross_sectional_area:npt.ArrayLike, vertex_IDs: Collection[int, int] | int | None = None,
                      coordinates: npt.ArrayLike | None = None) -> None:
        """
        Adds an edge to the graph containing a straight set of beam elements or just a single beam elemet.

        Parameters
        ----------
        number_of_elements : int
            The number of beam elements to add for this edge.
        E_modulus : array_like
            The modulus of elasticity of the beam elements. If a scalar is specified, all beam elements will have this value. If two values
            are specified, a linear spacing between the two values are used.
        moment_of_area : array_like
            The moment of area of the beam elements. If a scalar is specified, all beam elements will have this value. If two values
            are specified, a linear spacing between the two values are used.
        density : array_like
            The density of the beam elements. If a scalar is specified, all beam elements will have this value. If two values
            are specified, a linear spacing between the two values are used.
        cross_sectional_area : array_like
            The cross sectional area of the beam elements in the direction of the beams. If a scalar is specified, all beam elements will have 
            this value. If two values are specified, a linear spacing between the two values are used.
        vertex_IDs : collection of two int or int, optional
            The vertex ID(s) that the beam connects to. Should not be provided for isolated beam elements. If two ID's are specified the
            direction of the beam element is from the lowest to higest ID. If a single ID is given, 'coordinates' parameter must contain the
            coordinate set for a new vertex where the beam direction will go towards the new vertex.
        coordinates : array_like, optional
            The coordinate set(s) of the vertices not specified using the parameter 'vertex_IDs'. Can have either shape (2,) or (2, 2) for 
            isolated beams. Ignored if both vertices are defined using 'vertex_IDs'.
        """
        # Creates the start and end vertices based on the given combination of 'vertex_IDs' and 'coordinates'.
        if isinstance(vertex_IDs, Collection):
            if len(vertex_IDs) != 2:
                raise ValueError(f"'vertex_IDs' expected 2 values when given as a collection.")
            start_vertex = self.graph.vs[min(vertex_IDs)]
            end_vertex = self.graph.vs[max(vertex_IDs)]
        elif isinstance(vertex_IDs, int):
            coordinates = np.asarray(coordinates)
            if coordinates.shape == (2,):
                end_vertex = self.graph.add_vertex(coordinates=coordinates)
            else:
                raise ValueError(f"'coordinates' vector have shape {coordinates.shape} but expected shape (2,) when specifying 1 vertex ID.")
            start_vertex = self.graph.vs[vertex_IDs]
        else:
            coordinates = np.asarray(coordinates)
            if coordinates.shape == (2, 2):
                start_vertex = self.graph.add_vertex(coordinates=coordinates[0])
                end_vertex = self.graph.add_vertex(coordinates=coordinates[1])
            else:
                raise ValueError(f"'Coordinates' have shape {coordinates.shape} but expected shape (2, 2) when not specifying 'vertex_IDs'")

        # Determines the beam properties for each beam element.
        beam_properties = list(np.atleast_1d(E_modulus, moment_of_area, density, cross_sectional_area))
        for i, beam_property in enumerate(beam_properties):
            if len(beam_property) == 1:
                beam_properties[i] = np.full(number_of_elements, beam_property[0])
            if len(beam_property) == 2:
                beam_properties[i] = np.linspace(beam_property[0], beam_property[1], number_of_elements)
            elif len(beam_property) > 2 and len(beam_property) != number_of_elements:
                raise ValueError(f"One of the material property vectors have {len(beam_property)} elements but expected either 1, 2 or {number_of_elements} elements.")
        E_modulus, moment_of_area, density, cross_sectional_area = beam_properties

        # Determines the coordinates for the element vectors.
        edge_vector = end_vertex['coordinates'] - start_vertex['coordinates']
        edge_vertices_coordinates = np.array([start_vertex['coordinates'] + edge_vector * i for i in np.linspace(0, 1, number_of_elements+1)])

        # Calculates the total DOF of the edge.
        edge_DOF = 6*number_of_elements-(number_of_elements-1)*3
        # Initializes the mass and stiffness matrices for the entire edge.
        edge_mass_matrix = np.zeros((edge_DOF, edge_DOF))
        edge_stiffness_matrix = edge_mass_matrix.copy()
        # Loops over all beam elements.
        for i in range(number_of_elements):
            # Short handing the beam properties.
            E, I, RHO, A = E_modulus[i], moment_of_area[i], density[i], cross_sectional_area[i]
            L = np.linalg.norm(edge_vector) / number_of_elements
            # Determines the mass matrix per beam element.
            element_mass_matrix = np.array([[140,     0,       0,  70,    0,       0],
                                            [  0,   156,    22*L,   0,   54,   -13*L],
                                            [  0,  22*L,  4*L**2,   0, 13*L, -3*L**2], 
                                            [ 70,     0,       0, 140,    0,       0],
                                            [  0,    54,    13*L,   0,  156,   -22*L],
                                            [  0, -13*L, -3*L**2,   0, -22*L, 4*L**2]])
            element_mass_matrix *= RHO*A*L/420
            # Determines the stiffness matrix per beam element.
            element_stiffness_matrix = np.array([[ E*A/L,            0,           0, -E*A/L,            0,           0],
                                                 [     0,  12*E*I/L**3,  6*E*I/L**2,      0, -12*E*I/L**3,  6*E*I/L**2],
                                                 [     0,   6*E*I/L**2,     4*E*I/L,      0,  -6*E*I/L**2,     2*E*I/L],
                                                 [-E*A/L,            0,           0,  E*A/L,            0,           0],
                                                 [     0, -12*E*I/L**3, -6*E*I/L**2,      0,  12*E*I/L**3, -6*E*I/L**2],
                                                 [     0,   6*E*I/L**2,     2*E*I/L,      0,  -6*E*I/L**2,     4*E*I/L]])
            # Determines the combined stiffness and mass matrix for the entire edge.
            element_pickoff_operator = np.zeros((6, edge_DOF), dtype=np.int8)
            element_pickoff_operator[:, i*3:i*3+6] = np.eye(6, dtype=np.int8)
            edge_mass_matrix += element_pickoff_operator.T @ element_mass_matrix @ element_pickoff_operator
            edge_stiffness_matrix += element_pickoff_operator.T @ element_stiffness_matrix @ element_pickoff_operator
        
        # Rotates the mass and stiffness matrices to the global context.
        edge_angle_cos, edge_angle_sin = edge_vector / np.linalg.norm(edge_vector)
        rotational_matrix = np.array([[edge_angle_cos, -edge_angle_sin, 0], 
                                      [edge_angle_sin,  edge_angle_cos, 0],
                                      [             0,               0, 1]])
        transformation_matrix = block_diag(*(rotational_matrix.T,)*(number_of_elements+1))
        edge_mass_matrix = transformation_matrix.T @ edge_mass_matrix @ transformation_matrix
        edge_stiffness_matrix = transformation_matrix.T @ edge_stiffness_matrix @ transformation_matrix
        
        # Determines the shape function.
        shape_function: Callable[[float], npt.NDArray] = lambda x: rotational_matrix[:2,:2] @ \
                                                                   np.array([[1-x,               0,                   0, x,             0,              0],
                                                                             [  0, 1-3*x**2+2*x**3, x*L-2*L*x**2+L*x**3, 0, 3*x**2-2*x**3, -L*x**2+L*x**3]]) \
                                                                   @ block_diag(rotational_matrix, rotational_matrix).T
        
        # Adds the beam into the graph.
        self.graph.add_edge(start_vertex, end_vertex, 
                            edge_mass_matrix=edge_mass_matrix, 
                            edge_stiffness_matrix=edge_stiffness_matrix,
                            number_of_elements=number_of_elements,
                            shape_function=shape_function,
                            edge_vertices_coordinates=edge_vertices_coordinates)

    @property
    def system_DOF(self) -> int:
        """
        The total DOF of the system.
        """
        return 3*(np.sum(self.graph.es['number_of_elements'], dtype=int) + self.graph.vcount() - self.graph.ecount())

    def get_system_level_matrices(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Calculates the system-level mass- and stiffness matrices by combining all edge mass- and stiffness matrices.
        
        Returns
        -------
        tuple of two numpy arrays
            The first element is the system-level mass matrix and the second is the stiffness matrix. The order of the rows
            in each matrix is first vertices followed by the nodes. The vertices are by them selves orded by their
            respective ID. The nodes are firstly ordered by their respective edge ID and secondly by the direction of the edge.
        """
        # Calculates the number of DOF in the entire system.
        system_DOF = self.system_DOF
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

    def get_static_vertex_and_node_displacements(self, forces: dict[int, npt.ArrayLike], fixed_vertex_IDs: tuple[int, ...]) -> npt.NDArray:
        """
        Gets the displacement for all vertices and nodes under a static load. The displaced position is not calculated.

        Parameters
        ----------
        forces : dict with int key and array_like value
            The point forces applied to the system. The key values are the vertices where the point forces are applied and the
            values must be an array with shape (3,).
        fixed_vertex_IDs : tuple of int
            The vertex IDs that will have a fixed boundary condition.

        Returns
        -------
        numpy array
            The displacement of each vertex and node with shape (3*n,) where n is the number of total vertices and nodes.
            The array is ordered as (vertex displacements, nodal displacements) where the vertex displacements are order
            according to their ID and the nodal displacements are ordered firstly by their corresponding edge ID and secondly
            according to the direction of the edge. Each vertex/nodal displacement is then orded by (x, y, phi) displacement.
        """
        _, stiffness_matrix =  self.get_system_level_matrices()

        # Applies boundary conditions.
        fixed_DOFs = np.ravel([(3*fixed_vertex_ID, 
                                3*fixed_vertex_ID+1, 
                                3*fixed_vertex_ID+2) for fixed_vertex_ID in fixed_vertex_IDs])
        stiffness_matrix = np.delete(stiffness_matrix, fixed_DOFs, axis=0)
        stiffness_matrix = np.delete(stiffness_matrix, fixed_DOFs, axis=1)

        # Constructs the force vector.
        force_vector = np.zeros(self.system_DOF)
        for vertex_ID, force in forces.items():
            if vertex_ID in fixed_vertex_IDs:
                raise ValueError(f"Vertex {vertex_ID} is fixed and cannot have a force applied to it.")
            elif vertex_ID >= self.graph.vcount():
                raise ValueError(f"Force is trying to be applied to vertex ID {vertex_ID} but this vertex doesn't exist.")
            force_vector[3*vertex_ID:3*vertex_ID + 3] = np.asarray(force)
        force_vector = np.delete(force_vector, fixed_DOFs)

        # Calculates the displacements.
        displacements = np.linalg.inv(stiffness_matrix) @ force_vector

        # Inserts the fixed DOF back again.
        for fixed_DOF in fixed_DOFs:
            displacements = np.insert(displacements, fixed_DOF, 0.0)
        
        return displacements

    def get_displaced_vertices_and_node_position(self, forces: dict[int, npt.ArrayLike], fixed_vertex_IDs: tuple[int, ...]) -> list[npt.NDArray]:
        """
        Calculates the displaced position of each vertex and node in the system under a given static load.

        Parameters
        ----------
        forces : dict with int key and array_like value
            The point forces applied to the system. The key values are the vertices where the point forces are applied and the
            values must be an array with shape (3,).
        fixed_vertex_IDs : tuple of int
            The vertex IDs that will have a fixed boundary condition.

        Returns
        -------
        list of numpy array
            A list of 2D arrays with the displaced positions of all vertices and nodes. Each element in the list corrsponds to an edge and each
            array has the shape (number_of_elements + 2, 3) where number_of_elements refer to the given edge and plus two to include the source
            and target vertices of the edge.
        """
        # Gets the displacement for all vertices and nodes.
        vertex_and_node_displacements = self.get_static_vertex_and_node_displacements(forces, fixed_vertex_IDs)
        vertex_displacements, node_displacements = np.split(vertex_and_node_displacements, [3*self.graph.vcount()])
        # Initializes array for the displaced position of all vertices and nodes along any edge in the order of the edge direction. 
        # Shape: (number of edges, number of vertex and nodes for the given edge).
        vertex_and_node_displaced_positions = list()
        
        accumulative_edge_DOF = 0
        # Loops over all edges.
        for edge in self.graph.es:
            # Calculates the number of DOF in the given edge excluding the DOF's in its target and source vertices.
            edge_DOF = 3*(edge['number_of_elements']-1)
            # Calculates the displaced position for the source and target vertex of the edge.
            source_vertex_displaced_position = vertex_displacements[3*edge.source : 3*edge.source + 3] + np.concatenate([self.graph.vs[edge.source]['coordinates'], [0]])
            target_vertex_displaced_position = vertex_displacements[3*edge.target : 3*edge.target + 3] + np.concatenate([self.graph.vs[edge.target]['coordinates'], [0]])
            # Gets all node displacements for the current edge.
            edge_node_displacements = node_displacements[accumulative_edge_DOF : accumulative_edge_DOF + edge_DOF]
            accumulative_edge_DOF += edge_DOF
            # Calculates all node displaced positions for the current edge.
            edge_node_displaced_positions = edge_node_displacements.reshape((-1, 3)) + np.column_stack((edge['edge_vertices_coordinates'][1:-1], np.zeros(edge['number_of_elements']-1)))
            # Combines the vertices and nodal displacements in the order of the edge direction.
            vertex_and_node_displaced_positions.append(np.vstack((source_vertex_displaced_position, edge_node_displaced_positions, target_vertex_displaced_position)))
            
        return vertex_and_node_displaced_positions

    def get_displaced_shape_position(self, forces: dict[int, npt.ArrayLike], fixed_vertex_IDs: tuple[int, ...], resolution_per_element: int = 100) -> list[npt.NDArray]:
        """
        Calculates the shape of each beam element.

        Parameters
        ----------
        forces : dict with int key and array_like value
            The point forces applied to the system. The key values are the vertices where the point forces are applied and the
            values must be an array with shape (3,).
        fixed_vertex_IDs : tuple of int
            The vertex IDs that will have a fixed boundary condition.
        resolution_per_element : int, optional
            The number of points per element per edge (Default 100).

        Returns
        -------
        list of numpy array
            A list of 2D arrays with the coordinates of all shaped beam elements. Each element in the list corresponds to an edge and each
            array has the shape (resolution_per_element*number_of_elements, 2) where the number_of_elements refer to the corresponding
            edge. The coordinates are orded along the edge direction.
        """
        # Gets the displacement for all vertices and nodes.
        vertex_and_node_displacements = self.get_static_vertex_and_node_displacements(forces, fixed_vertex_IDs)
        vertex_displacements, node_displacements = np.split(vertex_and_node_displacements, [3*self.graph.vcount()])
        # Initializes array for the displacements of all vertices and nodes along any edge in the order of the edge direction. 
        # Shape: (number of edges, number of vertex and nodes for the given edge).
        edge_vertex_and_node_displacements: list[npt.NDArray] = list()
        # Initializes return array for the displaced positions of all the points along any edge.
        # Shape: (number of edges, number of elements * resolution per element for the given edge).
        edge_point_displaced_positions: list[npt.NDArray] = list()
        # Calculates the normalized distances along any element where the position of the displaced point will be evaluated.
        normalized_distances_along_element = np.linspace(0, 1, resolution_per_element)

        accumulative_edge_DOF = 0
        # Looping over all edges.
        for i, edge in enumerate(self.graph.es):
            # Calculates the number of DOF in the given edge excluding the DOF's in its target and source vertices.
            edge_DOF = 3*(edge['number_of_elements']-1)
            # Gets the displacements for the source and target vertex of the edge.
            source_vertex_displacement = vertex_displacements[3*edge.source : 3*edge.source + 3]
            target_vertex_displacement = vertex_displacements[3*edge.target : 3*edge.target + 3]
            # Gets the node displacements of the current edge.
            edge_node_displacements = node_displacements[accumulative_edge_DOF : accumulative_edge_DOF + edge_DOF].reshape(-1, 3)
            accumulative_edge_DOF += edge_DOF
            # Combines the vertices and nodal displacements in the order of the edge direction.
            edge_vertex_and_node_displacements.append(np.vstack((source_vertex_displacement, edge_node_displacements, target_vertex_displacement)))
            # Initializes the array that contains all the displaced positions of the current edge.
            edge_point_displaced_positions.append(np.empty((edge['number_of_elements'] * resolution_per_element, 2)))
            # Looping over all elements in each edge.
            for j in range(edge['number_of_elements']):
                # Gets the source and target node displacement of the current element.
                element_node_displacement = np.ravel(edge_vertex_and_node_displacements[i][j:j+2])
                # Gets the nominal position of the source and target nodes of the current element.
                element_source_position = edge['edge_vertices_coordinates'][j]
                element_target_position = edge['edge_vertices_coordinates'][j+1]
                element_vector = element_target_position - element_source_position
                # Looping over all points within each element within each edge.
                for k, normalized_distance_along_element in enumerate(normalized_distances_along_element):
                    # Calculates the nominal position of the current point.
                    point_position = element_source_position + element_vector * normalized_distance_along_element
                    # Calculates the displacement of the current point using the edge's shape function.
                    point_displacement = edge['shape_function'](normalized_distance_along_element) @ element_node_displacement
                    # Calculates the displaced position of the point and stores it into the return array.
                    edge_point_displaced_positions[i][j*resolution_per_element + k] = point_displacement + point_position

        return edge_point_displaced_positions

    def plot_lattice(self, ax: Axes, **kwargs) -> None:
        ig.plot(self.graph, target=ax, layout=self.graph.vs['coordinates'], **kwargs)

# Example.
if __name__ == "__main__":
    # Beam lattice example.
    import matplotlib.pyplot as plt
    
    ax = plt.subplot()
    
    for i in range(1, 10):
            
        beam_lattice = Beam_Lattice()
        
        beam_lattice.add_beam_edge(
            number_of_elements=i, 
            E_modulus=1e11, 
            moment_of_area=[8.33e-10, 1e-11], 
            density=1, 
            cross_sectional_area=0.01**2, 
            coordinates=[[0, 0], [1, 0]]
        )

        displaced_shape_points = beam_lattice.get_displaced_shape_position({1: [0, -10, 0]}, (0,))

        for displaced_shape_point in displaced_shape_points:
            ax.plot(displaced_shape_point[:, 0], displaced_shape_point[:, 1], linewidth=2.0, linestyle='--', label=i)
    beam_lattice.plot_lattice(ax, vertex_label=np.arange(beam_lattice.graph.vcount()), vertex_size=10, vertex_label_dist=3, vertex_label_angle=0.4)
    ax.axis('equal')
    ax.grid()
    plt.legend()
    plt.show()