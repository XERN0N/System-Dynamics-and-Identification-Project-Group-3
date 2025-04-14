import numpy as np
import numpy.typing as npt
from typing import Callable
import igraph as ig
from collections.abc import Collection, Generator
from scipy.linalg import block_diag, eig
from warnings import deprecated
from contextlib import contextmanager

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
                    The set of coordinates for all the nodes in the edge including the end vertices. Shape = (number_of_elements + 1, 3).
            vertex_attributes:
                coordinates : numpy array
                    The coordinates of the vertex in the global context with shape (3,).
                fixed : bool
                    Whether or not the vertex have a fixed boundary condition applied to it.
                force : None or Callable with float input and numpy array output
                    The force function applied to the vertex with time as input and the force vector with shape (6,) as output.
                    None if no force is applied.
    system_DOF : int
        The total DOF of the system.
    fixed_DOFs : numpy array
        A list of the DOF's that have a fixed boundary condition applied to.
    """
    def __init__(self) -> None:
        self.graph = ig.Graph()

    def add_beam_edge(self, number_of_elements: int, E_modulus: npt.ArrayLike, shear_modulus: npt.ArrayLike, primary_moment_of_area: npt.ArrayLike, 
                      secondary_moment_of_area: npt.ArrayLike, torsional_constant: npt.ArrayLike, density: npt.ArrayLike, 
                      cross_sectional_area: npt.ArrayLike, vertex_IDs: Collection[int, int] | int | None = None, coordinates: npt.ArrayLike | None = None, 
                      edge_polar_rotation: float | None = None) -> None:
        """
        Adds an edge to the graph containing a straight set of beam elements or just a single beam elemet.

        Parameters
        ----------
        number_of_elements : int
            The number of beam elements to add for this edge.
        E_modulus : array_like
            The modulus of elasticity of the beam elements. If a scalar is specified, all beam elements will have this value. If two values
            are specified, a linear spacing between the two values are used.
        shear_modulus : array_like
            The shear modulus of the beam elemets. If scalar, all beam elements will have this value. If two values are specified,
            a linear spacing between the two values are used.
        primary_moment_of_area : array_like
            The primary moment of area(s) for the beam (around the z-axis) with shape (n,). If the size of n is number_of_elements then the 
            values in the array corresponds to the value for each element of the beam. If n is a scaler all beam elements will have the same 
            value, and if n has two values, a linear spacing between the two values are used.
        secondary_moment_of_area : array_like
            The secondary moment of area(s) for the beam (around the y-axis) with shape (n,). If the size of n is number_of_elements then the 
            values in the array corresponds to the value for each element of the beam. If n is a scaler all beam elements will have the same 
            value, and if n has two values, a linear spacing between the two values are used.
        torsional_constant : array_like
            The torsional constant for the beam (around the x-axis) with shape (n,). If the size of n is number_of_elements then the 
            values in the array corresponds to the value for each element of the beam. If n is a scaler all beam elements will have the same 
            value, and if n has two values, a linear spacing between the two values are used.
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
            The coordinate set(s) of the vertices not specified using the parameter 'vertex_IDs'. Can have either shape (3,) or (2, 3) for 
            isolated beams. Ignored if both vertices are defined using 'vertex_IDs'.
        edge_polar_rotation : float, optional
            Rotation of the beam along the beam axis [rad]. The order of rotation is polar-primary-secondary (x-z-y) so this is the first
            rotation applied to the beam. Default 0.
        """
        # Additional parameters for the vertices.
        additional_vertex_parameters = {
            'fixed': False, 
            'force': None
        }

        # Creates the start and end vertices based on the given combination of 'vertex_IDs' and 'coordinates'.
        if isinstance(vertex_IDs, Collection):
            if len(vertex_IDs) != 2:
                raise ValueError(f"'vertex_IDs' expected 2 values when given as a collection.")
            start_vertex = self.graph.vs[min(vertex_IDs)]
            end_vertex = self.graph.vs[max(vertex_IDs)]
        elif isinstance(vertex_IDs, int):
            coordinates = np.asarray(coordinates)
            if coordinates.shape == (3,):
                end_vertex = self.graph.add_vertex(coordinates=coordinates, **additional_vertex_parameters)
            else:
                raise ValueError(f"'coordinates' vector have shape {coordinates.shape} but expected shape (3,) when specifying 1 vertex ID.")
            start_vertex = self.graph.vs[vertex_IDs]
        else:
            coordinates = np.asarray(coordinates)
            if coordinates.shape == (2, 3):
                start_vertex = self.graph.add_vertex(coordinates=coordinates[0], **additional_vertex_parameters)
                end_vertex = self.graph.add_vertex(coordinates=coordinates[1], **additional_vertex_parameters)
            else:
                raise ValueError(f"'Coordinates' have shape {coordinates.shape} but expected shape (2, 3) when not specifying 'vertex_IDs'")

        # Determines the beam properties for each beam element.
        beam_properties = list(np.atleast_1d(E_modulus, shear_modulus, primary_moment_of_area, secondary_moment_of_area, 
                                             torsional_constant, density, cross_sectional_area))
        for i, beam_property in enumerate(beam_properties):
            if len(beam_property) == 1:
                beam_properties[i] = np.full(number_of_elements, beam_property[0])
            if len(beam_property) == 2:
                beam_properties[i] = np.linspace(beam_property[0], beam_property[1], number_of_elements)
            elif len(beam_property) > 2 and len(beam_property) != number_of_elements:
                raise ValueError(f"One of the material property vectors have {len(beam_property)} elements but expected either 1, 2 or {number_of_elements} elements.")
        E_modulus, shear_modulus, primary_moment_of_area, secondary_moment_of_area, torsional_constant, density, cross_sectional_area = beam_properties

        # Determines the coordinates for the element vectors.
        edge_vector = end_vertex['coordinates'] - start_vertex['coordinates']
        edge_vertices_coordinates = np.array([start_vertex['coordinates'] + edge_vector * i for i in np.linspace(0, 1, number_of_elements+1)])

        # Calculates the total DOF of the edge.
        edge_DOF = 6*(number_of_elements + 1)
        # Initializes the mass and stiffness matrices for the entire edge.
        edge_mass_matrix = np.zeros((edge_DOF, edge_DOF))
        edge_stiffness_matrix = edge_mass_matrix.copy()
        # Loops over all beam elements.
        for i in range(number_of_elements):
            # Short handing the beam properties.
            E, G, I_z, I_y, J, RHO, A = E_modulus[i], shear_modulus[i], primary_moment_of_area[i], secondary_moment_of_area[i], torsional_constant[i], density[i], cross_sectional_area[i]
            I0 = I_y + I_z
            L = np.linalg.norm(edge_vector) / number_of_elements
            # Determines the mass matrix per beam element.
            element_mass_matrix = np.array([[140,     0,     0,         0,       0,       0,  70,     0,     0,        0,       0,       0],
                                            [  0,   156,     0,         0,       0,    22*L,   0,    54,     0,        0,       0,   -13*L],
                                            [  0,     0,   156,         0,   -22*L,       0,   0,     0,    54,        0,    13*L,       0],
                                            [  0,     0,     0,  140*I0/A,       0,       0,   0,     0,     0,  70*I0/A,       0,       0],
                                            [  0,     0, -22*L,         0,  4*L**2,       0,   0,     0, -13*L,        0, -3*L**2,       0],
                                            [  0,  22*L,     0,         0,       0,  4*L**2,   0,  13*L,     0,        0,       0, -3*L**2],
                                            [ 70,     0,     0,         0,       0,       0, 140,     0,     0,        0,       0,       0],
                                            [  0,    54,     0,         0,       0,    13*L,   0,   156,     0,        0,       0,   -22*L],
                                            [  0,     0,    54,         0,   -13*L,       0,   0,     0,   156,        0,    22*L,       0],
                                            [  0,     0,     0,   70*I0/A,       0,       0,   0,     0,     0, 140*I0/A,       0,       0],
                                            [  0,     0,  13*L,         0, -3*L**2,       0,   0,     0,  22*L,        0,  4*L**2,       0],
                                            [  0, -13*L,     0,         0,       0, -3*L**2,   0, -22*L,     0,        0,       0,  4*L**2]])
            element_mass_matrix *= RHO*A*L/420

            # Determines the stiffness matrix per beam element.
            element_stiffness_matrix = np.array([[ E*A/L, 0, 0, 0, 0, 0,              
                                                  -E*A/L, 0, 0, 0, 0, 0],
                                                 [0,  12*E*I_z/L**3, 0, 0, 0, 6*E*I_z/L**2,              
                                                  0, -12*E*I_z/L**3, 0, 0, 0, 6*E*I_z/L**2],
                                                 [0, 0,  12*E*I_y/L**3, 0, -6*E*I_y/L**2, 0,              
                                                  0, 0, -12*E*I_y/L**3, 0, -6*E*I_y/L**2, 0],
                                                 [0, 0, 0,  G*J/L, 0, 0,              
                                                  0, 0, 0, -G*J/L, 0, 0],
                                                 [0, 0, -6*E*I_y/L**2, 0, 4*E*I_y/L, 0,              
                                                  0, 0,  6*E*I_y/L**2, 0, 2*E*I_y/L, 0],
                                                 [0,  6*E*I_z/L**2, 0, 0, 0, 4*E*I_z/L,              
                                                  0, -6*E*I_z/L**2, 0, 0, 0, 2*E*I_z/L],
                                                 [-E*A/L, 0, 0, 0, 0, 0,          
                                                   E*A/L, 0, 0, 0, 0, 0],
                                                 [0, -12*E*I_z/L**3, 0, 0, 0, -6*E*I_z/L**2,              
                                                  0,  12*E*I_z/L**3, 0, 0, 0, -6*E*I_z/L**2],
                                                 [0, 0, -12*E*I_y/L**3, 0, 6*E*I_y/L**2, 0,              
                                                  0, 0,  12*E*I_y/L**3, 0, 6*E*I_y/L**2, 0],
                                                 [0, 0, 0, -G*J/L, 0, 0,              
                                                  0, 0, 0,  G*J/L, 0, 0],
                                                 [0, 0, -6*E*I_y/L**2, 0, 2*E*I_y/L, 0,              
                                                  0, 0,  6*E*I_y/L**2, 0, 4*E*I_y/L, 0],
                                                 [0,  6*E*I_z/L**2, 0, 0, 0, 2*E*I_z/L,              
                                                  0, -6*E*I_z/L**2, 0, 0, 0, 4*E*I_z/L]])
            
            # Determines the combined stiffness and mass matrix for the entire edge.
            element_pickoff_operator = np.zeros((12, edge_DOF), dtype=np.int8)
            element_pickoff_operator[:, i*6:i*6+12] = np.eye(12, dtype=np.int8)
            edge_mass_matrix += element_pickoff_operator.T @ element_mass_matrix @ element_pickoff_operator
            edge_stiffness_matrix += element_pickoff_operator.T @ element_stiffness_matrix @ element_pickoff_operator
        
        # Rotates the mass and stiffness matrices to the global context.
        edge_primary_angle_cos, edge_primary_angle_sin = edge_vector[:2] / np.linalg.norm(edge_vector[:2]) if not np.array_equal(edge_vector[:2], (0, 0)) else (1.0, 0.0)
        edge_secondary_angle_cos, edge_secondary_angle_sin = (np.linalg.norm(edge_vector[:2]), -edge_vector[2]) / np.linalg.norm(edge_vector)
        primary_rotation_matrix = np.array([[edge_primary_angle_cos, -edge_primary_angle_sin, 0],
                                            [edge_primary_angle_sin,  edge_primary_angle_cos, 0],
                                            [                     0,                       0, 1]])
        secondary_rotation_matrix = np.array([[ edge_secondary_angle_cos, 0, edge_secondary_angle_sin],
                                              [                        0, 1,                        0],
                                              [-edge_secondary_angle_sin, 0, edge_secondary_angle_cos]])
        polar_rotation_matrix = np.array([[1,                           0,                            0],
                                          [0, np.cos(edge_polar_rotation), -np.sin(edge_polar_rotation)],
                                          [0, np.sin(edge_polar_rotation),  np.cos(edge_polar_rotation)]])
        rotational_matrix = secondary_rotation_matrix @ primary_rotation_matrix @ polar_rotation_matrix
        transformation_matrix = block_diag(*(rotational_matrix.T,)*(2*(number_of_elements+1)))
        edge_mass_matrix = transformation_matrix.T @ edge_mass_matrix @ transformation_matrix
        edge_stiffness_matrix = transformation_matrix.T @ edge_stiffness_matrix @ transformation_matrix
        
        def shape_function(x: float) -> npt.NDArray:
            shape = np.array([[1-x,               0,               0, 0,                    0,                   0, x,             0,             0, 0,             0,              0],
                              [  0, 1-3*x**2+2*x**3,               0, 0,                    0, x*L-2*L*x**2+L*x**3, 0, 3*x**2-2*x**3,             0, 0,             0, -L*x**2+L*x**3],
                              [  0,               0, 1-3*x**2+2*x**3, 0, -x*L+2*L*x**2-L*x**3,                   0, 0,             0, 3*x**2-2*x**3, 0, L*x**2-L*x**3,              0]])
            return rotational_matrix @ shape @ block_diag(*(rotational_matrix.T,)*4)
        
        edge_parameters = {
            'edge_mass_matrix': edge_mass_matrix,
            'edge_stiffness_matrix': edge_stiffness_matrix,
            'number_of_elements': number_of_elements,
            'shape_function': shape_function,
            'edge_vertices_coordinates': edge_vertices_coordinates
        }

        # Adds the beam into the graph.
        self.graph.add_edge(start_vertex, end_vertex, **edge_parameters)

    @property
    def fixed_DOFs(self) -> npt.NDArray:
        """
        List of the DOF's that have a boundary condition applied to it.
        """
        if not np.any(self.graph.vs['fixed']):
            raise ValueError("No vertices have been fixed yet.")
        return np.ravel([6*fixed_vertex_ID + np.arange(6) for fixed_vertex_ID in np.flatnonzero(self.graph.vs['fixed'])])

    @property
    def system_DOF(self) -> int:
        """
        The total DOF of the system.
        """
        return 6*(np.sum(self.graph.es['number_of_elements'], dtype=int) + self.graph.vcount() - self.graph.ecount())

    def get_system_level_matrices(self, include_fixed_vertices: bool = False) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Calculates the system-level mass-, stiffness and damping matrices by combining all edge matrices.
        
        Parameters
        ----------
        include_fixed_vertices : bool, optional
            Whether or not to include the fixed vertices in the matrices. False by default.

        Returns
        -------
        tuple of two numpy arrays
            The first element  is the system-level mass matrix, the second is the stiffness matrix and the third is the damping
            matrix. The order of the rows in each matrix is first vertices followed by the nodes. The vertices are by them selves 
            orded by their respective ID. The nodes are firstly ordered by their respective edge ID and secondly by the direction 
            of the edge.
        """
        # Calculates the number of DOF in the entire system.
        system_DOF = self.system_DOF
        # Initializes the system level mass and stiffness matrices.
        system_mass_matrix = np.zeros((system_DOF, system_DOF))
        system_stiffness_matrix = system_mass_matrix.copy()
        system_damping_matrix = system_mass_matrix.copy()
        # Initializes the column index for the first edge in the edge pickoff operator.
        accumulative_edge_DOF = 6*self.graph.vcount()
        # Loops over all edges to add each edge contribution to the system level matrices.
        for edge in self.graph.es:
            # Calculates the number of DOF in each edge excluding the DOF's in its target and source vertices.
            edge_DOF = 6*(edge['number_of_elements']-1)
            # Initializes the edge pickoff operator where the first set of columns will represent the DOF's in the vertices
            # and the second set of columns will represent the DOF's in the edges alone.
            edge_pickoff_operator = np.zeros((edge_DOF + 12, system_DOF), dtype=np.int8)
            # Picking the correct indices for the edge DOF's.
            edge_pickoff_operator[6:-6, accumulative_edge_DOF:accumulative_edge_DOF+edge_DOF] = np.eye(edge_DOF, dtype=np.int8)
            # Picking the correct indices for the vertices DOF's.
            edge_pickoff_operator[ :6, 6*edge.source:6*edge.source + 6] = np.eye(6, dtype=np.int8)
            edge_pickoff_operator[-6:, 6*edge.target:6*edge.target + 6] = np.eye(6, dtype=np.int8)
            # Adding the edge contributions to the system level matrices.
            system_mass_matrix += edge_pickoff_operator.T @ edge['edge_mass_matrix'] @ edge_pickoff_operator
            system_stiffness_matrix += edge_pickoff_operator.T @ edge['edge_stiffness_matrix'] @ edge_pickoff_operator
            
            accumulative_edge_DOF += edge_DOF

        # Applies boundary conditions if present.
        if not include_fixed_vertices:
            fixed_DOFs = self.fixed_DOFs
            system_mass_matrix = np.delete(system_mass_matrix, fixed_DOFs, axis=0)
            system_mass_matrix = np.delete(system_mass_matrix, fixed_DOFs, axis=1)
            system_stiffness_matrix = np.delete(system_stiffness_matrix, fixed_DOFs, axis=0)
            system_stiffness_matrix = np.delete(system_stiffness_matrix, fixed_DOFs, axis=1)

        # Damping ready stuff here.
        eigvals, eigvecs = eig(system_stiffness_matrix, system_mass_matrix)
        # Modal mass matrix.
        modal_mass_matrix = eigvecs.T @ system_mass_matrix @ eigvecs
        #norm_eigvecs = eigvecs @ np.diag(1/np.sqrt(np.diag(Mt)))
        damping_ratio = 0.05
        # Modal damping matrix.
        modal_damping_matrix = np.eye(len(eigvals)) * (2 * damping_ratio * np.sqrt(np.abs(eigvals)) * modal_mass_matrix)
        # Damping matrix 
        system_damping_matrix = eigvecs @ modal_damping_matrix @ eigvecs.T   
        
        return system_mass_matrix, system_stiffness_matrix, system_damping_matrix
    
    def get_eigen_freq(self):
        """
        Calculates the eigen frequencies of the system.

        Returns
        -------
        numpy array
            Sorted eigen frequencies of the system with shape (n,) where n is the number of eigen frequencies.
            The eigen frequencies are sorted in ascending order, and converted to Hz.
        """ 
        # Gets the system level matrices.
        mass_matrix, stiffness_matrix, _= self.get_system_level_matrices()
        # Calculates the eigenvalues and eigenvectors.
        eigvals, _ = eig(stiffness_matrix, mass_matrix)
        # Calculates the eigen frequencies.
        eigen_freq = np.sqrt(np.abs(eigvals)) / (2*np.pi)
        # Sorts the eigen frequencies and eigenvectors.
        sort_indices = np.argsort(eigen_freq)
        eigen_freq = eigen_freq[sort_indices]

        return eigen_freq

    def fix_vertices(self, fixed_vertex_IDs: Collection[int]) -> None:
        """
        Applies a fixed boundary condition to a given collection of vertices.

        Parameters
        ----------
        fixed_vertex_IDs : Collection of int
            The ID of the vertices to be fixed.
        """
        for fixed_vertex_ID in fixed_vertex_IDs:
            fixed_vertex = self.graph.vs[fixed_vertex_ID]
            if fixed_vertex['force'] is None:
                fixed_vertex['fixed'] = True
            else:
                raise ValueError(f"Vertex ID '{fixed_vertex_ID}' have a force applied to it and can therefore not be fixed.")

    def free_vertices(self, vertex_IDs: Collection[int] | None = None) -> None:
        """
        Frees fixed vertices.

        Parameters
        ----------
        vertex_IDs : Collection of int, optional
            The vertex IDs that should be freed. If it is already free it will be ignored.
            If not specified all vertices are freed.
        """
        if vertex_IDs is None:
                # If no vertices are specifed, frees all vertices.
                self.graph.vs['fixed'] = False
        else:
            for vertex_ID in vertex_IDs:
                # Frees the given vertices (if given).
                self.graph.vs[vertex_ID]['fixed'] = False

    @contextmanager
    def fixed_vertices(self, fixed_vertex_IDs: Collection[int]) -> Generator[None, None, None]:
        """
        Context manager for fixed vertices which are freed afterwards.

        Parameters
        ----------
        fixed_vertex_IDs : Collection of int
            The ID of the vertices to be temporarely fixed.
        """
        self.fix_vertices(fixed_vertex_IDs)
        yield
        self.free_vertices(fixed_vertex_IDs)

    def add_forces(self, forces: dict[int, Callable[[float], Collection[int]]]) -> None:
        """
        Adds force(s) to the system.

        Parameters
        ----------
        forces : dict[int, Callable[[float], Collection[int]]]
            The force(s) applied to the vertices with vertex ID as the key and a callable as the value with time as input 
            and the force as output with shape (6,).
        """
        for vertex_ID, force in forces.items():
            vertex_with_applied_force = self.graph.vs[vertex_ID]
            if vertex_with_applied_force['fixed']:
                raise ValueError(f"Vertex ID '{vertex_ID}' is fixed and can therefore not have a force applied to it.")
            else:
                vertex_with_applied_force['force'] = force
    
    def remove_forces(self, vertex_IDs: Collection[int] | None = None) -> None:
            """
            Removes forces from vertices.

            Parameters
            ----------
            vertex_IDs : Collection of int, optional
                The list of vertex IDs to remove forces from. Vertices which already have no forces are ignored.
                If not specified, all forces are removed.
            """
            if vertex_IDs is None:
                # If no vertices are specifed, remove all forces.
                self.graph.vs['force'] = None
            else:
                for vertex_ID in vertex_IDs:
                    # Removes the force from the given vertices (if given).
                    self.graph.vs[vertex_ID]['force'] = None

    @contextmanager
    def added_forces(self, forces: dict[int, Callable[[float], Collection[int]]]) -> Generator[None, None, None]:
        """
        Context manager for adding forces which are removed afterwards.

        Parameters
        ----------
        forces : dict[int, Callable[[float], Collection[int]]]
            The force(s) temporarely applied to the vertices with vertex ID as the key and a callable as the value with time as input 
            and the force as output with shape (6,).
        """
        self.add_forces(forces)
        yield
        self.remove_forces(forces.keys())

    def get_force_vector(self, include_fixed_vertices: bool = False, time: float = 0.0) -> npt.NDArray:
        """
        Calculates the force vector based on the given forces applied.

        Parameters
        ----------
        include_fixed_vertices : bool, optional
            Whether or not to include the fixed vertices in the force output vector. False by default.
        time : float, optional
            The time parameter for the force functions. 0.0 by default.
        """
        # Constructs the force vector.
        force_vector = np.zeros(self.system_DOF)
        for vertex in self.graph.vs:
            if vertex['force'] is not None:
                force_vector[6*vertex.index:6*vertex.index + 6] = vertex['force'](time)
        if include_fixed_vertices:
            return force_vector
        else:
            return np.delete(force_vector, self.fixed_DOFs)

    @deprecated("Use Static from SystemSolver insted.")
    def get_static_vertex_and_node_displacements(self, include_fixed_vertices: bool = False) -> npt.NDArray:
        """
        Gets the displacement for all vertices and nodes under a static load given by the first time step of the force functions. 
        The displaced position is not calculated.

        Parameters
        ----------
        include_fixed_vertices : bool, optional
            Whether or not to include the fixed vertices in the displacement output vector. False by default.

        Returns
        -------
        numpy array
            The displacement of each vertex and node with shape (6*n,) where n is the number of total vertices and nodes.
            The array is ordered as (vertex displacements, nodal displacements) where the vertex displacements are order
            according to their ID and the nodal displacements are ordered firstly by their corresponding edge ID and secondly
            according to the direction of the edge. Each vertex/nodal displacement is then orded by (x, y, z, phi_x, phi_y, phi_z) 
            displacement.
        """
        # Gets the system level stiffness matrix.
        _, stiffness_matrix, _ = self.get_system_level_matrices()

        # Constructs the force vector.
        force_vector = self.get_force_vector()

        # Calculates the displacements.
        displacements = np.linalg.inv(stiffness_matrix) @ force_vector

        # Inserts the fixed DOF back again.
        if include_fixed_vertices:
            for fixed_DOF in self.fixed_DOFs:
                displacements = np.insert(displacements, fixed_DOF, 0.0)
        
        return displacements

    @deprecated("Use SystemSolver insted.")
    def get_displaced_vertices_and_node_position(self) -> list[npt.NDArray]:
        """
        Calculates the displaced position of each vertex and node in the system under a given static load.

        Returns
        -------
        list of numpy array
            A list of 2D arrays with the displaced positions of all vertices and nodes. Each element in the list corrsponds to an edge and each
            array has the shape (number_of_elements + 2, 6) where number_of_elements refer to the given edge and plus two to include the source
            and target vertices of the edge.
        """
        # Gets the displacement for all vertices and nodes.
        vertex_and_node_displacements = self.get_static_vertex_and_node_displacements()
        vertex_displacements, node_displacements = np.split(vertex_and_node_displacements, [6*self.graph.vcount()])
        # Initializes array for the displaced position of all vertices and nodes along any edge in the order of the edge direction. 
        # Shape: (number of edges, number of vertex and nodes for the given edge).
        vertex_and_node_displaced_positions = list()
        
        accumulative_edge_DOF = 0
        # Loops over all edges.
        for edge in self.graph.es:
            # Calculates the number of DOF in the given edge excluding the DOF's in its target and source vertices.
            edge_DOF = 6*(edge['number_of_elements']-1)
            # Calculates the displaced position for the source and target vertex of the edge.
            source_vertex_displaced_position = vertex_displacements[6*edge.source : 6*edge.source + 6] + np.concatenate([self.graph.vs[edge.source]['coordinates'], [0]])
            target_vertex_displaced_position = vertex_displacements[6*edge.target : 6*edge.target + 6] + np.concatenate([self.graph.vs[edge.target]['coordinates'], [0]])
            # Gets all node displacements for the current edge.
            edge_node_displacements = node_displacements[accumulative_edge_DOF : accumulative_edge_DOF + edge_DOF]
            accumulative_edge_DOF += edge_DOF
            # Calculates all node displaced positions for the current edge.
            edge_node_displaced_positions = edge_node_displacements.reshape((-1, 6)) + np.column_stack((edge['edge_vertices_coordinates'][1:-1], np.zeros(edge['number_of_elements']-1)))
            # Combines the vertices and nodal displacements in the order of the edge direction.
            vertex_and_node_displaced_positions.append(np.vstack((source_vertex_displaced_position, edge_node_displaced_positions, target_vertex_displaced_position)))
            
        return vertex_and_node_displaced_positions

    @deprecated("Use SystemSolver insted.")
    def get_displaced_shape_position(self, resolution_per_element: int = 100) -> list[npt.NDArray]:
        """
        Calculates the shape of each beam element.

        Parameters
        ----------
        resolution_per_element : int, optional
            The number of points per element per edge (Default 100).

        Returns
        -------
        list of numpy array
            A list of 2D arrays with the coordinates of all shaped beam elements. Each element in the list corresponds to an edge and each
            array has the shape (resolution_per_element*number_of_elements, 3) where the number_of_elements refer to the corresponding
            edge. The coordinates are orded along the edge direction.
        """
        # Gets the displacement for all vertices and nodes.
        vertex_and_node_displacements = self.get_static_vertex_and_node_displacements(include_fixed_vertices=True)
        vertex_displacements, node_displacements = np.split(vertex_and_node_displacements, [6*self.graph.vcount()])
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
            edge_DOF = 6*(edge['number_of_elements']-1)
            # Gets the displacements for the source and target vertex of the edge.
            source_vertex_displacement = vertex_displacements[6*edge.source : 6*edge.source + 6]
            target_vertex_displacement = vertex_displacements[6*edge.target : 6*edge.target + 6]
            # Gets the node displacements of the current edge.
            edge_node_displacements = node_displacements[accumulative_edge_DOF : accumulative_edge_DOF + edge_DOF].reshape(-1, 6)
            accumulative_edge_DOF += edge_DOF
            # Combines the vertices and nodal displacements in the order of the edge direction.
            edge_vertex_and_node_displacements.append(np.vstack((source_vertex_displacement, edge_node_displacements, target_vertex_displacement)))
            # Initializes the array that contains all the displaced positions of the current edge.
            edge_point_displaced_positions.append(np.empty((edge['number_of_elements'] * resolution_per_element, 3)))
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

# Example.
if __name__ == "__main__":
    # Beam lattice example.
    import matplotlib.pyplot as plt
    
    ax = plt.subplot(projection='3d')

    system = Beam_Lattice()
    
    system.add_beam_edge(
        number_of_elements=1, 
        E_modulus=2.1*10**11, 
        shear_modulus=7.9*10**10,
        primary_moment_of_area=2.157*10**-8,
        secondary_moment_of_area=1.113*10**-8,
        torsional_constant=3.7*10**-8,
        density=7850, 
        cross_sectional_area=1.737*10**-4, 
        coordinates=[[0, 0, 0], [0, 0, 1.7]],
        edge_polar_rotation=0
    )



    system.add_forces({1: lambda t: [0, 0, 0, 0, 0, 0]})
    system.fix_vertices((0,))
    displaced_shape_points = system.get_displaced_shape_position()
    output = system.get_eigen_freq()
    
    for displaced_shape_point in displaced_shape_points:
        ax.plot(displaced_shape_point[:, 0], displaced_shape_point[:, 1], displaced_shape_point[:, 2], linewidth=2.0, linestyle='--')
    MM, KK,_ = system.get_system_level_matrices()
    eigvals, eigvecs = eig(KK, MM)

    print(MM,KK)
    print(output)

    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid()
    plt.show()