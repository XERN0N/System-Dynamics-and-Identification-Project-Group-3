import numpy as np
import numpy.typing as npt
from SystemModels import Beam_Lattice
from abc import ABC, abstractmethod
from typing import Self

class Solver(ABC):
    def __init__(self, beam_graph: Beam_Lattice) -> None:
        self.beam_graph = beam_graph
        self.solution: dict[str, npt.NDArray | None] = {'displacements': None,
                                                        'velocites': None,
                                                        'accelerations': None,
                                                        'time': None}

    @abstractmethod
    def solve(self, include_fixed_vertices: bool = True) -> Self:
        pass

    def _insert_fixed_DOFs(self) -> None:
        # Inserts the fixed DOF back again.
        for fixed_DOF in self.beam_graph.fixed_DOFs:
                for kinematic in ('displacements', 'velocites', 'accelerations'):
                    self.solution[kinematic] = np.insert(self.solution[kinematic], fixed_DOF, 0.0, axis=1)

    def get_displaced_vertices_and_node_position(self) -> list[npt.NDArray]:
        """
        Calculates the displaced position of each vertex and node in the system under a given static load.

        Returns
        -------
        list of numpy array
            A list of 3D arrays with the displaced positions of all vertices and nodes. Each element in the list corrsponds to an edge and each
            array has the shape (timesteps, number_of_elements + 2, 6) where number_of_elements refer to the given edge and plus two to include the source
            and target vertices of the edge.
        """
        # Initializes array for the displaced position of all vertices and nodes along any edge in the order of the edge direction.
        graph_displaced_position = list()
        # Loops over all timesteps.
        for t, displacement in enumerate(self.solution['displacements']):
            vertex_displacements, node_displacements = np.split(displacement, [6*self.beam_graph.graph.vcount()])
            accumulative_edge_DOF = 0
            # Loops over all edges.
            for i, edge in enumerate(self.beam_graph.graph.es):
                # Calculates the number of DOF in the given edge excluding the DOF's in its target and source vertices.
                edge_DOF = 6*(edge['number_of_elements']-1)
                # If first time step, initialize the solution array.
                if t == 0: graph_displaced_position.append(np.empty((len(self.solution['time']), edge['number_of_elements'] + 2, 6)))
                # Calculates the displaced position for the source and target vertex of the edge.
                source_vertex_displaced_position = vertex_displacements[6*edge.source : 6*edge.source + 6] + np.concatenate([self.beam_graph.graph.vs[edge.source]['coordinates'], [0]])
                target_vertex_displaced_position = vertex_displacements[6*edge.target : 6*edge.target + 6] + np.concatenate([self.beam_graph.graph.vs[edge.target]['coordinates'], [0]])
                # Gets all node displacements for the current edge.
                edge_node_displacements = node_displacements[accumulative_edge_DOF : accumulative_edge_DOF + edge_DOF]
                accumulative_edge_DOF += edge_DOF
                # Calculates all node displaced positions for the current edge.
                edge_node_displaced_positions = edge_node_displacements.reshape((-1, 6)) + np.column_stack((edge['edge_vertices_coordinates'][1:-1], np.zeros(edge['number_of_elements']-1)))
                # Combines the vertices and nodal displacements in the order of the edge direction.
                graph_displaced_position[i][t] = np.vstack((source_vertex_displaced_position, edge_node_displaced_positions, target_vertex_displaced_position))
            
        return graph_displaced_position

    def get_displaced_shape_position(self, scaling_factor: float = 1.0, resolution_per_element: int = 10) -> list[npt.NDArray]:
        """
        Calculates the shape of each beam element.

        Parameters
        ----------
        scaling_factor : float, optional
            The scaling factor for the displacements (Default 1.0).
        resolution_per_element : int, optional
            The number of points per element per edge (Default 10).

        Returns
        -------
        list of numpy array
            A list of 3D arrays with the coordinates of all shaped beam elements. Each element in the list corresponds to an edge and each
            array has the shape (timesteps, resolution_per_element*number_of_elements, 3) where the number_of_elements refer to the corresponding
            edge. The coordinates are orded along the edge direction.
        """
        # Initializes return array for the displaced positions of all the points along any edge.
        # Shape: (number of edges, number of elements * resolution per element for the given edge).
        edge_point_displaced_positions: list[npt.NDArray] = list()
        
        for t, displacement in enumerate(self.solution['displacements']):    
            vertex_displacements, node_displacements = np.split(scaling_factor * displacement, [6*self.beam_graph.graph.vcount()])
            # Calculates the normalized distances along any element where the position of the displaced point will be evaluated.
            normalized_distances_along_element = np.linspace(0, 1, resolution_per_element)
            accumulative_edge_DOF = 0
            # Looping over all edges.
            for i, edge in enumerate(self.beam_graph.graph.es):
                # Calculates the number of DOF in the given edge excluding the DOF's in its target and source vertices.
                edge_DOF = 6*(edge['number_of_elements']-1)
                # Gets the displacements for the source and target vertex of the edge.
                source_vertex_displacement = vertex_displacements[6*edge.source : 6*edge.source + 6]
                target_vertex_displacement = vertex_displacements[6*edge.target : 6*edge.target + 6]
                # Gets the node displacements of the current edge.
                edge_node_displacements = node_displacements[accumulative_edge_DOF : accumulative_edge_DOF + edge_DOF].reshape(-1, 6)
                accumulative_edge_DOF += edge_DOF
                # Combines the vertices and nodal displacements in the order of the edge direction.
                edge_vertex_and_node_displacements = np.vstack((source_vertex_displacement, edge_node_displacements, target_vertex_displacement))
                # Initializes the array that contains all the displaced positions of the current edge.
                if t == 0: edge_point_displaced_positions.append(np.empty((len(self.solution['time']), edge['number_of_elements'] * resolution_per_element, 3)))
                # Looping over all elements in each edge.
                for j in range(edge['number_of_elements']):
                    # Gets the source and target node displacement of the current element.
                    element_node_displacement = np.ravel(edge_vertex_and_node_displacements[j:j+2])
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
                        edge_point_displaced_positions[i][t, j*resolution_per_element + k] = point_displacement + point_position

        return edge_point_displaced_positions

class Static(Solver):
        
    def solve(self, include_fixed_vertices: bool = True) -> Self:
        """
        Gets the displacement for all vertices and nodes under a static load given by the first time step of the force functions. 
        The displaced position is not calculated.
        
        Parameters
        ----------
        include_fixed_vertices : bool, optional
            Whether or not to include the fixed vertices in the displacement output vector. True by default.
        """
        # Gets the system level stiffness matrix.
        _, stiffness_matrix, _ = self.beam_graph.get_system_level_matrices()

        # Constructs the force vector.
        force_vector = self.beam_graph.get_force_vector()
        
        # Calculates the displacements.
        self.solution['displacements'] = (np.linalg.inv(stiffness_matrix) @ force_vector).reshape((1, -1))

        # Sets the velocites and accelerations.
        self.solution['velocites'] = np.zeros(force_vector.shape).reshape((1, -1))
        self.solution['accelerations'] = self.solution['velocites'].copy()
        self.solution['time'] = np.zeros((1, 1))

        # Inserts the fixed DOF back again.
        if include_fixed_vertices:
            self._insert_fixed_DOFs()

        return self
    
class Newmark(Solver):
    
    def __init__(self, beam_graph: Beam_Lattice, inital_condition_solver: Solver, end_time: int, time_increment: float, integration_parameters: tuple[float, float] = (1/4, 1/2)) -> None:
        """
        Parameters
        ----------
        end_time : int
            The end time step [s].
        initial_condition_solver : Solver
            The solver which last time step should be the inital time step of this solver.
        time_increment : float
            The length of each time step [s].
        integration_parameters : tuple of float
            The beta and gamma integration parameters for the Newmark time integration
            method.
        """
        super().__init__(beam_graph)
        self.end_time = end_time
        self.time_increment = time_increment
        self.integration_parameters = integration_parameters
        self.inital_condition_solver = inital_condition_solver

    def solve(self, include_fixed_vertices: bool = True) -> Self:
        """
        Performs newmark time integration on a given system with n DOF.        
        
        Parameters
        ----------
        include_fixed_vertices : bool, optional
            Whether or not to include the fixed vertices in the displacement output vector. True by default.    
        """
        beta, gamma = self.integration_parameters
        stiffness, mass, damping = self.beam_graph.get_system_level_matrices()
        
        effective_stiffness = 1/(beta*self.time_increment**2) * mass + gamma/(beta*self.time_increment) * damping + stiffness

        def effective_force(displacement: npt.NDArray, velocity: npt.NDArray, acceleration: npt.NDArray, time: float) -> npt.NDArray:
            effective_mass_displacement = 1/(beta*self.time_increment**2) * displacement
            effective_mass_velocity = 1/(beta*self.time_increment) * velocity
            effective_mass_acceleration = (1/(2*beta)-1) * acceleration
            force = self.beam_graph.get_force_vector(time=time)
            effective_damping_displacement = gamma/(beta*self.time_increment) * displacement
            effective_damping_velocity = (gamma/beta-1) * velocity
            effective_damping_acceleration = self.time_increment * (gamma/(2*beta)-1) * acceleration
            return force + mass @ (effective_mass_displacement + effective_mass_velocity + effective_mass_acceleration) + \
                        damping @ (effective_damping_displacement + effective_damping_velocity + effective_damping_acceleration)
        
        time_steps = np.arange(self.time_increment, self.end_time, self.time_increment)

        initial_solution = self.inital_condition_solver.solve(False).solution

        non_fixed_DOFs = self.beam_graph.system_DOF - len(self.beam_graph.fixed_DOFs)

        self.solution = {'displacements': np.empty((len(time_steps)+1, non_fixed_DOFs)),
                         'velocites': np.empty((len(time_steps)+1, non_fixed_DOFs)),
                         'accelerations': np.empty((len(time_steps)+1, non_fixed_DOFs)),
                         'time': np.insert(time_steps, 0, 0)}
        
        self.solution['displacements'][0] = initial_solution['displacements'][-1]
        self.solution['velocites'][0] = initial_solution['velocites'][-1]
        self.solution['accelerations'][0] = np.linalg.solve(mass, - damping @ self.solution['velocites'][0] - stiffness @ self.solution['displacements'][0])        
        
        for i, time in enumerate(time_steps):
            self.solution['displacements'][i+1] = np.linalg.solve(effective_stiffness, effective_force(self.solution['displacements'][i], 
                                                                                                       self.solution['velocites'][i], 
                                                                                                       self.solution['accelerations'][i], 
                                                                                                       time))
            self.solution['velocites'][i+1] = gamma/(beta*self.time_increment) * (self.solution['displacements'][i+1] - self.solution['displacements'][i]) - (gamma/beta-1) * self.solution['velocites'][i] - self.time_increment * (gamma/(2*beta) - 1) * self.solution['accelerations'][i]
            self.solution['accelerations'][i+1] = 1/(beta*self.time_increment**2) * (self.solution['displacements'][i+1] - self.solution['displacements'][i] - self.time_increment * self.solution['velocites'][i]) - (1/(2*beta)-1) * self.solution['accelerations'][i]
        
        # Inserts the fixed DOF back again.
        if include_fixed_vertices:
            self._insert_fixed_DOFs()

        return self

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    system = Beam_Lattice()

    system.add_beam_edge(
        number_of_elements=1,
        E_modulus=2.1e11,
        shear_modulus=7.9e10,
        primary_moment_of_area=2.157e-8,
        secondary_moment_of_area=2.157e-8,
        polar_mass_moment_of_inertia=3.7e-8,
        density=7850,
        cross_sectional_area=1.737e-4,
        coordinates=((0, 0, 0), (1, 0, 0)),
        edge_polar_rotation=0
    )

    system.add_beam_edge(
        number_of_elements=1,
        E_modulus=2.1e11,
        shear_modulus=7.9e10,
        primary_moment_of_area=2.157e-8,
        secondary_moment_of_area=2.157e-8,
        polar_mass_moment_of_inertia=3.7e-8,
        density=7850,
        cross_sectional_area=1.737e-4,
        coordinates=(1, 1, 0),
        vertex_IDs=1,
        edge_polar_rotation=0
    )

    def delta_force(time: float) -> npt.ArrayLike:
        if time > 0.0:
            return [0, 0, 0, 0, 0, 0]
        else:
            return [0, 0, 1e3, 0, 0, 0]
        
    system.add_forces({1: delta_force})
    system.fix_vertices((0,))

    end_time = 3000
    time_increment = 10
    scaling_factor = 1
    initial_condition_solver = Static(system)
    displaced_shape_points = Newmark(system, initial_condition_solver, end_time, time_increment).solve().get_displaced_shape_position(scaling_factor)

    plot_lines = list()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for displaced_shape_point in displaced_shape_points:
        plot_lines.append(ax.plot(displaced_shape_point[0, :, 0], displaced_shape_point[0, :, 1], displaced_shape_point[0, :, 2], linewidth=2.0, linestyle='--'))

    def update(frame):
        for i, lines in enumerate(plot_lines):
            for line in lines:
                line.set_data_3d(*displaced_shape_points[i][frame, :].T)
        ax.set_title(f"{frame*time_increment:.3f}")
        return plot_lines

    ani = animation.FuncAnimation(fig=fig, func=update, frames=int(end_time/time_increment), interval=int(time_increment))

    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid()
    plt.show()