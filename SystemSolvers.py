import numpy as np
import numpy.typing as npt
from SystemModels import Beam_Lattice
from abc import ABC, abstractmethod
from typing import Self
from dataclasses import dataclass

@dataclass
class Solution:
    time: npt.NDArray
    displacements: npt.NDArray
    velocites: npt.NDArray
    accelerations: npt.NDArray

class Solver(ABC):
    def __init__(self, beam_graph: Beam_Lattice, initial_condition_solver: Self | None = None) -> None:
        self.beam_graph = beam_graph
        self.initial_condition_solver = initial_condition_solver
        self.solution: Solution | None = None

    def _initialize_solution(self, time_steps: npt.ArrayLike) -> None:
        """
        Initializes the solution data class without including the fixed vertices in the solution and also sets the initial
        displacement and velocity based on the initial condition solver if given.

        Parameters
        ----------
        time_steps : array_like
            The timesteps that will be evaluated by the solver.
        """
        number_of_timesteps = len(time_steps)
        DOF = self.beam_graph.system_DOF - len(self.beam_graph.fixed_DOFs)
        self.solution = Solution(np.empty((number_of_timesteps, DOF)), *np.empty((3, number_of_timesteps, DOF)))
        self.solution.time = np.asarray(time_steps)
        if self.initial_condition_solver is not None:
            if self.initial_condition_solver.solution is None:
                raise AttributeError("Run 'solve()' for the initial condition solver before running 'solve()' for the current solver.")
            self.initial_condition_solver._remove_fixed_DOFs()
            initial_solution = self.initial_condition_solver.solution
            self.solution.displacements[0] = initial_solution.displacements[-1]
            self.solution.velocites[0] = initial_solution.velocites[-1]

    @abstractmethod
    def solve(self, include_fixed_vertices: bool = False) -> Self:
        pass

    @property
    def fixed_DOFs_included(self) -> bool | None:
        """
        Checks whether or not the solution includes the fixed DOF's or not. Returns None if solve() hasn't been run yet.
        """
        if self.solution is None:
            return None
        else:
            return True if self.beam_graph.system_DOF == self.solution.displacements.shape[1] else False
        
    def _insert_fixed_DOFs(self) -> None:
        """
        Inserts the fixed DOF's into the solution dict. Ignored if they are already added.
        """
        if not self.fixed_DOFs_included and self.fixed_DOFs_included is not None:
            for fixed_DOF in self.beam_graph.fixed_DOFs:
                    self.solution.displacements = np.insert(self.solution.displacements, fixed_DOF, 0.0, axis=1)
                    self.solution.velocites = np.insert(self.solution.velocites, fixed_DOF, 0.0, axis=1)
                    self.solution.accelerations = np.insert(self.solution.accelerations, fixed_DOF, 0.0, axis=1)

    def _remove_fixed_DOFs(self) -> None:
        """
        Removes the fixed DOF's from the solution dict. Ignored if they are already removed.
        """
        if self.fixed_DOFs_included and self.fixed_DOFs_included is not None:
            self.solution.displacements = np.delete(self.solution.displacements, self.beam_graph.fixed_DOFs, axis=1)
            self.solution.velocites = np.delete(self.solution.velocites, self.beam_graph.fixed_DOFs, axis=1)
            self.solution.accelerations = np.delete(self.solution.accelerations, self.beam_graph.fixed_DOFs, axis=1)

    def get_displaced_vertices_and_node_position(self, include_fixed_vertices: bool = True) -> list[npt.NDArray]:
        """
        Calculates the displaced position of each vertex and node in the system under a given static load.

        include_fixed_vertices : bool, optional
            Whether or not to include the fixed vertices in the displacement vectors. True by default.

        Returns
        -------
        list of numpy array
            A list of 3D arrays with the displaced positions of all vertices and nodes. Each element in the list corrsponds to an edge and each
            array has the shape (timesteps, number_of_elements + 1, 6) where number_of_elements refer to the given edge and plus one to include the source
            and target vertices of the edge.
        """
        # Ensures that there is a solution.
        if self.solution is None:
            raise ValueError("Run 'solve()' before 'get_displaced_vertices_and_node_position()'.")
        # Initializes array for the displaced position of all vertices and nodes along any edge in the order of the edge direction.
        graph_displaced_position = list()
        # Removes or adds fixed DOF's from the solution depending on the parameter.
        self._insert_fixed_DOFs() if include_fixed_vertices else self._remove_fixed_DOFs()
        # Loops over all timesteps.
        for t, displacement in enumerate(self.solution.displacements):
            vertex_displacements, node_displacements = np.split(displacement, [6*self.beam_graph.graph.vcount()])
            accumulative_edge_DOF = 0
            # Loops over all edges.
            for i, edge in enumerate(self.beam_graph.graph.es):
                # Calculates the number of DOF in the given edge excluding the DOF's in its target and source vertices.
                edge_DOF = 6*(edge['number_of_elements']-1)
                # If first time step, initialize the solution array.
                if t == 0: graph_displaced_position.append(np.empty((len(self.solution.time), edge['number_of_elements'] + 1, 6)))
                # Calculates the displaced position for the source and target vertex of the edge.
                source_vertex_displaced_position = vertex_displacements[6*edge.source : 6*edge.source + 6] + np.concatenate([self.beam_graph.graph.vs[edge.source]['coordinates'], [0, 0, 0]])
                target_vertex_displaced_position = vertex_displacements[6*edge.target : 6*edge.target + 6] + np.concatenate([self.beam_graph.graph.vs[edge.target]['coordinates'], [0, 0, 0]])
                # Gets all node displacements for the current edge.
                edge_node_displacements = node_displacements[accumulative_edge_DOF : accumulative_edge_DOF + edge_DOF]
                accumulative_edge_DOF += edge_DOF
                # Calculates all node displaced positions for the current edge.
                edge_node_displaced_positions = edge_node_displacements.reshape((-1, 6)) + np.column_stack((edge['edge_vertices_coordinates'][1:-1], np.zeros((edge['number_of_elements']-1, 3))))
                # Combines the vertices and nodal displacements in the order of the edge direction.
                graph_displaced_position[i][t] = np.vstack((source_vertex_displaced_position, edge_node_displaced_positions, target_vertex_displaced_position))
            
        return graph_displaced_position

    def get_displaced_shape_position(self, scaling_factor: float = 1.0, resolution_per_element: int = 10, include_fixed_vertices: bool = True) -> list[npt.NDArray]:
        """
        Calculates the shape of each beam element.

        Parameters
        ----------
        scaling_factor : float, optional
            The scaling factor for the displacements (Default 1.0).
        resolution_per_element : int, optional
            The number of points per element per edge (Default 10).
        include_fixed_vertices : bool, optional
            Whether or not to include the fixed vertices in the position vectors. True by default.

        Returns
        -------
        list of numpy array
            A list of 3D arrays with the coordinates of all shaped beam elements. Each element in the list corresponds to an edge and each
            array has the shape (timesteps, resolution_per_element*number_of_elements, 3) where the number_of_elements refer to the corresponding
            edge. The coordinates are orded along the edge direction.
        """
        # Ensures that there is a solution.
        if self.solution is None:
            raise ValueError("Run 'solve()' before 'get_displaced_shape_position()'.")
        # Initializes return array for the displaced positions of all the points along any edge.
        # Shape: (number of edges, number of elements * resolution per element for the given edge).
        edge_point_displaced_positions: list[npt.NDArray] = list()
        # Removes or adds fixed DOF's from the solution depending on the parameter.
        self._insert_fixed_DOFs() if include_fixed_vertices else self._remove_fixed_DOFs()
        for t, displacement in enumerate(self.solution.displacements):    
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
                if t == 0: edge_point_displaced_positions.append(np.empty((len(self.solution.time), edge['number_of_elements'] * resolution_per_element, 3)))
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

    def __init__(self, beam_graph: Beam_Lattice) -> None:
        """
        Static solver to solve the system in the steady state case.

        Parameters
        ----------
        beam_graph : Beam_Lattice
            The system to solve.
        """
        super().__init__(beam_graph)
        
    def solve(self, include_fixed_vertices: bool = False) -> Self:
        """
        Gets the displacement for all vertices and nodes under a static load given by the first time step of the force functions. 
        The displaced position is not calculated.
        
        Parameters
        ----------
        include_fixed_vertices : bool, optional
            Whether or not to include the fixed vertices in the displacement output vector. False by default.
        """
        # Gets the system level stiffness matrix.
        _, stiffness_matrix, _ = self.beam_graph.get_system_level_matrices()
        # Constructs the force vector.
        force_vector = self.beam_graph.get_force_vector()
        # Initializes the solution array.
        self._initialize_solution((0,))
        # Calculates the displacements.
        self.solution.displacements = (np.linalg.inv(stiffness_matrix) @ force_vector).reshape((1, -1))
        # Sets the velocites and accelerations.
        self.solution.velocites = np.zeros(force_vector.shape).reshape((1, -1))
        self.solution.accelerations = self.solution.velocites.copy()
        self.solution.time = np.zeros((1, 1))
        # Inserts the fixed DOF back again.
        if include_fixed_vertices:
            self._insert_fixed_DOFs()

        return self
    
class Newmark(Solver):
    
    def __init__(self, beam_graph: Beam_Lattice, initial_condition_solver: Solver, end_time: int, time_increment: float, integration_parameters: tuple[float, float] = (1/4, 1/2)) -> None:
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
        super().__init__(beam_graph, initial_condition_solver)
        self.end_time = end_time
        self.time_increment = time_increment
        self.beta, self.gamma = integration_parameters

    def solve(self, include_fixed_vertices: bool = False) -> Self:
        """
        Performs newmark time integration on a given system with n DOF.        
        
        Parameters
        ----------
        include_fixed_vertices : bool, optional
            Whether or not to include the fixed vertices in the displacement output vector. False by default.    
        """
        # Sets the timesteps.
        time_steps = np.arange(0, self.end_time, self.time_increment)
        # Initializes the solution vector including the initial displacement- and velocity vectors.
        self._initialize_solution(time_steps)
        # Gets the system level matrices.
        mass, stiffness, damping = self.beam_graph.get_system_level_matrices()
        # Calculates the initial acceleration.
        self.solution.accelerations[0] = np.linalg.solve(mass, self.beam_graph.get_force_vector() - damping @ self.solution.velocites[0] - stiffness @ self.solution.displacements[0])
        # Calculates the effective stiffness.
        effective_stiffness = 1/(self.beta*self.time_increment**2) * mass + self.gamma/(self.beta*self.time_increment) * damping + stiffness
        # Solving...
        for i, time in enumerate(time_steps[1:]):
            # Solves the next displacement.
            self.solution.displacements[i+1] = np.linalg.solve(effective_stiffness, self._effective_force(self.solution.displacements[i], 
                                                                                                          self.solution.velocites[i], 
                                                                                                          self.solution.accelerations[i],
                                                                                                          mass, damping, time))
            # Solves the next velocity.
            self.solution.velocites[i+1] = self.gamma/(self.beta*self.time_increment) * (self.solution.displacements[i+1] - self.solution.displacements[i]) - \
                (self.gamma/self.beta-1) * self.solution.velocites[i] - self.time_increment * (self.gamma/(2*self.beta) - 1) * self.solution.accelerations[i]
            # Solves the next acceleration.
            self.solution.accelerations[i+1] = 1/(self.beta*self.time_increment**2) * (self.solution.displacements[i+1] - \
                self.solution.displacements[i] - self.time_increment * self.solution.velocites[i]) - (1/(2*self.beta)-1) * self.solution.accelerations[i]
        # Inserts the fixed DOF back again.
        if include_fixed_vertices:
            self._insert_fixed_DOFs()

        return self

    def _effective_force(self, displacement: npt.NDArray, velocity: npt.NDArray, acceleration: npt.NDArray, mass: npt.NDArray, damping: npt.NDArray, time: float) -> npt.NDArray:
        """
        Calculates the effective force given the state and system level matrices.
        """
        effective_mass_displacement = 1/(self.beta*self.time_increment**2) * displacement
        effective_mass_velocity = 1/(self.beta*self.time_increment) * velocity
        effective_mass_acceleration = (1/(2*self.beta)-1) * acceleration
        force = self.beam_graph.get_force_vector(time=time)
        effective_damping_displacement = self.gamma/(self.beta*self.time_increment) * displacement
        effective_damping_velocity = (self.gamma/self.beta-1) * velocity
        effective_damping_acceleration = self.time_increment * (self.gamma/(2*self.beta)-1) * acceleration
        return force + mass @ (effective_mass_displacement + effective_mass_velocity + effective_mass_acceleration) + \
                    damping @ (effective_damping_displacement + effective_damping_velocity + effective_damping_acceleration)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    system = Beam_Lattice()

    primary_moment_of_area = 1.94e-6
    secondary_moment_of_area = 1.02e-6
    torsional_constant = 2.29e-6
    cross_sectional_area = 1.74e-4

    system.add_beam_edge(
        number_of_elements=1,
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

    system.add_beam_edge(
        number_of_elements=1,
        E_modulus=2.1e11,
        shear_modulus=7.9e10,
        primary_moment_of_area=primary_moment_of_area,
        secondary_moment_of_area=secondary_moment_of_area,
        torsional_constant=torsional_constant,
        density=7850,
        cross_sectional_area=cross_sectional_area,
        coordinates=(0, 0, 2*1.7),
        vertex_IDs=1,
        edge_polar_rotation=0
    )
    
    system.fix_vertices((0, ))
    with system.added_forces({1: lambda t: [1000, 0, 0, 0, 0, 0]}):
        initial_condition_solver = Static(system).solve()
        
    end_time = 1
    time_increment = 0.01
    scaling_factor = 100
    displaced_shape_points = Newmark(system, initial_condition_solver, end_time, time_increment).solve().get_displaced_shape_position(scaling_factor)

    plot_lines = list()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for displaced_shape_point in displaced_shape_points:
        plot_lines.append(ax.plot(displaced_shape_point[0, :, 0], displaced_shape_point[0, :, 1], displaced_shape_point[0, :, 2], linewidth=2.0))

    def update(frame):
        for i, lines in enumerate(plot_lines):
            for line in lines:
                line.set_data_3d(*displaced_shape_points[i][frame, :].T)
        ax.set_title(f"t = {frame*time_increment:.3f} s")
        return plot_lines

    ani = animation.FuncAnimation(fig=fig, func=update, frames=int(end_time/time_increment), interval=int(time_increment))

    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid()
    plt.show()