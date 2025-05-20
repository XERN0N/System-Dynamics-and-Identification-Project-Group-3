# Script for model updating based on sensitivity analysis
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from OriginalModel import *
from typing import Any
import inspect

# Function to compute the Jacobian matrix
def compute_jacobian(phi, omega, **model_parameters):
    perturbations = list()
    list_of_perturbed_model_parameters: list[dict[str, Any]] = list()
    for key, value in model_parameters.items():
        perturbed_model_parameter = model_parameters.copy()
        perturbation = value*1e-2
        perturbations.append(perturbation)
        perturbed_model_parameter.update({key: value + perturbation})
        list_of_perturbed_model_parameters.append(perturbed_model_parameter)
 
    # Base model
    base_model = generate_original_model(**model_parameters)
    base_mass_matrix, base_stiffness_matrix, _ = base_model.get_system_level_matrices()

    mass_matrix_derivative = list()
    stiffness_matrix_derivative = list()
    # perturb
    for i, perturbed_model_parameter in enumerate(list_of_perturbed_model_parameters):
        perturbed_model = generate_original_model(**perturbed_model_parameter)
        perturbed_mass_matrix, perturbed_stiffness_matrix, _ = perturbed_model.get_system_level_matrices()
        mass_matrix_derivative.append((perturbed_mass_matrix - base_mass_matrix) / perturbations[i])
        stiffness_matrix_derivative.append((perturbed_stiffness_matrix - base_stiffness_matrix) / perturbations[i])

    # Compute Jacobian
    Jacobian = np.zeros((len(omega), len(model_parameters)))
    for i in range(Jacobian.shape[0]):
        for j in range(Jacobian.shape[1]):
            phi_i = phi[:, i]
            w = omega[i]
            Jacobian[i, j] = 1/2 * phi_i.T @ (1/w * stiffness_matrix_derivative[j] - w * mass_matrix_derivative[j]) @ phi_i
    return Jacobian

# Function to perform the Newton update
def newton_update(parameters: dict[str, Any | None], target_frequencies: npt.ArrayLike, frequency_indices: npt.ArrayLike, global_relative_tolerance: float = 1e-6, iteration_limit: int = 100):
    """
    Performs model updating by varying beam parameters until eigen frequencies match target frequencies.

    Parameters
    ----------
    parameters : dict[str, Any | None]
        The parameters to update as keys with their initial values. If the value is None, a default value is used.
    target_frequencies : array_like
        The frequencies to aim for with shape (n,).
    frequency_indices : array_like
        The indices of the target frequencies in the model eigen values.
    global_relative_tolerance : float, optional
        The relative tolerance stop criteria for all parameters. Default is 1e-6.
    iteration_limit : int, optional
        The maximum number of iterations stop criteria. Default is 100.

    Returns
    -------
    tuple[Beam_Lattice, dict[str, Any], ndarray]
        A tuple containing firstly the final model, secondly the history of the parameters, and thirdly the history of the features.
    """
    frequency_indices = np.asarray(frequency_indices)

    default_values = Default_beam_edge_parameters.default_beam_edge_parameters.value.copy()
    default_values.update(Default_beam_edge_parameters.default_point_mass_parameters.value.copy())

    parameter_history: dict[str, list[Any]] = dict()
    for key, value in parameters.items():
        if value is None:
            value = default_values[key]
            parameters[key] = value
        parameter_history.update({key: [value]})
    
    parameter_old = parameters
    delta = 2 * global_relative_tolerance
    k = 1
    feature_history = []

    while delta >= global_relative_tolerance:
        model = generate_original_model(**parameter_old)
        omega, phi,_ = model.get_modal_param(eigen_value_sort=True, convert_to_frequencies=False, normalize=True)
        omega = np.sqrt(omega[frequency_indices])
        feature_history.append(omega)
        Jacobian = compute_jacobian(phi, omega, **parameter_old)
        parameter_new = dict()
        parameter_update = pinv(Jacobian) @ (target_frequencies - omega)
        for i, (key, value) in enumerate(parameter_old.items()):
            parameter_new.update({key: value + parameter_update[i]})
        delta = np.max(np.abs((np.array(list(parameter_new.values())) - list(parameter_old.values()))) / np.abs(list(parameter_old.values())))
        parameter_old = parameter_new

        for key, value in parameter_new.items():
            parameter_history[key].append(value)

        k += 1
        if k > iteration_limit:
            print("Not converged within iteration limit")
            break

    print(f"Convergence after {k-1} iterations")
    return model, parameter_history, np.array(feature_history).T

if __name__ == "__main__":

    # System features (for comparison)
    system_features = generate_original_model()
    #omega_target, _, _ = system_features.get_modal_param(eigen_value_sort=True, convert_to_frequencies=False, normalize=True)
    #omega_target = np.sqrt(np.abs(omega_target[[0, 2, 4, 9]]))
    omega_target = np.array([26.704, 37.07, 225.315])

    # Model parameters
    theta0 = {'cross_sectional_area': None, 'point_mass': None}

    # Run Newton update
    _, theta_hist, omega_hist = newton_update(theta0, omega_target, (0, 1, 2))    

    parameter_fig, parameter_axs = plt.subplots(len(theta_hist))

    for i, key in enumerate(theta_hist.keys()):
        parameter_axs[i].plot(theta_hist[key])
        parameter_axs[i].grid()
        parameter_axs[i].legend()
        parameter_axs[i].set_title(f"Convergence of {key}")
    
    parameter_fig.tight_layout()

    feature_fig, feature_axs = plt.subplots(len(omega_hist))

    for i, feature_history in enumerate(omega_hist):
        feature_axs[i].plot(feature_history)
        feature_axs[i].grid()
        feature_axs[i].legend()
        feature_axs[i].set_title(f"Convergence of feature {i}")

    feature_fig.tight_layout()

    plt.show()