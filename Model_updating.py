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
def newton_update(theta0: dict[str, Any | None], omega_target, eps=1e-6, it_limit=100):

    default_values = Default_beam_edge_parameters.default_beam_edge_parameters.value.copy()
    default_values.update(Default_beam_edge_parameters.default_point_mass_parameters.value.copy())

    theta_hist: dict[str, list[Any]] = dict()
    for key, value in theta0.items():
        if value is None:
            value = default_values[key]
            theta0[key] = value
        theta_hist.update({key: [value]})
    
    theta_old = theta0
    delta = 2 * eps
    k = 1

    while delta >= eps:
        model = generate_original_model(**theta_old)
        omega, phi,_ = model.get_modal_param(eigen_value_sort=True, convert_to_frequencies=False, normalize=True)
        omega = np.sqrt(omega[[0, 1, 2]])
        Jacobian = compute_jacobian(phi, omega, **theta_old)
        theta_new = dict()
        parameter_update = pinv(Jacobian) @ (omega_target - omega)
        for i, (key, value) in enumerate(theta_old.items()):
            theta_new.update({key: value + parameter_update[i]})

        delta = np.max(np.abs((np.array(list(theta_new.values())) - np.array(list(theta_old.values()))) / (np.array(list(theta_old.values())) + 1e-12)))
        
        theta_old = theta_new

        for key, value in theta_new.items():
            theta_hist[key].append(value)

        k += 1
        if k > it_limit:
            print("Not converged within iteration limit")
            break

    print(f"Convergence after {k-1} iterations")
    print(omega)
    return theta_hist

if __name__ == "__main__":

    # System features (for comparison)
    system_features = generate_original_model()
    #omega_target, _, _ = system_features.get_modal_param(eigen_value_sort=True, convert_to_frequencies=False, normalize=True)
    #omega_target = np.sqrt(np.abs(omega_target[[0, 2, 4, 9]]))
    omega_target = np.array([26.704, 37.07, 225.315])

    # Model parameters
    theta0 = {'E_modulus': None, 'density': None, 'point_mass': None}

    # Run Newton update
    theta_hist = newton_update(theta0, omega_target)
    """ print(f"Iteration 0:     E = {theta_hist[0, 0]:.3e}, rho = {theta_hist[1, 0]:.2f}")
    print(f"Iteration {theta_hist.shape[1] - 1}: E = {theta_hist[0, -1]:.3e}, rho = {theta_hist[1, -1]:.2f}")
    print(f"Number of eigenfreq {omega_target.shape}") """
    

    # Plotting the convergence of E modulus and density
    plt.figure()
    plt.plot(theta_hist['E_modulus'], label="E modulus")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter estimate")
    plt.title("Convergence of E modulus")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(theta_hist['density'], label="Density", color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Density [kg/m³]")
    plt.title("Convergence of Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(theta_hist['point_mass'], label="point mass", color='green')
    plt.xlabel("Iteration")
    plt.ylabel("mass [kg]")
    plt.title("Convergence of point mass")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
