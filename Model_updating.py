# Script for model updating based on sensitivity analysis
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.linalg import eigh, pinv
from SystemModels import Beam_Lattice
from OriginalModel import generate_original_model


model = generate_original_model()

# Function to compute the Jacobian matrix
def compute_jacobian(phi, omega, E, rho, step=1e-6):
    delta_E = 1e3

    # Base model
    model = generate_original_model(density=rho, E_modulus=E)
    M_base, K_base, _ = model.get_system_level_matrices()

    # Perturb E
    model_E = generate_original_model(density=rho,E_modulus= E + delta_E)
    _, K_E, _ = model_E.get_system_level_matrices()
    dK_dE = (K_E - K_base) / delta_E

    # Perturb rho
    dM_drho = M_base / rho

    # Compute Jacobian
    g = len(omega)
    J = np.zeros((g, 2))  # 2 columns: [∂ω/∂E, ∂ω/∂rho]
    for i in range(g):
        phi_i = phi[:, i]
        w = omega[i]
        J[i, 0] = (phi_i.T @ dK_dE @ phi_i) / (2 * w)
        J[i, 1] = - (w / 2) * (phi_i.T @ dM_drho @ phi_i)
    return J

# Function to perform the Newton update
def newton_update(theta0: npt.NDArray, omega_target, eps=1e-3, it_limit=100, alpha=1):
    theta_hist = [theta0.copy()]
    delta = 2 * eps
    k = 1

    while delta >= eps:
        E, rho = theta_hist[-1]
        model = generate_original_model(density=rho, E_modulus=E)
        omega, phi,_ = model.get_modal_param()
        omega = omega[:5] * 2*np.pi
        J = compute_jacobian(phi, omega, E, rho)
        update_step = pinv(J) @ (omega_target - omega)

        theta_k = theta_hist[-1] + alpha * update_step  # Damped update
        # theta_k = np.maximum(theta_k, np.array([1e9, 100.0]))  # Clamp E and ρ to physical values

        delta = np.max(np.abs((theta_k - theta_hist[-1]) / (theta_hist[-1] + 1e-12)))
        theta_hist.append(theta_k)

        k += 1

        if k > it_limit:
            print("Not converged within iteration limit")
            break

    print(f"Convergence after {k-1} iterations")
    return np.array(theta_hist).T

# Create target system (for comparison)

system_features = generate_original_model()
omega_target, _, _ = system_features.get_modal_param()
omega_target = omega_target[:5]*2*np.pi

# Initial guess for E and rho
E_init = 2.2e11  # Pa (steel)
rho_init = 7950  # kg/m³
theta0 = np.array([E_init, rho_init])

# Run Newton update
theta_hist = newton_update(theta0, omega_target)

# Plot convergence
iterations = np.arange(theta_hist.shape[1])

# Plotting the convergence of E modulus
plt.figure()
plt.plot(theta_hist[0], label="E modulus")
plt.xlabel("Iteration")
plt.ylabel("Parameter estimate")
plt.title("Convergence of E modulus")
plt.grid(True)
plt.legend()

# Plotting the convergence of density
plt.figure()
plt.plot(theta_hist[1], label="Density", color='green')
plt.xlabel("Iteration")
plt.ylabel("Density [kg/m³]")
plt.title("Convergence of Density")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
