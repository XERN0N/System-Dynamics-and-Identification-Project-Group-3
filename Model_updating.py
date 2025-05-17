# Script for model updating based on sensitivity analysis
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eigh, pinv
from SystemModels import Beam_Lattice
from OriginalModel import generate_original_model
from SystemIdentifier import system_identifier


model = generate_original_model()

# Function to compute the Jacobian matrix
def compute_jacobian(phi, omega, E, rho):
    delta_E = 1e-1

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
def newton_update(theta0: npt.NDArray, omega_target, eps=1e-6, it_limit=1000, alpha=1):
    theta_hist = [theta0.copy()]
    delta = 2 * eps
    k = 1

    while delta >= eps:
        E, rho = theta_hist[-1]
        model = generate_original_model(density=rho, E_modulus=E)
        omega, phi,_ = model.get_modal_param(eigen_value_sort=False, convert_to_frequencies=False, normalize=True)
        omega = np.sqrt(omega[:5])
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



if __name__ == "__main__":

    # SSI for system (measurement)
    data_2d_prim = pd.read_csv(r"Sorted timeseries\Forced\2D\Accelerometer\Calibrated\timeseries_serial_output_2_gr3_2D.csv", nrows=1000)
    data_2d_sec = pd.read_csv(r"Sorted timeseries\Forced\2D\Accelerometer\Calibrated\timeseries_serial_output_2_gr3_2D.csv", nrows=1000)
    data_1d_twist = pd.read_csv("Sorted timeseries/Free/1D/Calibrated/Torsional_axis_data/X/timeseries_gr3_twisting.txt", nrows=1000)

    # Desired column order
    ordered_columns = [
    "Sensor 1 - X", "Sensor 2 - X", "Sensor 3 - X", "Sensor 4 - X",
    "Sensor 1 - Y", "Sensor 2 - Y", "Sensor 3 - Y", "Sensor 4 - Y"]

    # Extract reordered sensor data as a NumPy array
    sensor_data_2d_prim = data_2d_prim[ordered_columns[4:8]].to_numpy()  # shape: (1000, 8)
    sensor_data_2d_sec = data_2d_sec[ordered_columns[0:4]].to_numpy()  # shape: (1000, 8)
    sensor_data_1d_twist = data_1d_twist[ordered_columns[0:4]].to_numpy()  # shape: (1000, 8)

    # Calculate eigenfrequencies [Hz] of the data
    omega_target_si_2d_primary,_,_ = system_identifier(output_data=sensor_data_2d_prim)
    omega_target_si_2d_secondary,_,_ = system_identifier(output_data=sensor_data_2d_sec)
    omega_target_si_twist,_,_ = system_identifier(output_data=sensor_data_1d_twist, time_step= 1/927)

    # Eigenfrequencies [Hz] of the model
    system_features = generate_original_model()
    omega_model, _, _ = system_features.get_modal_param(eigen_value_sort=False, convert_to_frequencies=True, normalize=True)

    from itertools import zip_longest

    print(f"{'Primary [Hz]':>15} {'Secondary [Hz]':>15} {'Torsional [Hz]':>15} {'Model [Hz]':>15}")

    for p, s, t, m in zip_longest(omega_target_si_2d_primary, omega_target_si_2d_secondary, omega_target_si_twist, omega_model[:12], fillvalue=np.nan):
        print(f"{p:15.2f} {s:15.2f} {t:15.2f} {m:15.2f}")


    # System features (for comparison)
    system_features = generate_original_model()
    omega_target, _, _ = system_features.get_modal_param(eigen_value_sort=False, convert_to_frequencies=False, normalize=True)
    omega_target = np.sqrt(omega_target)[:5]

    # Model parameters - Initial guess for E and rho
    E_init = 2.2e11  # Pa (steel)
    rho_init = 7950  # kg/m³
    theta0 = np.array([E_init, rho_init])

    # Run Newton update
    theta_hist = newton_update(theta0, omega_target)
    print(f"Iteration 0:     E = {theta_hist[0, 0]:.3e}, rho = {theta_hist[1, 0]:.2f}")
    print(f"Iteration {theta_hist.shape[1] - 1}: E = {theta_hist[0, -1]:.3e}, rho = {theta_hist[1, -1]:.2f}")

    # Plotting the convergence of E modulus and density
    plt.figure()
    plt.plot(theta_hist[0], label="E modulus")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter estimate")
    plt.title("Convergence of E modulus")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(theta_hist[1], label="Density", color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Density [kg/m³]")
    plt.title("Convergence of Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
