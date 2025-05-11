# Script for model updating based on sensitivity analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, pinv

def chain(m_vec, c_vec, k_vec):
    dof = len(m_vec)
    M = np.diag(m_vec)
    K = np.zeros((dof, dof))
    C = np.zeros((dof, dof))
    
    for i in range(dof):
        K[i, i] += k_vec[i]
        if i > 0:
            K[i, i] += k_vec[i-1]
            K[i, i-1] -= k_vec[i-1]
            K[i-1, i] -= k_vec[i-1]

        C[i, i] += c_vec[i]
        if i > 0:
            C[i, i] += c_vec[i-1]
            C[i, i-1] -= c_vec[i-1]
            C[i-1, i] -= c_vec[i-1]

    return M, C, K

def compute_modal_properties(M, K):
    lam, phi = eigh(K, M)
    omega = np.sqrt(np.maximum(lam, 0))
    idx = np.argsort(omega)
    return omega[idx], phi[:, idx]

def compute_jacobian(phi, omega, dK_list):
    g = len(omega)
    p = len(dK_list)
    J = np.zeros((g, p))
    for j in range(g):
        for i in range(p):
            J[j, i] = (phi[:, j].T @ dK_list[i] @ phi[:, j]) / (2 * omega[j])
    return J

def newton_update(theta0, dK_list, M, omega_target, eps=1e-8, it_limit=1000):
    theta_hist = [theta0.copy()]
    delta = 2 * eps
    i = 1
    while delta >= eps:
        theta = theta_hist[-1]
        _, _, K = chain(np.zeros_like(theta), np.zeros_like(theta), theta)
        omega, phi = compute_modal_properties(M, K)
        J = compute_jacobian(phi, omega, dK_list)
        theta_new = theta + pinv(J) @ (omega_target - omega)
        delta = np.max(np.abs((theta_new - theta) / (theta + 1e-12)))
        theta_hist.append(theta_new)
        i += 1
        if i > it_limit:
            print("Not converged within iteration limit")
            break
    print(f"Converged after {i-1} iterations")
    return np.array(theta_hist).T  # shape: (dof, iterations)

# --- MAIN SCRIPT ---
dof = 3
m = np.ones(dof)
k = 100 * np.ones(dof)
c = 0.01 * np.ones(dof)

# Initial model
M, _, K = chain(m, np.zeros_like(c), k)
omega_model, phi_model = compute_modal_properties(M, K)

# Perturbed system (truth)
ks = k.copy()
ks[0] *= 0.8
ks[-1] *= 1.1
_, _, Ks = chain(m, np.zeros_like(c), ks)
omega_target, _ = compute_modal_properties(M, Ks)

# Sensitivities
dK_list = []
for i in range(dof):
    dk = np.zeros(dof)
    dk[i] = 1.0
    _, _, dK = chain(m, np.zeros(dof), dk)
    dK_list.append(dK)

# Run Newton update
theta0 = k.copy()
theta_hist = newton_update(theta0, dK_list, M, omega_target)

# Plotting
plt.figure(figsize=(8, 5))
for i in range(dof):
    plt.semilogy(theta_hist[i], label=f'$k_{i+1}$')
plt.xlabel('Iteration')
plt.ylabel('Parameter estimate')
plt.title('Evolution of estimated parameters')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
