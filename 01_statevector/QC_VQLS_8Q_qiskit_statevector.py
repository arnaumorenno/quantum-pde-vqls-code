
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# --- 1. PROBLEM SETUP ---
num_qubits = 8
num_nodes  = 2**num_qubits  # 256

main_diag = np.full(num_nodes, 1.5)
off_diag  = np.full(num_nodes - 1, -0.2)
A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

x_grid      = np.linspace(-1, 1, num_nodes)
b_classical = np.exp(-3 * x_grid**2)
b_norm      = b_classical / np.linalg.norm(b_classical)
classical_solution = np.linalg.solve(A, b_norm)

print("\n--- MATRIX PROPERTIES ---")
print(f"Matrix Condition Number k(A): {np.linalg.cond(A):.4f}")

# --- 2. CIRCUIT ---
# Paper pattern: num_layers = num_qubits - 1 = 7
num_layers = 7
num_params = num_qubits * num_layers  # = 56
weights    = ParameterVector('w', num_params)

qc = QuantumCircuit(num_qubits)
idx = 0
for _ in range(num_layers):
    for i in range(num_qubits):
        qc.ry(weights[idx], i)
        idx += 1
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)   # linear chain — no ring

# --- 3. VQLS COST FUNCTION---
def get_statevector(w_vals):
    param_dict = dict(zip(weights, w_vals))
    bound_qc   = qc.assign_parameters(param_dict)
    return Statevector(bound_qc).data  # complex, shape (256,)

def vqls_cost(w_vals):
    psi   = get_statevector(w_vals)
    Apsi  = A @ psi
    numer = abs(np.dot(b_norm.conj(), Apsi))**2
    denom = np.real(np.dot(psi.conj(), A @ Apsi))
    return float(1.0 - numer / (denom + 1e-8))

# --- 4. EXECUTION ---
np.random.seed(2026)
cost_history, residual_history = [], []

def objective_fn(w):
    cost = vqls_cost(w)
    cost_history.append(cost)
    psi    = get_statevector(w)
    x_real = np.real(psi)
    scalar = np.dot(x_real, classical_solution) / (np.dot(x_real, x_real) + 1e-12)
    residual = np.linalg.norm(A @ (x_real * scalar) - b_norm)
    residual_history.append(residual)
    return cost

print(f"\nOptimizing 8-Qubit True VQLS ({num_params} parameters, {num_layers} layers)...")
print("Using exact Statevector — no shots, no sampling noise.\n")
start_time = time.time()

result = minimize(objective_fn, np.random.randn(num_params) * 0.1,
                  method='COBYLA',
                  options={'maxiter': 6000, 'disp': True,
                           'rhobeg': 0.2, 'catol': 1e-6})

end_time      = time.time()
final_weights = result.x

print("\n--- QUANTUM RESOURCE METRICS ---")
print(f"Qubits Used          : {num_qubits}")
print(f"Ansatz Layers        : {num_layers}")
print(f"Ansatz Parameters    : {num_params}")
print(f"Shots                : None (exact Statevector)")
print(f"Total Iterations     : {len(cost_history)}")
print(f"Total Execution Time : {end_time - start_time:.2f} seconds")
print(f"Final VQLS Cost      : {result.fun:.6f}")
print(f"\nfinal_weights = np.array({list(np.round(final_weights, 8))})")

# --- 5. EXTRACT SOLUTION ---
psi_final        = get_statevector(final_weights)
quantum_solution = np.real(psi_final)
scalar           = np.dot(quantum_solution, classical_solution) / \
                   (np.dot(quantum_solution, quantum_solution) + 1e-12)
scaled_quantum   = quantum_solution * scalar
state_error      = np.abs(classical_solution - scaled_quantum)

print("\n--- HEAT STATE RESULTS ---")
print(f"Mean Absolute Error : {np.mean(state_error):.4f}")

# --- 6. COLE-HOPF DECODE ---
nu = 0.1
dx = x_grid[1] - x_grid[0]

d_phi_dx_quantum   = np.gradient(scaled_quantum, dx)
d_phi_dx_classical = np.gradient(classical_solution, dx)
u_quantum   = -2 * nu * (d_phi_dx_quantum   / scaled_quantum)
u_classical = -2 * nu * (d_phi_dx_classical / classical_solution)

print("\n--- FINAL BURGERS' VELOCITY (u) ---")
print(f"Mean Absolute Error: {np.mean(np.abs(u_classical - u_quantum)):.4f}")

# --- 7. PLOTS ---
fig1, (ax_cost, ax_res) = plt.subplots(1, 2, figsize=(14, 5))
ax_cost.plot(cost_history, 'b-', linewidth=2, label='VQLS Cost $C(\\theta)$')
ax_cost.set_yscale('log')
ax_cost.set_title(f"True VQLS Cost Convergence ({num_qubits} Qubits)")
ax_cost.set_xlabel('Iteration'); ax_cost.set_ylabel('Cost')
ax_cost.grid(True, which="both", ls="--"); ax_cost.legend()

ax_res.plot(residual_history, 'r-', linewidth=2, label='Residual $||Ax - b||$')
ax_res.set_yscale('log')
ax_res.set_title("Physical Residual Error")
ax_res.set_xlabel('Iteration'); ax_res.set_ylabel('Residual')
ax_res.grid(True, which="both", ls="--"); ax_res.legend()
plt.tight_layout()

fig2, (ax_heat, ax_vel) = plt.subplots(1, 2, figsize=(14, 5))
ax_heat.plot(x_grid, classical_solution, 'k-', linewidth=2, label='Classical Exact')
ax_heat.plot(x_grid, scaled_quantum, 'b-', linewidth=2, alpha=0.8, label='True VQLS (Statevector)')
ax_heat.set_title(f"Heat State $\\phi(x, t)$ ({num_nodes} Nodes)")
ax_heat.set_xlabel("Space (x)"); ax_heat.set_ylabel("Amplitude")
ax_heat.grid(True); ax_heat.legend()

ax_vel.plot(x_grid, u_classical, 'k-', linewidth=2, label='Classical Exact')
ax_vel.plot(x_grid, u_quantum, 'r-', linewidth=2, alpha=0.8, label='True VQLS (Statevector)')
ax_vel.set_title(f"Burgers' Fluid Velocity $u(x, t)$ ({num_nodes} Nodes)")
ax_vel.set_xlabel("Space (x)"); ax_vel.set_ylabel("Velocity")
ax_vel.set_ylim([-2, 2])
ax_vel.grid(True); ax_vel.legend()
plt.tight_layout()

param_dict = dict(zip(weights, np.round(final_weights, 3)))
qc_bound   = qc.assign_parameters(param_dict)
fig3       = qc_bound.draw('mpl', style='clifford', fold=-1)
plt.tight_layout()
plt.show()

# --- 8. NOISE ROBUSTNESS ---
print("\n" + "="*68)
print(" STARTING NOISE ROBUSTNESS ANALYSIS")
print("="*68)
print(f"{'Noise Variance (%)':<20} | {'Heat State Error (%)':<22} | {'Velocity Error (%)':<20}")
print("-" * 68)

noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
for noise_var in noise_levels:
    noise        = np.random.normal(0, noise_var * np.mean(b_classical), size=b_classical.shape)
    b_noisy_norm = (b_classical + noise) / np.linalg.norm(b_classical + noise)

    def noisy_objective_fn(w):
        psi   = get_statevector(w)
        Apsi  = A @ psi
        numer = abs(np.dot(b_noisy_norm.conj(), Apsi))**2
        denom = np.real(np.dot(psi.conj(), A @ Apsi))
        return float(1.0 - numer / (denom + 1e-8))

    res_noisy  = minimize(noisy_objective_fn,
                          np.random.randn(num_params) * 0.1,
                          method='COBYLA',
                          options={'maxiter': 500, 'disp': False,
                                   'rhobeg': 0.2, 'catol': 1e-6})

    psi_noisy    = get_statevector(res_noisy.x)
    noisy_qs     = np.real(psi_noisy)
    s            = np.dot(noisy_qs, classical_solution) / (np.dot(noisy_qs, noisy_qs) + 1e-12)
    scaled_noisy = noisy_qs * s

    heat_err = np.linalg.norm(classical_solution - scaled_noisy) / \
               np.linalg.norm(classical_solution) * 100
    u_noisy  = -2 * nu * (np.gradient(scaled_noisy, dx) / scaled_noisy)
    vel_err  = np.linalg.norm(u_classical - u_noisy) / np.linalg.norm(u_classical) * 100
    print(f"{int(noise_var*100):<18}% | {heat_err:<20.2f}% | {vel_err:<18.2f}%")

print("=" * 68)
print("Noise Robustness Analysis Complete.")