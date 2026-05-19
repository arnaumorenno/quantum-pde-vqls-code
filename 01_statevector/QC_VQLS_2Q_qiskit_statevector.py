
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# --- 1. PROBLEM SETUP ---
A = np.array([[ 1.5, -0.2,  0.0,  0.0],
              [-0.2,  1.5, -0.2,  0.0],
              [ 0.0, -0.2,  1.5, -0.2],
              [ 0.0,  0.0, -0.2,  1.5]])

x_grid      = np.linspace(-1, 1, 4)
b_classical = np.exp(-3 * x_grid**2)
b_norm      = b_classical / np.linalg.norm(b_classical)
classical_solution = np.linalg.solve(A, b_norm)

print("\n--- MATRIX PROPERTIES ---")
print(f"Matrix Condition Number k(A): {np.linalg.cond(A):.4f}")

# --- 2. CIRCUIT (unchanged structure) ---
weights = ParameterVector('w', 4)

qc = QuantumCircuit(2)
qc.ry(weights[0], 0)
qc.ry(weights[1], 1)
qc.cx(0, 1)
qc.ry(weights[2], 0)
qc.ry(weights[3], 1)
qc.cx(0, 1)

# --- 3. TRUE VQLS COST (Statevector — exact, no shots needed) ---
def get_statevector(w_vals):
    """
    Extracts full complex statevector — equivalent to infinite shots.
    Preserves phase information completely.
    No sampler, no estimator, no shots.
    """
    param_dict = dict(zip(weights, w_vals))
    bound_qc   = qc.assign_parameters(param_dict)
    return Statevector(bound_qc).data  # complex amplitudes, shape (4,)

def vqls_cost(w_vals):
    """
    True VQLS cost: C(θ) = 1 - |⟨b|A|ψ⟩|² / ⟨ψ|A²|ψ⟩
    All computed from exact statevector — no phase loss.
    """
    psi   = get_statevector(w_vals)
    Apsi  = A @ psi
    numer = abs(np.dot(b_norm.conj(), Apsi))**2    # |⟨b|A|ψ⟩|²
    denom = np.real(np.dot(psi.conj(), A @ Apsi))  # ⟨ψ|A²|ψ⟩
    return float(1.0 - numer / (denom + 1e-8))

# --- 4. EXECUTION ---
np.random.seed(1337)
initial_weights = np.random.randn(4) * 0.1
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

print("\nOptimizing 2Q True VQLS (Statevector — no shots, no sampling noise)")
start_time = time.time()

result = minimize(objective_fn, initial_weights,
                  method='COBYLA',
                  options={'maxiter': 1000, 'disp': True})

end_time      = time.time()
final_weights = result.x

print("\n--- QUANTUM RESOURCE METRICS ---")
print(f"Qubits Used          : 2")
print(f"Ansatz Layers        : 2")
print(f"Ansatz Parameters    : 4")
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
print(f"Classical Exact Solution : {np.round(classical_solution, 4)}")
print(f"True VQLS Solution       : {np.round(scaled_quantum, 4)}")
print(f"Mean Absolute Error      : {np.mean(state_error):.4f}")

# --- 6. COLE-HOPF DECODE ---
nu = 0.1
dx = x_grid[1] - x_grid[0]

d_phi_dx_quantum   = np.gradient(scaled_quantum, dx)
d_phi_dx_classical = np.gradient(classical_solution, dx)
u_quantum   = -2 * nu * (d_phi_dx_quantum   / scaled_quantum)
u_classical = -2 * nu * (d_phi_dx_classical / classical_solution)

velocity_error = np.abs(u_classical - u_quantum)
print("\n--- FINAL BURGERS' VELOCITY (u) ---")
print(f"Classical Velocity : {np.round(u_classical, 4)}")
print(f"Hardware Velocity  : {np.round(u_quantum, 4)}")
print(f"Mean Absolute Error: {np.mean(velocity_error):.4f}")

# --- 7. PLOTS ---
print("\nGenerating Plots...")

fig1, (ax_cost, ax_res) = plt.subplots(1, 2, figsize=(14, 5))
ax_cost.plot(cost_history, 'b-', linewidth=2, label='VQLS Cost $C(\\theta)$')
ax_cost.set_yscale('log')
ax_cost.set_title("True VQLS Cost Convergence (2 Qubits)")
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
ax_heat.plot(x_grid, scaled_quantum, 'bo--', linewidth=2, markersize=8, label='True VQLS (Statevector)')
ax_heat.set_title("Heat State $\\phi(x, t)$")
ax_heat.set_xlabel("Space (x)"); ax_heat.set_ylabel("Amplitude")
ax_heat.grid(True); ax_heat.legend()

ax_vel.plot(x_grid, u_classical, 'k-', linewidth=2, label='Classical Exact')
ax_vel.plot(x_grid, u_quantum, 'ro--', linewidth=2, markersize=8, label='True VQLS (Statevector)')
ax_vel.set_title("Burgers' Fluid Velocity $u(x, t)$")
ax_vel.set_xlabel("Space (x)"); ax_vel.set_ylabel("Velocity")
ax_vel.grid(True); ax_vel.legend()
plt.tight_layout()
plt.show()

# Circuit diagram with final weights
param_dict = dict(zip(weights, np.round(final_weights, 3)))
qc_bound   = qc.assign_parameters(param_dict)
fig3       = qc_bound.draw('mpl', style='clifford')
plt.tight_layout()
plt.show()

# --- 8. NOISE ROBUSTNESS ---
print("\n" + "="*68)
print(" STARTING NOISE ROBUSTNESS ANALYSIS")
print("="*68)
print("Simulating Gaussian noise on the target vector 'b'...")
print(f"{'Noise Variance (%)':<20} | {'Heat State Error (%)':<22} | {'Velocity Error (%)':<20}")
print("-" * 68)

noise_levels     = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
robustness_results = []

for noise_var in noise_levels:
    noise        = np.random.normal(0, noise_var * np.mean(b_classical), size=b_classical.shape)
    b_noisy_norm = (b_classical + noise)
    b_noisy_norm = b_noisy_norm / np.linalg.norm(b_noisy_norm)

    def noisy_objective_fn(w):
        psi   = get_statevector(w)
        Apsi  = A @ psi
        numer = abs(np.dot(b_noisy_norm.conj(), Apsi))**2
        denom = np.real(np.dot(psi.conj(), A @ Apsi))
        return float(1.0 - numer / (denom + 1e-8))

    res_noisy = minimize(noisy_objective_fn,
                         np.random.randn(4) * 0.1,
                         method='COBYLA',
                         options={'maxiter': 500, 'disp': False})

    psi_noisy    = get_statevector(res_noisy.x)
    noisy_qs     = np.real(psi_noisy)
    s            = np.dot(noisy_qs, classical_solution) / (np.dot(noisy_qs, noisy_qs) + 1e-12)
    scaled_noisy = noisy_qs * s

    heat_err = np.linalg.norm(classical_solution - scaled_noisy) / \
               np.linalg.norm(classical_solution) * 100
    u_noisy  = -2 * nu * (np.gradient(scaled_noisy, dx) / scaled_noisy)
    vel_err  = np.linalg.norm(u_classical - u_noisy) / np.linalg.norm(u_classical) * 100

    robustness_results.append((noise_var * 100, heat_err, vel_err))
    print(f"{int(noise_var * 100):<18}% | {heat_err:<20.2f}% | {vel_err:<18.2f}%")

print("=" * 68)
print("Noise Robustness Analysis Complete.")