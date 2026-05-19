
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# --- 1. PROBLEM SETUP ---
num_qubits = 4
num_nodes  = 2**num_qubits  # 16

main_diag = np.full(num_nodes, 1.5)
off_diag  = np.full(num_nodes - 1, -0.2)
A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

x_grid      = np.linspace(-1, 1, num_nodes)
b_classical = np.exp(-3 * x_grid**2)
b_norm      = b_classical / np.linalg.norm(b_classical)
classical_solution = np.linalg.solve(A, b_norm)

print("\n--- MATRIX PROPERTIES ---")
print(f"Matrix Condition Number k(A): {np.linalg.cond(A):.4f}")

# --- 2. ANSATZ CIRCUIT ---
num_layers = 3
num_params = num_qubits * num_layers
weights    = ParameterVector('w', num_params)

qc = QuantumCircuit(num_qubits)
idx = 0
for _ in range(num_layers):
    for i in range(num_qubits):
        qc.ry(weights[idx], i)
        idx += 1
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)

qc_meas = qc.copy()
qc_meas.measure_all()

# --- 3. VQLS COST FUNCTION ---
def get_statevector(w_vals):
    """
    Extracts full complex statevector amplitudes.
    This is the statevector-simulator equivalent of the Hadamard test.
    It preserves complete phase information — the key fix over sqrt(probs).
    """
    param_dict = dict(zip(weights, w_vals))
    bound_qc   = qc.assign_parameters(param_dict)
    sv         = Statevector(bound_qc)
    return sv.data  # complex amplitudes, shape (16,)

def vqls_cost_true(w_vals):
    """
    True VQLS cost: C(θ) = 1 - |⟨b|A|ψ(θ)⟩|² / ⟨ψ(θ)|A²|ψ(θ)⟩

    Original broken version:
        probs = |ψ|²   → loses sign → A @ sqrt(probs) is wrong
        numer = (b · A @ sqrt(probs))²  ← not the true inner product

    This version:
        psi = full complex ψ  → phases intact
        numer = |b† · A|ψ⟩|²  ← true quantum inner product
    """
    psi   = get_statevector(w_vals)            # complex, shape (16,)
    Apsi  = A @ psi                             # A|ψ⟩, complex
    numer = abs(np.dot(b_norm.conj(), Apsi))**2 # |⟨b|A|ψ⟩|²
    denom = np.real(np.dot(psi.conj(), A @ Apsi)) # ⟨ψ|A²|ψ⟩
    return float(1.0 - numer / (denom + 1e-8))

cost_history, residual_history = [], []

def objective_fn(w):
    cost = vqls_cost_true(w)
    cost_history.append(cost)

    psi    = get_statevector(w)
    x_real = np.real(psi)   # RY+CX ansatz → real amplitudes
    scalar = np.dot(x_real, classical_solution) / (np.dot(x_real, x_real) + 1e-12)
    residual = np.linalg.norm(A @ (x_real * scalar) - b_norm)
    residual_history.append(residual)
    return cost

# --- 4. OPTIMIZATION WITH RESTARTS ---
np.random.seed(42)
print(f"\nOptimizing True VQLS ({num_params} params, {num_layers} layers)")
print("Cost: |⟨b|A|ψ⟩|² / ⟨ψ|A²|ψ⟩  ← full phase information\n")

start_time  = time.time()
best_result = None

for restart in range(3):
    print(f"  Restart {restart+1}/3...")
    init_w = np.random.randn(num_params) * 0.1
    res    = minimize(objective_fn, init_w,
                      method='COBYLA',
                      options={'maxiter': 3000, 'disp': True,
                               'rhobeg': 0.2, 'catol': 1e-6})
    print(f"  Cost: {res.fun:.6f}")
    if best_result is None or res.fun < best_result.fun:
        best_result = res

end_time      = time.time()
final_weights = best_result.x

print("\n--- QUANTUM RESOURCE METRICS ---")
print(f"Qubits Used          : {num_qubits}")
print(f"Ansatz Layers        : {num_layers}")
print(f"Ansatz Parameters    : {num_params}")
print(f"Total Iterations     : {len(cost_history)}")
print(f"Total Execution Time : {end_time - start_time:.2f} seconds")
print(f"Final VQLS Cost      : {best_result.fun:.6f}")
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
fig1, (ax_cost, ax_res) = plt.subplots(1, 2, figsize=(14, 5))
ax_cost.plot(cost_history, 'b-', linewidth=2, label='True VQLS Cost $C(\\theta)$')
ax_cost.set_yscale('log')
ax_cost.set_title("True VQLS Cost Convergence (Phase-Preserving)")
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
ax_heat.plot(x_grid, scaled_quantum, 'bo--', linewidth=2, markersize=6,
             label='True VQLS (Phase-Preserving)')
ax_heat.set_title("Heat State $\\phi(x, t)$ (16 Nodes)")
ax_heat.set_xlabel("Space (x)"); ax_heat.set_ylabel("Amplitude")
ax_heat.grid(True); ax_heat.legend()

ax_vel.plot(x_grid, u_classical, 'k-', linewidth=2, label='Classical Exact')
ax_vel.plot(x_grid, u_quantum, 'ro--', linewidth=2, markersize=6,
            label='True VQLS (Phase-Preserving)')
ax_vel.set_title("Burgers' Fluid Velocity $u(x, t)$ (16 Nodes)")
ax_vel.set_xlabel("Space (x)"); ax_vel.set_ylabel("Velocity")
ax_vel.grid(True); ax_vel.legend()
plt.tight_layout()

fig3 = qc.draw('mpl')
fig3.suptitle(f"4-Qubit True VQLS Ansatz ({num_layers} Layers)")
plt.show()