
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_aer import AerSimulator

# ── Noisy backend ─────────────────────────────────────────────
fake_backend = AerSimulator.from_backend(FakeFez())
SHOTS        = 8000

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

# --- 2. BUILD HARDWARE-COMPATIBLE OBSERVABLES ---
A_op         = SparsePauliOp(['II', 'XI', 'XX', 'YY'], [1.5, -0.2, -0.1, -0.1])
A_squared_op = A_op.compose(A_op).simplify()

b_outer  = np.outer(b_norm, b_norm)
O_matrix = A @ b_outer @ A
O_op     = SparsePauliOp.from_operator(O_matrix).chop(1e-12)

print(f"\nNumerator O Pauli terms   : {len(O_op)}")
print(f"Denominator A² Pauli terms: {len(A_squared_op)}")

# --- 3. ESTIMATOR ---
from qiskit_ibm_runtime import EstimatorOptions
options                = EstimatorOptions()
options.default_shots  = SHOTS
estimator              = EstimatorV2(mode=fake_backend, options=options)

# --- 4. CIRCUIT ---
weights = ParameterVector('w', 4)
qc = QuantumCircuit(2)
qc.ry(weights[0], 0)
qc.ry(weights[1], 1)
qc.cx(0, 1)
qc.ry(weights[2], 0)
qc.ry(weights[3], 1)
qc.cx(0, 1)

# ── Transpile and map observables to device layout ────────────
pm        = generate_preset_pass_manager(backend=fake_backend, optimization_level=1)
isa_qc    = pm.run(qc)
O_op_isa  = O_op.apply_layout(isa_qc.layout)
A2_op_isa = A_squared_op.apply_layout(isa_qc.layout)

# --- 5. COST FUNCTION ---
def vqls_cost(w_vals):
    job    = estimator.run([(isa_qc, O_op_isa,  w_vals),
                            (isa_qc, A2_op_isa, w_vals)])
    result = job.result()
    numer  = float(result[0].data.evs)
    denom  = float(result[1].data.evs)
    return float(1.0 - numer / (denom + 1e-8))

# --- 6. EXECUTION ---
np.random.seed(1337)
initial_weights        = np.random.randn(4) * 0.1
cost_history, residual_history = [], []

def objective_fn(w):
    cost = vqls_cost(w)
    cost_history.append(cost)
    residual_history.append(np.sqrt(max(cost, 0)))
    return cost

print("\nOptimizing 2Q VQLS — AerSimulator (FakeFez noise model)")
print(f"Shots per evaluation: {SHOTS}\n")
start_time = time.time()

result = minimize(objective_fn, initial_weights,
                  method='COBYLA',
                  options={'maxiter': 1000, 'disp': True, 'rhobeg': 0.1, 'catol': 1e-3})

end_time      = time.time()
final_weights = result.x

print("\n--- QUANTUM RESOURCE METRICS ---")
print(f"Qubits Used          : 2")
print(f"Ansatz Layers        : 2")
print(f"Ansatz Parameters    : 4")
print(f"Simulation Backend   : AerSimulator (FakeFez noise model)")
print(f"Shots per Evaluation : {SHOTS}")
print(f"Total Iterations     : {len(cost_history)}")
print(f"Total Execution Time : {end_time - start_time:.2f} seconds")
print(f"Final VQLS Cost      : {result.fun:.6f}")
print(f"\nfinal_weights = np.array({list(np.round(final_weights, 8))})")

# --- 7. EXTRACT SOLUTION ---
param_dict       = dict(zip(weights, final_weights))
psi_final        = Statevector(qc.assign_parameters(param_dict)).data
quantum_solution = np.real(psi_final)
scalar           = np.dot(quantum_solution, classical_solution) / \
                   (np.dot(quantum_solution, quantum_solution) + 1e-12)
scaled_quantum   = quantum_solution * scalar
state_error      = np.abs(classical_solution - scaled_quantum)

print("\n--- HEAT STATE RESULTS ---")
print(f"Classical Exact Solution : {np.round(classical_solution, 4)}")
print(f"VQLS Solution            : {np.round(scaled_quantum, 4)}")
print(f"Mean Absolute Error      : {np.mean(state_error):.4f}")

# --- 8. COLE-HOPF DECODE ---
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

# --- 9. PLOTS ---
print("\nGenerating Plots...")

fig1, (ax_cost, ax_res) = plt.subplots(1, 2, figsize=(14, 5))
ax_cost.plot(cost_history, 'b-', linewidth=2, label='VQLS Cost $C(\\theta)$')
ax_cost.set_yscale('log')
ax_cost.set_title("VQLS Convergence — AerSimulator FakeFez (2 Qubits)")
ax_cost.set_xlabel('Iteration'); ax_cost.set_ylabel('Cost')
ax_cost.grid(True, which="both", ls="--"); ax_cost.legend()

ax_res.plot(residual_history, 'r-', linewidth=2, label='$\\sqrt{C(\\theta)}$')
ax_res.set_yscale('log')
ax_res.set_title("Cost Proxy (Residual Estimate)")
ax_res.set_xlabel('Iteration'); ax_res.set_ylabel('Value')
ax_res.grid(True, which="both", ls="--"); ax_res.legend()
plt.tight_layout()

fig2, (ax_heat, ax_vel) = plt.subplots(1, 2, figsize=(14, 5))
ax_heat.plot(x_grid, classical_solution, 'k-', linewidth=2, label='Classical Exact')
ax_heat.plot(x_grid, scaled_quantum, 'bo--', linewidth=2, markersize=8,
             label='AerSimulator VQLS (FakeFez)')
ax_heat.set_title("Heat State $\\phi(x, t)$")
ax_heat.set_xlabel("Space (x)"); ax_heat.set_ylabel("Amplitude")
ax_heat.grid(True); ax_heat.legend()

ax_vel.plot(x_grid, u_classical, 'k-', linewidth=2, label='Classical Exact')
ax_vel.plot(x_grid, u_quantum, 'ro--', linewidth=2, markersize=8,
            label='AerSimulator VQLS (FakeFez)')
ax_vel.set_title("Burgers' Fluid Velocity $u(x, t)$")
ax_vel.set_xlabel("Space (x)"); ax_vel.set_ylabel("Velocity")
ax_vel.grid(True); ax_vel.legend()
plt.tight_layout()
plt.show()

param_dict = dict(zip(weights, np.round(final_weights, 3)))
qc_bound   = qc.assign_parameters(param_dict)
fig3       = qc_bound.draw('mpl', style='clifford')
plt.tight_layout()
plt.show()

# --- 10. NOISE ROBUSTNESS ---
print("\n" + "="*68)
print(" STARTING NOISE ROBUSTNESS ANALYSIS")
print("="*68)
print(f"{'Noise Variance (%)':<20} | {'Heat State Error (%)':<22} | {'Velocity Error (%)':<20}")
print("-" * 68)

noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

for noise_var in noise_levels:
    noise        = np.random.normal(0, noise_var * np.mean(b_classical), size=b_classical.shape)
    b_noisy_norm = (b_classical + noise)
    b_noisy_norm = b_noisy_norm / np.linalg.norm(b_noisy_norm)
    O_noisy      = SparsePauliOp.from_operator(A @ np.outer(b_noisy_norm, b_noisy_norm) @ A).chop(1e-12)

    def noisy_objective_fn(w):
        O_noisy_isa = O_noisy.apply_layout(isa_qc.layout)
        job    = estimator.run([(isa_qc, O_noisy_isa, w),
                                (isa_qc, A2_op_isa,   w)])
        result = job.result()
        numer  = float(result[0].data.evs)
        denom  = float(result[1].data.evs)
        return float(1.0 - numer / (denom + 1e-8))

    res_noisy = minimize(noisy_objective_fn,
                         np.random.randn(4) * 0.1,
                         method='COBYLA',
                         options={'maxiter': 500, 'disp': False})

    param_dict_n = dict(zip(weights, res_noisy.x))
    psi_noisy    = Statevector(qc.assign_parameters(param_dict_n)).data
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