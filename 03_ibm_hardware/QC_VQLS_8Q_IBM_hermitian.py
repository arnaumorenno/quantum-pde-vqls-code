
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, EstimatorOptions

# ============================================================
# IBM QUANTUM CREDENTIALS
# ============================================================
IBM_API_TOKEN = "your_API_token_here"  # <-- REPLACE WITH YOUR IBM QUANTUM API TOKEN
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=IBM_API_TOKEN,
    overwrite=True
)

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

# --- 2. BUILD OBSERVABLES ---
print("\nBuilding hardware-compatible observables...")

A_op         = SparsePauliOp.from_operator(A).chop(1e-10)
A_squared_op = A_op.compose(A_op).simplify().chop(1e-10)

b_outer  = np.outer(b_norm, b_norm)
O_matrix = A @ b_outer @ A
O_op     = SparsePauliOp.from_operator(O_matrix).chop(1e-12)

print(f"Numerator O Pauli terms   : {len(O_op)}")
print(f"Denominator A² Pauli terms: {len(A_squared_op)}")
print(f"Total Pauli circuits      : {len(O_op) + len(A_squared_op)}")

# --- 3. CIRCUIT ---
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
        qc.cx(i, i + 1)

# ============================================================
# PHASE 1: LOCAL OPTIMIZATION (StatevectorEstimator — fast)
# ============================================================
print(f"\nPhase 1: Local optimization ({num_params} parameters, {num_layers} layers)...")

local_estimator = StatevectorEstimator(default_precision=0.0, seed=2026)

def vqls_cost_local(w_vals):
    job    = local_estimator.run([(qc, O_op,         w_vals),
                                  (qc, A_squared_op, w_vals)])
    result = job.result()
    numer  = float(result[0].data.evs)
    denom  = float(result[1].data.evs)
    return float(1.0 - numer / (denom + 1e-8))

np.random.seed(2026)
initial_weights        = np.random.randn(num_params) * 0.1
cost_history, residual_history = [], []

def objective_fn(w):
    cost = vqls_cost_local(w)
    cost_history.append(cost)
    residual_history.append(np.sqrt(max(cost, 0)))
    return cost

start_time = time.time()
result_opt = minimize(objective_fn, initial_weights,
                      method='COBYLA',
                      options={'maxiter': 6000, 'disp': True,
                               'rhobeg': 0.5, 'catol': 1e-3})
end_local     = time.time()
final_weights = result_opt.x

print(f"\nLocal optimization complete in {end_local - start_time:.2f}s")
print(f"Final local cost : {result_opt.fun:.6f}")
print(f"Total iterations : {len(cost_history)}")
print(f"\nfinal_weights = np.array({list(np.round(final_weights, 8))})")

# ============================================================
# PHASE 2: FINAL EVALUATION ON REAL IBM HARDWARE (2 jobs)
# ============================================================
print("\nPhase 2: Final evaluation on real IBM hardware...")

service      = QiskitRuntimeService(channel="ibm_quantum_platform")
real_backend = service.least_busy(operational=True, simulator=False)
print(f"Running on: {real_backend.name}")

print(f"\nNumerator O Pauli terms   : {len(O_op)}")
print(f"Denominator A² Pauli terms: {len(A_squared_op)}")
print(f"Total Pauli circuits      : {len(O_op) + len(A_squared_op)}")
print(f"Shots                     : 4000")
print(f"Estimated QPU time        : ~{(len(O_op) + len(A_squared_op)) * 4000 / (12 * 8000) * 24:.0f} seconds")

pm        = generate_preset_pass_manager(backend=real_backend, optimization_level=1)
isa_qc    = pm.run(qc)
O_op_isa  = O_op.apply_layout(isa_qc.layout)
A2_op_isa = A_squared_op.apply_layout(isa_qc.layout)

options               = EstimatorOptions()
options.default_shots = 4000
real_estimator        = EstimatorV2(mode=real_backend, options=options)

print("\nSubmitting 2 jobs to IBM hardware...")
ibm_start = time.time()

job       = real_estimator.run([(isa_qc, O_op_isa,  final_weights),
                                 (isa_qc, A2_op_isa, final_weights)])
ibm_result    = job.result()
numer_hw      = float(ibm_result[0].data.evs)
denom_hw      = float(ibm_result[1].data.evs)
final_cost_hw = float(1.0 - numer_hw / (denom_hw + 1e-8))

ibm_end = time.time()
print(f"IBM hardware evaluation complete in {ibm_end - ibm_start:.2f}s")

# --- EXTRACT SOLUTION ---
param_dict       = dict(zip(weights, final_weights))
psi_final        = Statevector(qc.assign_parameters(param_dict)).data
quantum_solution = np.real(psi_final)
scalar           = np.dot(quantum_solution, classical_solution) / \
                   (np.dot(quantum_solution, quantum_solution) + 1e-12)
scaled_quantum   = quantum_solution * scalar
state_error      = np.abs(classical_solution - scaled_quantum)

print("\n--- QUANTUM RESOURCE METRICS ---")
print(f"Qubits Used             : {num_qubits}")
print(f"Ansatz Layers           : {num_layers}")
print(f"Ansatz Parameters       : {num_params}")
print(f"IBM Backend             : {real_backend.name}")
print(f"Shots (final eval)      : 4000")
print(f"IBM Jobs Submitted      : 2")
print(f"Local Opt. Time         : {end_local - start_time:.2f} seconds")
print(f"IBM Eval. Time          : {ibm_end - ibm_start:.2f} seconds")
print(f"Local Opt. Cost         : {result_opt.fun:.6f}")
print(f"IBM Hardware Cost       : {final_cost_hw:.6f}")

print("\n--- HEAT STATE RESULTS ---")
print(f"Mean Absolute Error : {np.mean(state_error):.4f}")

# --- COLE-HOPF DECODE ---
nu = 0.1
dx = x_grid[1] - x_grid[0]

d_phi_dx_quantum   = np.gradient(scaled_quantum, dx)
d_phi_dx_classical = np.gradient(classical_solution, dx)
u_quantum   = -2 * nu * (d_phi_dx_quantum   / scaled_quantum)
u_classical = -2 * nu * (d_phi_dx_classical / classical_solution)

print("\n--- FINAL BURGERS' VELOCITY (u) ---")
print(f"Mean Absolute Error: {np.mean(np.abs(u_classical - u_quantum)):.4f}")

# --- PLOTS ---
print("\nGenerating Plots...")

fig1, (ax_cost, ax_res) = plt.subplots(1, 2, figsize=(14, 5))
ax_cost.plot(cost_history, 'b-', linewidth=2, label='VQLS Cost $C(\\theta)$')
ax_cost.set_yscale('log')
ax_cost.set_title(f"Local Optimization Convergence ({num_qubits} Qubits)")
ax_cost.set_xlabel('Iteration'); ax_cost.set_ylabel('Cost')
ax_cost.grid(True, which="both", ls="--"); ax_cost.legend()

ax_res.plot(residual_history, 'r-', linewidth=2, label='$\\sqrt{C(\\theta)}$')
ax_res.set_yscale('log')
ax_res.set_title("Cost Proxy (Local Optimization)")
ax_res.set_xlabel('Iteration'); ax_res.set_ylabel('Value')
ax_res.grid(True, which="both", ls="--"); ax_res.legend()
plt.tight_layout()

fig2, (ax_heat, ax_vel) = plt.subplots(1, 2, figsize=(14, 5))
ax_heat.plot(x_grid, classical_solution, 'k-', linewidth=2, label='Classical Exact')
ax_heat.plot(x_grid, scaled_quantum, 'b-', linewidth=2, alpha=0.8,
             label=f'IBM HW VQLS ({real_backend.name})')
ax_heat.set_title(f"Heat State $\\phi(x, t)$ ({num_nodes} Nodes)")
ax_heat.set_xlabel("Space (x)"); ax_heat.set_ylabel("Amplitude")
ax_heat.grid(True); ax_heat.legend()

ax_vel.plot(x_grid, u_classical, 'k-', linewidth=2, label='Classical Exact')
ax_vel.plot(x_grid, u_quantum, 'r-', linewidth=2, alpha=0.8,
            label=f'IBM HW VQLS ({real_backend.name})')
ax_vel.set_title(f"Burgers' Fluid Velocity $u(x, t)$ ({num_nodes} Nodes)")
ax_vel.set_xlabel("Space (x)"); ax_vel.set_ylabel("Velocity")
ax_vel.set_ylim([-2, 2])
ax_vel.grid(True); ax_vel.legend()
plt.tight_layout()
plt.show()

param_dict = dict(zip(weights, np.round(final_weights, 3)))
qc_bound   = qc.assign_parameters(param_dict)
fig3       = qc_bound.draw('mpl', style='clifford', fold=-1)
plt.tight_layout()
plt.show()