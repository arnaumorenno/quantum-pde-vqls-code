# quantum-pde-vqls-code

# VWLS Quantum Algorithm Implementation — Thesis Code

This repository contains all code written for the thesis:
**"Quantum Computing for Solving PDEs"**
Arnau Moreno Sánchez — Lund University, 2026

## Overview
Nine Python scripts implementing quantum circuits across three simulation and
execution backends for VQLS algorithm, used to produce the results presented in the thesis.

## Repository Structure

| Folder | Description |
|--------|-------------|
| `01_statevector/` | Ideal simulation using Qiskit's Statevector simulator |
| `02_aersimulator_fakefez/` | Noise-aware simulation using AerSimulator with FakeFez backend |
| `03_ibm_hardware/` | Execution on real IBM Quantum hardware |


## Dependencies
- Python 3.10+
- qiskit
- qiskit-aer
- qiskit-ibm-runtime
- numpy
- matplotlib
- scipy

## Usage
Each folder is self-contained.
For IBM hardware scripts, you will need an [IBM Quantum account](https://quantum.ibm.com/)
and API token set as an environment variable.

## Citation
If you use this code, please cite:
Moreno Sánchez, A. "Quantum Computing for Solving PDEs" Lund University, 2026.
