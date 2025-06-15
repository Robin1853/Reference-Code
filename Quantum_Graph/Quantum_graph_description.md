# Quantum Graph Learning and Simulation

This project combines two quantum approaches to characterize and learn properties of Ising-like Hamiltonians defined on graphs.

## VQTE

**Variational Quantum Thermalization via Imaginary Time Evolution** (VQTE) uses Qiskit’s `VarQITE` and `SciPyImaginaryEvolver` to approximate ground states and compare them to exact imaginary time evolution.  
A custom Hamiltonian is constructed based on graph topology, weights, and biases. Auxiliary observables and energy expectation values are tracked and visualized throughout the evolution.

## QuantumGraph

Implements a **Quantum Graph Recurrent Neural Network** (QGRNN) in PennyLane to reconstruct the structure of an unknown Hamiltonian.  
Using a SWAP test between evolved and reference states, the QGRNN learns to approximate the original Hamiltonian via gradient descent.  
Matrix visualizations and parameter comparisons illustrate the learning performance.

---

The overall workflow demonstrates **forward and inverse problems** in graph-based quantum systems:  
simulation via VQTE and structure reconstruction via a QGRNN model.
