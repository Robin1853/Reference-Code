from qiskit.quantum_info import SparsePauliOp, Operator, Pauli
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

node_num = 5

ising_graph = nx.cycle_graph(node_num)

np.random.seed(0)
target_weights = np.round(np.random.uniform(-3, 3, node_num), 2)
target_bias = np.round(np.random.uniform(-3, 3, node_num), 2)
edges = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]

print(target_weights)
print(target_bias)
print(edges)
###5nodes
# edges = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]
# target_weights = [ 0.29,  1.29,  0.62, -0.46, ]
# target_bias = [ 0.88, -0.37,  2.35,  2.78]

# from qiskit.quantum_info import SparsePauliOp
#
# def create_hamiltonian_list(n_qubits, graph, weights, bias):
#     sparse_list = []
#
#     # interaction component of Hamiltonian
#     for i, edge in enumerate(graph.edges):
#         sparse_list.append(("Z"*n_qubits, list(edge), weights[i]))
#
#     # bias component / qubit value
#     for i in range(n_qubits):
#         sparse_list.append(("Z", [i], bias[i]))
#         sparse_list.append(("X", [i], bias[i]))
#
#     return sparse_list

def create_hamiltonian_matrix(n_qubits, graph, weights, bias):
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    id_matrix = np.identity(2)

    full_matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)

    # interaction component of Hamiltonian
    for i, edge in enumerate(graph.edges()):
        interaction_term = 1
        for qubit in range(n_qubits):
            if qubit in edge:
                interaction_term = np.kron(interaction_term, pauli_z)
            else:
                interaction_term = np.kron(interaction_term, id_matrix)
        full_matrix += weights[i] * interaction_term

    # bias component
    for i in range(n_qubits):
        z_term = x_term = 1
        for j in range(n_qubits):
            if j == i:
                z_term = np.kron(z_term, pauli_z)
                x_term = np.kron(x_term, pauli_x)
            else:
                z_term = np.kron(z_term, id_matrix)
                x_term = np.kron(x_term, id_matrix)
        full_matrix += bias[i] * z_term + x_term

    return Operator(full_matrix)

qbit_num = len(target_bias)
sparse_list = create_hamiltonian_matrix(qbit_num, ising_graph, target_weights, target_bias)

hamiltonian = SparsePauliOp.from_operator(sparse_list)
print(hamiltonian)
magnetization = SparsePauliOp(["IIIZ", "IIZI", "IZII", "ZIII"], coeffs=[1, 1, 1, 1])

# hamiltonian = SparsePauliOp(["ZZ", "IX", "XI"], coeffs=[-0.2, -1, -1])
# magnetization = SparsePauliOp(["IZ", "ZI"], coeffs=[1, 1])

from qiskit.circuit.library import EfficientSU2

ansatz = EfficientSU2(hamiltonian.num_qubits, reps=1)
ansatz.decompose().draw("mpl")

plt.show()

import numpy as np

init_param_values = {}
for i in range(len(ansatz.parameters)):
    init_param_values[ansatz.parameters[i]] = np.pi / 2

from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit_algorithms import TimeEvolutionProblem
from qiskit_algorithms import VarQITE
from qiskit.primitives import Estimator
import pickle
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ReverseQGT
from qiskit.quantum_info import Statevector
from qiskit_algorithms import SciPyImaginaryEvolver

time = 5.0
aux_ops = [hamiltonian]

###strict forward
# var_principle = ImaginaryMcLachlanPrinciple()
# evolution_problem = TimeEvolutionProblem(hamiltonian, time, aux_operators=aux_ops)
# var_qite = VarQITE(ansatz, init_param_values, var_principle, Estimator())
# # an Estimator instance is necessary, if we want to calculate the expectation value of auxiliary operators.
# evolution_result = var_qite.evolve(evolution_problem)

###improved gradient handling
var_principle_eff = ImaginaryMcLachlanPrinciple(qgt=ReverseQGT(), gradient=ReverseEstimatorGradient())
evolution_problem_eff = TimeEvolutionProblem(hamiltonian, time, aux_operators=aux_ops)
var_qite_eff = VarQITE(ansatz, init_param_values, var_principle_eff, Estimator())
evolution_result = var_qite_eff.evolve(evolution_problem_eff)

###long processing time save result and continue
with open('evolution_result4.pickle', 'wb') as f:
    pickle.dump(evolution_result, f)
###load acquired result
with open('evolution_result4.pickle', 'rb') as f:
    evolution_result = pickle.load(f)
#print(evolution_result.observables[0])

init_state = Statevector(ansatz.assign_parameters(init_param_values))
evolution_problem = TimeEvolutionProblem(
    hamiltonian, time, initial_state=init_state, aux_operators=aux_ops
)
exact_evol = SciPyImaginaryEvolver(num_timesteps=501)
sol = exact_evol.evolve(evolution_problem)

h_exp_val = np.array([ele[0][0] for ele in evolution_result.observables])

exact_h_exp_val = sol.observables[0][0].real

times = evolution_result.times

plt.plot(times, h_exp_val, label="VarQITE")
plt.plot(times, exact_h_exp_val, label="Exact", linestyle="--")
plt.xlabel("Time")
plt.ylabel(r"$\langle H \rangle$ (energy)")
plt.legend(loc="upper right")
plt.show()

#print(evolution_result)
print("Ground state energy", h_exp_val[-1])

from qiskit_aer import Aer
from qiskit import transpile
from qiskit import QuantumCircuit, ClassicalRegister
import jax.numpy as jnp

optimized_params = evolution_result.parameter_values[-1]
ground_state = ansatz.assign_parameters(optimized_params)
# Now, ground_state is a QuantumCircuit representing the ground state
backend = Aer.get_backend('statevector_simulator')  # replace with your quantum backend
transp_circuit = transpile(ground_state, backend)
job = backend.run(transp_circuit)
psi = job.result().get_statevector()
print(psi)
psi = psi.data
psi = jnp.array(psi)
hamiltonian_matrix = hamiltonian.to_matrix()
#final_energy = np.real(hamiltonian_matrix.expectation_value(psi))
final_energy = np.vdot(psi, (hamiltonian_matrix @ psi))
plt.matshow(np.real(hamiltonian_matrix), cmap="hot")
plt.show()
print(f"Energy Expectation: {final_energy}")

# from qiskit_algorithms.time_evolvers.variational import RealMcLachlanPrinciple
#
# aux_ops = [magnetization]
#
# from qiskit_algorithms import VarQRTE
# from qiskit_algorithms.gradients import DerivativeType
#
# time = 10.0
# # var_principle_re = RealMcLachlanPrinciple(qgt=ReverseQGT(), gradient=ReverseEstimatorGradient(derivative_type=DerivativeType.IMAG))
# # evolution_problem_re = TimeEvolutionProblem(hamiltonian, time, aux_operators=aux_ops)
# # var_qrte = VarQRTE(ansatz, init_param_values, var_principle_re, Estimator())
# # evolution_result_re = var_qrte.evolve(evolution_problem_re)
#
# ###long processing time save result and continue
# # with open('evolution_result_re4.pickle', 'wb') as f:
# #     pickle.dump(evolution_result_re, f)
# # ###load acquired result
# with open('evolution_result_re4.pickle', 'rb') as f:
#     evolution_result_re = pickle.load(f)
#
# init_circ = ansatz.assign_parameters(init_param_values)
#
# from qiskit_algorithms import SciPyRealEvolver
#
# evolution_problem = TimeEvolutionProblem(
#     hamiltonian, time, initial_state=init_circ, aux_operators=aux_ops
# )
# rtev = SciPyRealEvolver(1001)
# sol = rtev.evolve(evolution_problem)
#
# optimized_params = evolution_result_re.parameter_values[-1]
# ground_state = ansatz.assign_parameters(optimized_params)
# measurements = ClassicalRegister(ground_state.num_qubits, 'measure')
# ground_state.add_register(measurements)
# ground_state.measure(ground_state.qubits, ground_state.clbits)
# # Now, ground_state is a QuantumCircuit representing the ground state
# backend = Aer.get_backend('statevector_simulator')  # replace with your quantum backend
# transp_circuit = transpile(ground_state, backend)
# job = backend.run(transp_circuit)
# psi = job.result().get_statevector()
# print(psi)
#
# mz_exp_val_re = np.array([ele[0][0] for ele in evolution_result_re.observables])
# exact_mz_exp_val_re = sol.observables[0][0].real
# times = evolution_result_re.times
# plt.plot(times, mz_exp_val_re, label="VarQRTE")
# plt.plot(times, exact_mz_exp_val_re, label="Exact", linestyle="--")
# plt.xlabel("Time")
# plt.ylabel(r"$\langle m_z \rangle$")
# plt.legend(loc="upper right")
# plt.show()