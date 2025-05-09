import pennylane as qml
from matplotlib import pyplot as plt
from pennylane import numpy as np
import scipy
import networkx as nx
import copy
from qiskit.quantum_info import SparsePauliOp
from pennylane.numpy import tensor

qbit_num = 5
qubits = range(qbit_num)

ising_graph = nx.cycle_graph(qbit_num)

print(f"edges: {ising_graph.edges()}")
nx.draw(ising_graph)

np.random.seed(0)
# target_weights = np.round(np.random.uniform(-3, 3, qbit_num), 2)
# target_bias = np.round(np.random.uniform(-3, 3, qbit_num), 2)
# edges = [(0, 1), (0, 3), (1, 2), (2, 3)]

#5 nodes
edges = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]
target_weights = [ 0.29,  1.29,  0.62,  0.27, -0.46]
target_bias = [ 0.88, -0.37,  2.35,  2.78, -0.7 ]

print(f"target_weights: {target_weights}, target_bias: {target_bias}")


def create_hamiltonian_matrix(n_qubits, graph, weights, bias):
    full_matrix = np.zeros((2 ** n_qubits, 2 ** n_qubits))

    # interaction component of Hamiltonian
    for i, edge in enumerate(graph.edges()):
        interaction_term = 1
        for qubit in range(0, n_qubits):
            if qubit in edge:
                interaction_term = np.kron(interaction_term, qml.matrix(qml.PauliZ(0)))
            else:
                interaction_term = np.kron(interaction_term, np.identity(2))
        full_matrix += weights[i] * interaction_term

    # bias component / qubit value
    for i in range(0, n_qubits):
        z_term = x_term = 1
        for j in range(0, n_qubits):
            if j == i:
                z_term = np.kron(z_term, qml.matrix(qml.PauliZ(0)))
                x_term = np.kron(x_term, qml.matrix(qml.PauliX(0)))
            else:
                z_term = np.kron(z_term, np.identity(2))
                x_term = np.kron(x_term, np.identity(2))
        full_matrix += bias[i] * z_term + x_term

    return full_matrix


# visual representation of hamiltonian matrix
ham_matrix = create_hamiltonian_matrix(qbit_num, ising_graph, target_weights, target_bias)

plt.matshow(ham_matrix, cmap="hot")
plt.show()

import jax.numpy as jnp

# low_energy_state = [(0.003904800548899+0.000685609712828j), (0.003754649475738+0.020787128717192j), (0.023000058249516+3.9691117396e-05j), (0.003409089780411+0.000913817002457j), (0.005658954807791+0.005139979055575j), (9.5703984317e-05+0.156279006497625j), (0.017769540727955+0.000187802023309j), (0.001108926402511+0.004022922207083j), (0.008811301094507+0.002807586168434j), (0.000616009245371+0.085123737611367j), (0.00323017616986+0.000162533228719j), (0.000251926289955+0.00374203257311j), (0.000790502036861+0.021048215906361j), (0.004024981452926+0.639962573564229j), (0.034889453049718+0.000769052016861j), (0.005516232677973+0.016473933355782j), (0.075500253853262+2.273774125e-05j), (0.037813798765291+0.00068938977675j), (0.096318814026157+1.316337851e-06j), (0.025665581359447+3.0306360508e-05j), (0.035836828618822+0.000170474042849j), (0.01356230936014+0.00518319509998j), (0.354538460120631+6.228687908e-06j), (0.037343848977037+0.000133425190175j), (0.019280856446433+0.000129853121932j), (0.002375315786447+0.00393704142008j), (0.099420033857694+7.517297368e-06j), (0.005319452222987+0.000173072136232j), (0.014021948026046+0.000973498141092j), (0.008934481985864+0.029598821051104j), (0.050156581480169+3.5569314855e-05j), (0.007079129119267+0.000761933521738j)]
# low_energy_state = [0.01584237 + 0.02705964j, -0.12259349 - 0.20709744j,
#                     -0.00984184 - 0.01675181j, 0.0678769 + 0.11457059j,
#                     -0.05762965 - 0.09834919j, 0.44166771 + 0.74604064j,
#                     0.02455546 + 0.03851193j, -0.16173527 - 0.27367131j,
#                     -0.00380591 - 0.00647135j, 0.0294841 + 0.04960178j,
#                     0.00244558 + 0.00425272j, -0.01696184 - 0.02899256j,
#                     0.0139341 + 0.0237856j, -0.10678024 - 0.18040841j,
#                     -0.0059144 - 0.00924647j, 0.03892166 + 0.06575019j]

#5nodes
low_energy_state = [ 6.90292276e-03+1.71086894e-03j,
             -1.97214907e-03-2.83671046e-04j,
             -3.38768632e-02-7.53339565e-03j,
              1.05775455e-02+2.40123455e-03j,
             -4.03849802e-02-9.44751282e-03j,
              1.16493456e-02+1.46476443e-03j,
              1.80519145e-01+4.02585195e-02j,
             -5.63878554e-02-1.28450441e-02j,
             -2.34693853e-03-2.25140261e-04j,
              6.49734853e-04-1.48566762e-05j,
              1.22749455e-02+2.80281869e-03j,
             -3.83282010e-03-8.89646101e-04j,
              1.45421261e-02+3.46841114e-03j,
             -4.19858648e-03-5.48216953e-04j,
             -6.48714001e-02-1.44523105e-02j,
              2.02635300e-02+4.61192439e-03j,
             -3.13997907e-02-7.73337591e-03j,
              8.96810368e-03+1.27504887e-03j,
              1.54187405e-01+3.42980160e-02j,
             -4.81427569e-02-1.09318296e-02j,
              1.83796582e-01+4.30058177e-02j,
             -5.30179841e-02-6.66916207e-03j,
             -8.21547368e-01-1.83215443e-01j,
              2.56622605e-01+5.84575473e-02j,
              1.07246179e-02+1.17166348e-03j,
             -2.97706121e-03+2.32431524e-05j,
             -5.58289804e-02-1.27189013e-02j,
              1.74323488e-02+4.03847010e-03j,
             -6.61734720e-02-1.57575517e-02j,
              1.91041337e-02+2.48671622e-03j,
              2.95238356e-01+6.57803649e-02j,
             -9.22220333e-02-2.09911219e-02j]

# Assuming you have 'data_np' as your numpy array
low_tensor = jnp.array(low_energy_state)

res = np.vdot(low_energy_state, (ham_matrix @ low_energy_state))
energy_exp = np.real(res)
print(f"Energy Expectation: {energy_exp}")

ground_state_energy = np.real_if_close(min(np.linalg.eig(ham_matrix)[0]))
print(f"Ground State Energy: {ground_state_energy}")


def state_evolve(hamiltonian, qubits, time):
    U = scipy.linalg.expm(-1j * hamiltonian * time)
    qml.QubitUnitary(U, wires=qubits)


def qgrnn_layer(weights, bias, qubits, graph, trotter_step):
    # Applies a layer of RZZ gates (based on a graph)
    for i, edge in enumerate(graph.edges):
        qml.MultiRZ(2 * weights[i] * trotter_step, wires=(edge[0], edge[1]))

    # Applies a layer of RZ gates
    for i, qubit in enumerate(qubits):
        qml.RZ(2 * bias[i] * trotter_step, wires=qubit)

    # Applies a layer of RX gates
    for qubit in qubits:
        qml.RX(2 * trotter_step, wires=qubit)


def swap_test(control, register1, register2):
    qml.Hadamard(wires=control)
    for reg1_qubit, reg2_qubit in zip(register1, register2):
        qml.CSWAP(wires=(control, reg1_qubit, reg2_qubit))
    qml.Hadamard(wires=control)


# Defines some fixed values

reg1 = tuple(range(qbit_num))  # First qubit register
reg2 = tuple(range(qbit_num, 2 * qbit_num))  # Second qubit register

control = 2 * qbit_num  # Index of control qubit
trotter_step = 0.01  # Trotter step size

# Defines the interaction graph for the new qubit system

new_ising_graph = nx.complete_graph(reg2)

print(f"Edges: {new_ising_graph.edges}")
nx.draw(new_ising_graph)
plt.show()


def qgrnn(weights, bias, time=None):
    # Prepares the low energy state in the two registers
    qml.StatePrep(np.kron(low_energy_state, low_energy_state), wires=reg1 + reg2)

    # Evolve the first qubit register with the time evolution circuit to prepare a piece of quantum data
    state_evolve(ham_matrix, reg1, time)

    # Applies the QGRNN Layers to the second qubit register
    depth = time / trotter_step  # P = t/delta
    for _ in range(0, int(depth)):
        qgrnn_layer(weights, bias, reg2, new_ising_graph, trotter_step)

        # Applies the SWAP test between the registers
        swap_test(control, reg1, reg2)

    # Returns the result of the swap test
    return qml.expval(qml.PauliZ(control))


N = 15  # Number of pieces of quantum data used for each step
max_time = 0.1  # Maximum value of data that can be used for quantum data

rng = np.random.default_rng(seed=42)


def cost_function(weight_params, bias_params):
    # randomly samples times at which the qgrnn
    times_sampled = rng.random(size=N) * max_time

    # cycles through each of the sampled times and calculates the cost
    total_cost = 0
    for dt in times_sampled:
        result = qgrnn_qnode(weight_params, bias_params, time=dt)
        # print("cost_function result:", result)
        total_cost += -1 * result

        return total_cost / N


# defines the new device
qgrnn_dev = qml.device('default.qubit', wires=2 * qbit_num + 1)

# Defines the new QNode
qgrnn_qnode = qml.QNode(qgrnn, qgrnn_dev)

steps = 300

optimizer = qml.AdamOptimizer(stepsize=0.5)

weights = rng.random(size=len(new_ising_graph.edges), requires_grad=True) - 0.5
bias = rng.random(size=qbit_num, requires_grad=True) - 0.5

# weights = rng.random(size=len(new_ising_graph.edges)) - 0.5
# bias = rng.random(size=qbit_num) - 0.5

initial_weights = copy.copy(weights)
initial_bias = copy.copy(bias)

for i in range(0, steps):
    (weights, bias), cost = optimizer.step_and_cost(cost_function, weights, bias)
    # print("weights bias and cost:", weights, bias, cost)

    # Prints the value of the cost function
    if i % 5 == 0:
        print(f"Cost and step {i}: {cost}")
        print(f"Weights at step {i}: {weights}")
        print(f"Bias at step {i}: {bias}")
        print("----------------------------------------")


new_ham_matrix = create_hamiltonian_matrix(
    qbit_num, nx.complete_graph(qbit_num), weights, bias
)

init_ham = create_hamiltonian_matrix(
    qbit_num, nx.complete_graph(qbit_num), initial_weights, initial_bias
)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 6))

axes[0].matshow(ham_matrix, vmin=-7, vmax=7, cmap="hot")
axes[0].set_title("Target", y=1.13)

axes[1].matshow(init_ham, vmin=-7, vmax=7, cmap="hot")
axes[1].set_title("Initial", y=1.13)

axes[2].matshow(new_ham_matrix, vmin=-7, vmax=7, cmap="hot")
axes[2].set_title("Learned", y=1.13)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()

# We first pick out the weights of edges (1, 3) and (2, 0)
# and then remove them from the list of target parameters

weights_node = []
weights_edge = []

for ii, edge in enumerate(new_ising_graph.edges):
    if (edge[0] - qbit_num, edge[1] - qbit_num) in ising_graph.edges:
        weights_edge.append(weights[ii])
    else:
        weights_node.append(weights[ii])

print("Target parameters         Learned parameters")
print("Weights")
print("-" * 41)
for ii_target, ii_learned in zip(target_weights, weights_edge):
    print(f"{ii_target : <20}|{ii_learned : >20}")

print("\nBias")
print("-"*41)
for ii_target, ii_learned in zip(target_bias, bias):
    print(f"{ii_target : <20}|{ii_learned : >20}")

print(f"\nNon-Existing Edge Parameters: {[val.unwrap() for val in weights_node]}")