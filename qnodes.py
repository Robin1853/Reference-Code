# qnodes.py – contains all 16 QNodes for hybrid quantum model

import pennylane as qml
import torch

N_QUBITS = 4
DEVICE = qml.device("default.qubit", wires=N_QUBITS)


def make_angle_embedded_qnode(gate_sequence):
    @qml.qnode(DEVICE, interface="torch")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        gate_sequence(weights)
        return qml.expval(qml.PauliZ(0))
    return qnode


# Gate pattern definitions from original script (qnode1 to qnode16)
def gate_pattern_1(weights):
    qml.CNOT(wires=[1, 0])
    qml.RX(weights[1][0], wires=0)
    qml.CNOT(wires=[2, 0])
    qml.RX(weights[1][1], wires=0)
    qml.CNOT(wires=[3, 0])
    qml.RX(weights[1][2], wires=0)

def gate_pattern_2(weights):
    qml.Toffoli(wires=[2, 1, 0])
    qml.RX(weights[1][0], wires=0)
    qml.Toffoli(wires=[3, 2, 0])
    qml.RX(weights[1][1], wires=0)
    qml.Toffoli(wires=[1, 3, 0])
    qml.RX(weights[1][2], wires=0)

def gate_pattern_3(weights):
    qml.MultiControlledX(wires=[3, 2, 1, 0])
    qml.RX(weights[1][0], wires=0)

def gate_pattern_4(weights):
    for i in range(4):
        qml.RX(weights[0][i], wires=i)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])

def gate_pattern_5(weights):
    for i in range(4):
        qml.RX(weights[0][i], wires=i)
    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])

def gate_pattern_6(weights):
    for i in range(4):
        qml.RX(weights[0][i], wires=i)
    qml.MultiControlledX(wires=[3, 2, 1, 0])

def gate_pattern_7(weights):
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])

def gate_pattern_8(weights):
    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])

def gate_pattern_9(weights):
    qml.MultiControlledX(wires=[3, 2, 1, 0])

def gate_pattern_10(weights):
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])
    qml.MultiControlledX(wires=[3, 2, 1, 0])

def gate_pattern_11(weights):
    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])
    qml.MultiControlledX(wires=[3, 2, 1, 0])

def gate_pattern_12(weights):
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])
    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])

def gate_pattern_13(weights):
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])
    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])
    qml.MultiControlledX(wires=[3, 2, 1, 0])

def gate_pattern_14(weights):
    qml.CRZ(weights[0][0], wires=[1, 0])
    qml.PauliX(wires=1)
    qml.CRX(weights[0][1], wires=[1, 0])
    qml.CRZ(weights[0][0], wires=[3, 2])
    qml.PauliX(wires=3)
    qml.CRX(weights[0][1], wires=[3, 2])
    qml.CRZ(weights[0][0], wires=[2, 0])
    qml.PauliX(wires=2)
    qml.CRX(weights[0][1], wires=[2, 0])

def gate_pattern_15(weights):
    for a, b in [(0, 1), (2, 3), (0, 2)]:
        qml.RY(weights[0][0], wires=a)
        qml.RY(weights[0][1], wires=b)
        qml.CNOT(wires=[b, a])
        qml.CRZ(weights[0][2], wires=[b, a])
        qml.PauliX(wires=b)
        qml.CRX(weights[0][3], wires=[b, a])

def gate_pattern_16(weights):
    for a, b in [(0, 1), (2, 3), (0, 2)]:
        qml.Hadamard(wires=a)
        qml.Hadamard(wires=b)
        qml.CZ(wires=[b, a])
        qml.RX(weights[0][0], wires=a)
        qml.RX(weights[0][1], wires=b)
        qml.CRZ(weights[0][2], wires=[b, a])
        qml.PauliX(wires=b)
        qml.CRX(weights[0][3], wires=[b, a])


# Generate all QNodes from their respective gate patterns
qnodes = [
    make_angle_embedded_qnode(fn)
    for fn in [
        gate_pattern_1, gate_pattern_2, gate_pattern_3, gate_pattern_4,
        gate_pattern_5, gate_pattern_6, gate_pattern_7, gate_pattern_8,
        gate_pattern_9, gate_pattern_10, gate_pattern_11, gate_pattern_12,
        gate_pattern_13, gate_pattern_14, gate_pattern_15, gate_pattern_16
    ]
]

# Common weight shape declaration
WEIGHT_SHAPES = {"weights": (2, 4)}
