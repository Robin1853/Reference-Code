import os
import sys
import numpy as np
from math import pi

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

import torchvision
from torchvision import transforms

import pennylane as qml


# === Constants & Configurations ===
N_QUBITS = 4
WEIGHT_SHAPES = {"weights": (2, 4)}
DEVICE = qml.device("default.qubit", wires=N_QUBITS)


# === Data Normalization ===
def normalize_images_to_2pi(tensor):
    """Normalize pixel values from [0, 255] to [0, 2pi]."""
    return tensor / 255 * (2 * pi)


def scale_unit_images_to_2pi(tensor):
    """Scale already normalized images in [0,1] to [0, 2pi]."""
    return tensor * (2 * pi)


# === Data Loading Function ===
def load_dataset(source, batch_size, train_size, data_dir):
    """Load dataset depending on the source identifier."""

    if source == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), scale_unit_images_to_2pi])
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

        train_data = Subset(trainset, range(train_size))
        test_data = Subset(testset, range(train_size // 6))

    elif source == "breastmnist":
        train_images = np.load(os.path.join(data_dir, 'breastmnist', 'train_images.npy'))
        test_images = np.load(os.path.join(data_dir, 'breastmnist', 'test_images.npy'))
        train_labels = np.load(os.path.join(data_dir, 'breastmnist', 'train_labels.npy'))
        test_labels = np.load(os.path.join(data_dir, 'breastmnist', 'test_labels.npy'))

        train_tensor = torch.tensor(normalize_images_to_2pi(train_images), dtype=torch.float32)
        test_tensor = torch.tensor(normalize_images_to_2pi(test_images), dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)

        train_data = Subset(TensorDataset(train_tensor, train_labels), range(train_size))
        test_data = Subset(TensorDataset(test_tensor, test_labels), range(train_size // 6))

    elif source == "5_7_MNIST":
        train_images = np.load(os.path.join(data_dir, '5_7_MNIST_reduced', 'Resizing_trainset_5_7_MNIST.npy'))
        test_images = np.load(os.path.join(data_dir, '5_7_MNIST_reduced', 'Resizing_testset_5_7_MNIST.npy'))
        train_labels = np.load(os.path.join(data_dir, '5_7_MNIST_reduced', 'trainlab_5_7_MNIST.npy'))
        test_labels = np.load(os.path.join(data_dir, '5_7_MNIST_reduced', 'testlab_5_7_MNIST.npy'))

        train_tensor = torch.tensor(scale_unit_images_to_2pi(train_images), dtype=torch.float32)
        test_tensor = torch.tensor(scale_unit_images_to_2pi(test_images), dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)

        train_data = Subset(TensorDataset(train_tensor, train_labels), range(train_size))
        test_data = Subset(TensorDataset(test_tensor, test_labels), range(train_size // 6))

    else:
        sys.exit("Error: Unknown data source")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# === Example QNode (refactor others similarly if needed) ===
@qml.qnode(DEVICE, interface="torch")
def qnode_example(inputs, weights):
    """Example quantum node with angle embedding and RX rotations."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    for i in range(N_QUBITS):
        qml.RX(weights[0][i], wires=i)
    return qml.expval(qml.PauliZ(0))


# === Quantum-Classical Hybrid Model ===
class HybridQuantumModel(nn.Module):
    def __init__(self, qnode_layer):
        super().__init__()
        self.q_layer = qnode_layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 4, 30)  # adjust input shape as needed
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 28, 28).reshape(batch_size, 14, 2, 14, 2)
        x = x.swapaxes(2, 3).reshape(batch_size, 14, 14, 4)  # pseudo 4-channel
        x = self.q_layer(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# === Training Routine ===
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


# === Evaluation Routine ===
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy
