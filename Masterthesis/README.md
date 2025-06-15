# Quantum Neural Network with Pooling – Master’s Thesis Code

This repository contains the core implementation of my master's thesis. The project explores the classification performance of hybrid quantum-classical neural networks using a diverse set of quantum circuit architectures.

## Overview

The main module `QNN_1Layer_Pooling.py` contains the full training pipeline, which includes:

- Data loading and preprocessing for multiple datasets
- Definition and execution of 16 different quantum circuits
- Integration of quantum circuits as layers within a PyTorch neural network
- Training and evaluation over multiple seeds and epochs
- Logging and visualization of loss and accuracy metrics

The quantum circuits are implemented using **PennyLane** and integrated into the model via `qml.qnn.TorchLayer`.

Each circuit is evaluated:
- across **10 random seeds** (for parameter initialization),
- over **10 training epochs**,
- on different datasets (see below),
- using consistent metrics for comparison (loss, accuracy, generalization gap).

## Quantum Architecture

A total of **16 distinct quantum circuits** are defined, including variations with:
- Entangling layers (CNOT, Toffoli, MultiControlledX)
- Rotational gates (RX, RY, CRZ, CRX)
- Hadamard and controlled-Z layers
- Shallow and deeper variational forms

These circuits are embedded in a simple feedforward architecture and trained using the Adam optimizer with cross-entropy loss.

## Datasets

The code supports several datasets. The training set sizes can be adjusted via parameters. Main datasets used:

- **MNIST** – [LeCun et al.](http://yann.lecun.com/exdb/mnist) – handwritten digit classification  
  - Default: 10,000 training samples  
- **BreastMNIST** – [MedMNIST](https://medmnist.com/) – binary classification of ultrasound breast images  
  - Default: 546 training samples  
- **5/7 MNIST Subset** – Custom subset of MNIST containing only digits 5 and 7  
  - Default: 400 training samples  

Each dataset is normalized and reshaped to suit the 4-qubit quantum circuit input size.

## Code Structure

For better structure and maintainability, the original monolithic script was later modularized into:

- `main.py` – main training loop
- `qnodes.py` – definitions of quantum circuits
- `train_loop.py` – training and testing logic
- `quantum_classifier.py` – hybrid model architecture
- `plot_utils.py` – result visualization and figure generation

> This repository shows the original, complete implementation as used in the thesis.

## Results

- During training, models are evaluated across seeds, and results are saved (loss/accuracy curves, parameter checkpoints).
- Quantum circuit drawings are automatically generated using `qml.draw_mpl`.
- Loss and accuracy are visualized with mean and standard deviation over seeds.
- Final generalization gap is computed (train vs. validation loss difference).
