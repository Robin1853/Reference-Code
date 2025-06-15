# Reference-Code
This Repository gives some introduction to my past coding experience. You are welcome to explore it.

### Bachelorthesis ###
The code of my bachelor's thesis. Population finder uses DBSCAN (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) to locate and distinguish populations in the Chamaeleon star-forming region. Chamaeleon_plot is the code for visual analysis of the observed stars for different parameters. Chamaeleon_accretors is the code for the analysis of the pre-main-sequence stars and the identification of accretors. The data itself belongs to ESO and is not presented here.
This is the code for my bachelor's thesis. The project focuses on identifying and analyzing stellar populations in the Chamaeleon star-forming region using observational data (not included here, data ownership: ESO).

-Population_finder.py uses the DBSCAN algorithm (scikit-learn documentation) to locate and distinguish spatial and kinematic groupings of stars based on Gaia and OmegaCAM data.

-Chamaeleon_plot.py provides a visual analysis of the observed stars with respect to photometric parameters (e.g., color-magnitude diagrams, proper motions, spatial distributions). It includes parallax filtering and color-based classification for population separation.

-Chamaeleon_accretors.py focuses on the identification of pre-main-sequence stars and ongoing accretion processes using Hα excess diagnostics. The code applies photometric corrections, constructs empirical ZAMS references via robust polynomial fitting, and estimates Hα equivalent widths (EW) from r–Hα color indices. Accreting objects are identified using EW thresholds (>10 Å, >20 Å), and their spatial, photometric, and kinematic properties are analyzed.

All photometric corrections, sequence fitting, and accretor diagnostics are fully implemented and visualized, including outlier-resistant modeling (astropy.modeling with sigma_clip) and interpolation of Hα excess for EW computation.

### Brownian Motion Analysis ###
Brownian Motion Analysis is a recent project showcasing my skills in simulation, signal processing, and both unsupervised and supervised learning. The core idea is to simulate three-dimensional trajectories of particles undergoing Brownian motion in a drift field, and then use these trajectories to reconstruct the underlying field dynamics and diffusion properties.
In the unsupervised approach, I apply spectral filtering techniques to suppress noise-dominated frequencies in the trajectory data. The local drift field is then estimated using a KDTree-based method, which evaluates the displacement patterns of nearby trajectory segments to infer the field’s direction and strength at given spatial positions. The reconstruction process depends on two key hyperparameters: the radius used for local drift estimation and the weighting ratio between data fidelity and smoothness in the denoised function. These cannot be optimized directly in an unsupervised setting, while all other parameters are selected automatically in the process.
A supervised approach will follow soon.

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
             
### 2Layer_QNN ###
In the context of my master's thesis, I also conducted Investigations of two-layer quanvolutions in hybrid networks. CUDA is used to accelerate the training of the network to acceptable training times. The code is comparable to the Masterthesis code, but less intuitive, due to the hands-on approach of the reshaping of the images for the two-layer quantum circuits.

### Paper_Crawler ###
A small hands-on code of mine for automatic paper search and storage of information regarding possible relevant papers from arXiv.

### Quantum_Graph ###
A quantum graph network to estimate the ground state of a five-particle system. The variational quantum time evolution (VQTE) is used to find general approximations to the structure/landscape of the eigenstates.
