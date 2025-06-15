# Reference-Code
This Repository gives some introduction to my past coding experience. You are welcome to explore it.

## Bachelorthesis
The code for my bachelor's thesis. The project focuses on identifying and analyzing stellar populations in the Chamaeleon star-forming region using Gaia and OmegaCAM data. DBSCAN is used to find distinct populations, photometric and kinematic properties are visualized, and pre-main-sequence accretors are identified via Hα excess. The observational data (ESO) is not included.

## Brownian Motion Analysis
Brownian Motion Analysis is a recent project showcasing a part of my skills in simulation, signal processing, and both unsupervised and supervised learning. The core idea is to simulate three-dimensional trajectories of particles undergoing Brownian motion in a drift field, and then use these trajectories to reconstruct the underlying field dynamics and diffusion properties.
In the unsupervised approach, I apply spectral filtering techniques to suppress noise-dominated frequencies in the trajectory data. The local drift field is then estimated using a KDTree-based method, which evaluates the displacement patterns of nearby trajectory segments to infer the field’s direction and strength at given spatial positions. The reconstruction process depends on two key hyperparameters: the radius used for local drift estimation and the weighting ratio between data fidelity and smoothness in the denoised function. These cannot be optimized directly in an unsupervised setting, while all other parameters are selected automatically in the process.
A supervised approach will follow soon.

## Masterthesis
This is the core code of my master's thesis, the largest project I've worked on so far. The script QNN_1Layer_Pooling runs hybrid quantum-classical neural networks using 16 different quantum circuit architectures. Each circuit is trained over 10 epochs with 10 different random initializations. The dataset size varies depending on the task (e.g., 5_7_MNIST: 400, BreastMNIST: 546, MNIST: 10,000). For better structure, the code was later modularized into main, qnodes, train_loop, quantum_classifier, and plot_utils. Only the most essential code is shown here.
             
## 2Layer_QNN
As part of my master's thesis, this code investigates two-layer quantum convolutional (quanvolutional) circuits in hybrid quantum-classical neural networks. Training is accelerated using CUDA. Compared to the main thesis code, this version is less modular but includes a hands-on implementation of image reshaping and circuit construction for two-layer quantum models. Results are saved and visualized across multiple seeds, with a focus on generalization performance.

## Paper_Crawler
A small hands-on code of mine for a Scrapy spider that collects metadata from arXiv based on a search query. It extracts titles, authors, abstracts, categories, submission dates, comments, and journal info from all matching entries. Pagination is handled automatically. The spider supports custom search terms and date ranges, and fetches data across multiple fields and disciplines.

## Quantum_Graph
A quantum graph network to estimate the ground state of a five-particle system. The variational quantum time evolution (VQTE) is used to find general approximations to the structure/landscape of the eigenstates.
