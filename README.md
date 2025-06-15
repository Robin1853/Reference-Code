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

### Masterthesis ###
My master's thesis is my largest project yet. I wrote additional code for analyzing the data and slight variations for complementary investigations, but only the most important part is shown here. QNN_1Layer_Pooling is the original code used in my thesis. For better structure and readability, I split it into different modules (main, qnodes, train_loop, quantum_classifier, plot_utils). Overall, 16 different quantum circuits were implemented in the network architecture. Each one was trained for 10 different initialization parameters, and over 10 epochs. The number of training images varies for the different datasets (5_7_MNIST: 400, Breastmnist: 546, MNIST: 10.000).
For the two datasets used, please look here: 
MNIST:    http://yann.lecun.com/exdb/mnist
Breastmnist:   https://medmnist.com/
             
### 2Layer_QNN ###
In the context of my master's thesis, I also conducted Investigations of two-layer quanvolutions in hybrid networks. CUDA is used to accelerate the training of the network to acceptable training times. The code is comparable to the Masterthesis code, but less intuitive, due to the hands-on approach of the reshaping of the images for the two-layer quantum circuits.

### Paper_Crawler ###
A small hands-on code of mine for automatic paper search and storage of information regarding possible relevant papers from arXiv.

### Quantum_Graph ###
A quantum graph network to estimate the ground state of a five-particle system. The variational quantum time evolution (VQTE) is used to find general approximations to the structure/landscape of the eigenstates.
