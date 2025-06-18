# Brownian Motion Analysis

This project simulates 3D Brownian motion of particles under the influence of a drift field and analyzes the resulting trajectories to infer underlying dynamics. It combines **stochastic simulation**, **spectral denoising**, and **unsupervised learning** via spatial statistics.

## Key Features

- **Trajectory Simulation**  
  Uses a customizable drift function and diffusion constant to simulate noisy Brownian trajectories in 3D.

- **Drift Field Generation**  
  Visualizes the true analytical drift field used for the simulation.

- **Spectral Filtering (the lowpass_filter_approach)**  
  Applies frequency-domain filtering to suppress noise in the trajectory data. The cutoff is optimized by balancing fidelity to the original data and smoothness (low curvature) of the result. This approach needs fewer trajectories for drift estimation due to filtering noise.

- **Direct Local Drift Estimation (local_denoising_approach)**  
  Estimates the drift field by evaluating local displacement patterns in a KDTree neighborhood. A radial weighting scheme improves robustness to noise. Nevertheless, more trajectories are needed for accurate drift estimation.

- **Error Analysis**  
  Computes the mean squared error between the true and estimated drift field, and visualizes both fields for comparison.

## Code Structure

- `main_simulation.py` – Executes the simulation pipeline, from trajectory generation to drift estimation.
- `Drift_field.py` – Defines and visualizes the true underlying drift field.
- `Trajectory_calculator.py` – Generates Brownian motion trajectories using stochastic integration.
- `lowpassfilter_approach.py` – Contains spectral filtering, KDTree-based local estimation, visualization of the reconstructed field, and diffusion estimation for homogeneous, diffusion field estimation for inhomogeneous fields.
- `local_denoising_approach.py` - Contains KDTree-based local estimation, visualization of the reconstructed field, Wiener filtered Trajectory, and diffusion estimation for homogeneous, diffusion field estimation for inhomogeneous fields.

## Methods Used

- 3D stochastic differential equations (Euler–Maruyama method)
- Welch power spectral density estimation
- FFT-based low-pass filtering with reflection padding
- Curvature-based roughness penalty
- KDTree for fast spatial neighborhood queries

## Future Work

A supervised learning approach (e.g., neural field regression or Gaussian Process estimation) is planned to complement the unsupervised drift reconstruction.

---

*Note: All data is synthetically generated; no external datasets are used.*
