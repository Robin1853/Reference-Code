# Chamaeleon Star-Forming Region – Population and Accretor Analysis

This repository contains the code used for the analysis in my bachelor's thesis. The project investigates stellar populations and accretion activity in the Chamaeleon star-forming region using Gaia and OmegaCAM data.

> **Note:** The observational data belongs to ESO and is not included in this repository.

## Structure

### `population_finder.py`
Uses [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) from `scikit-learn` to identify and separate stellar populations based on position and proper motion data. It allows for unsupervised discovery of clustering structures in the dataset.

### `chamaeleon_plot.py`
Provides visual diagnostics of the entire catalog:

- Color-magnitude diagrams (e.g., *r vs. r–i*)
- Proper motion and parallax filtering
- Spatial distribution of stars
- Combined figure layouts using `matplotlib` and `seaborn`

### `chamaeleon_accretors.py`
Focuses on identifying **pre-main-sequence (PMS) accretors** using photometric diagnostics:

- Computes Hα excess using color–color diagrams (*r–i vs. r–Hα*)
- Applies empirical correction to Hα magnitudes
- Constructs a reference main sequence (ZAMS) by fitting a robust 3rd-order polynomial (`astropy.modeling` with `sigma_clip`)
- Estimates Hα equivalent width (EW) from the r–Hα excess using a calibrated transformation
- Flags accreting stars with EW > 10 Å and EW > 20 Å
- Visualizes accretors in photometric, spatial, and kinematic plots

## Requirements

- Python ≥ 3.8
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `astropy`, `scipy`

## Output

All figures are automatically saved (e.g. `Ha.png`, `Ha norm.png`) and show:

- Accretion diagnostics
- Stellar populations
- Polynomial fits to the non-accreting sequence
- Proper motion and spatial filtering
