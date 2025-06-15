#!/usr/bin/env python
# coding: utf-8

# --- Import core libraries for data analysis and visualization ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.visualization import hist
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# --- Enable Plotly and Matplotlib inline rendering for Jupyter ---
get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode(connected=True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')

# --- Load cluster candidate data (old GAIA DR2) ---
dfo = pd.read_csv('datapath/Chaemeleon/Cluster/CLRobSt_0.14+10_0.csv')
dfo2 = pd.read_csv('datapath/Chaemeleon/Cluster/CLRobSt_0.14+10_1.csv')
whole = pd.read_csv('datapath/Chaemeleon/riHa_ChamP103_GAIA.csv')  # full field catalog

# --- Load isochrones and Zero-Age Main Sequence (ZAMS) for OmegaCAM and Gaia filters ---
isoG = np.genfromtxt('datapath/Chaemeleon/Omega-Gaia/isoG.dat', skip_header=12, names=True, dtype=None)
isoO = np.genfromtxt('datapath/Chaemeleon/Omega-Gaia/iso.dat', skip_header=12, names=True, dtype=None)
zamsO = np.genfromtxt('datapath/Chaemeleon/Omega-Gaia/zams.dat', skip_header=12, names=True, dtype=None)
zamsG = np.genfromtxt('datapath/Chaemeleon/Omega-Gaia/zamsG.dat', skip_header=12, names=True, dtype=None)

# --- Create density plot of RA/DEC for the first candidate cluster ---

fig, axes = plt.subplots(1, figsize=(24, 12))
sns.set_context('paper', font_scale=2)

# --- KDE (Kernel Density Estimate) of star positions ---
sns.kdeplot(dfo['ra'], dfo['dec'], cmap="Blues", shade=True, common_norm=False, shade_lowest=False)

# --- Overlay scatter of actual cluster members ---
sns.scatterplot(x='ra', y='dec', data=dfo, s=100, alpha=1, color="blue", label="ChaI population")

# --- Set plot labels and view window for DR2 (can be adjusted for EDR3) ---
plt.title('Cluster star density')
plt.xlabel('ra [degree]')
plt.ylabel('dec [degree]')
plt.xlim(162.1, 170.5)
plt.ylim(-78.3, -75.2)
plt.legend(loc=1, fontsize='small', framealpha=1)

# --- Save the resulting figure ---
plot0 = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('cluster density', bbox_inches=plot0.expanded(0.9, 0.9))
#fig.savefig('cluster density G3.png', bbox_inches=plot0.expanded(0.9, 0.9))

# --- Define extinction coefficients for OmegaCAM and Gaia filters (A_V = 1 assumed here) ---
Av = 1
coefR = 0.854 * Av
coefI = 0.68128 * Av
coefHa = 0.80881 * Av
coefG = 0.86209 * Av
coefbp = 1.07198 * Av
coefrp = 0.64648 * Av

# --- Compute absolute magnitudes (distance corrected using parallax) ---
Rabs = dfo['r'] + 5 * np.log10(dfo['parallax']) - 10
Rabs1 = dfo2['r'] + 5 * np.log10(dfo2['parallax']) - 10
RabsW = whole['r'] + 5 * np.log10(whole['parallax']) - 10

Gabs = dfo['phot_g_mean_mag'] + 5 * np.log10(dfo['parallax']) - 10
Gabs1 = dfo2['phot_g_mean_mag'] + 5 * np.log10(dfo2['parallax']) - 10
GabsW = whole['phot_g_mean_mag'] + 5 * np.log10(whole['parallax']) - 10

# --- Select field stars with significant parallax (> 4 mas) ---
cond = (whole['parallax'] > 4)

# --- Plot CMDs (Color-Magnitude Diagrams) for OmegaCAM and Gaia filters ---
fig, axes = plt.subplots(1, 2, figsize=(24, 12))

# --- OmegaCAM CMD ---
sns.scatterplot(x=whole['r'][cond] - whole['i'][cond], y=RabsW[cond],
                data=whole, s=5, alpha=0.2, label="field SOs", ax=axes[0], color="black")
sns.scatterplot(x=dfo['r'] - dfo['i'], y=Rabs,
                data=dfo, alpha=1, s=250, marker="*", label="ChaI population", ax=axes[0], color="blue")

# --- Isochrones and ZAMS overlay ---
axes[0].plot((isoO['rmag'][isoO['logAge'] == 6] + coefR) - (isoO['imag'][isoO['logAge'] == 6] + coefI),
             isoO['rmag'][isoO['logAge'] == 6] + coefR, color='red', label='1 Myr')
axes[0].plot((isoO['rmag'][isoO['logAge'] == 6.69897] + coefR) - (isoO['imag'][isoO['logAge'] == 6.69897] + coefI),
             isoO['rmag'][isoO['logAge'] == 6.69897] + coefR, color='red', ls='--', label='5 Myr')
axes[0].plot((isoO['rmag'][isoO['logAge'] == 7] + coefR) - (isoO['imag'][isoO['logAge'] == 7] + coefI),
             isoO['rmag'][isoO['logAge'] == 7] + coefR, color='red', ls='-.', label='10 Myr')
axes[0].plot((zamsO['rmag'] + coefR) - (zamsO['imag'] + coefI),
             zamsO['rmag'] + coefR, color='black', ls='--', label='ZAMS')

# --- Format OmegaCAM plot ---
axes[0].set_xlim(-0.49, 3.49)
axes[0].set_ylim(15.99, 0.1)
axes[0].set_xlabel('r-i')
axes[0].set_ylabel('M$_r$ [mag]')
axes[0].set_title('OmegaCam')
axes[0].legend()

# --- Gaia CMD ---
sns.scatterplot(x=whole['bp_rp'][cond], y=GabsW[cond],
                data=whole, s=5, alpha=0.2, label="field SOs", ax=axes[1], color="black")
sns.scatterplot(x=dfo['bp_rp'], y=Gabs,
                data=dfo, alpha=1, s=250, marker="*", label="ChaI population", ax=axes[1], color="blue")

# --- Isochrones and ZAMS for Gaia filters ---
axes[1].plot((isoG['G_BPftmag'][isoG['logAge'] == 6] + coefbp) - (isoG['G_RPmag'][isoG['logAge'] == 6] + coefrp),
             isoG['Gmag'][isoG['logAge'] == 6] + coefG, color='red', label='1 Myr')
axes[1].plot((isoG['G_BPftmag'][isoG['logAge'] == 6.69897] + coefbp) - (isoG['G_RPmag'][isoG['logAge'] == 6.69897] + coefrp),
             isoG['Gmag'][isoG['logAge'] == 6.69897] + coefG, color='red', ls='--', label='5 Myr')
axes[1].plot((isoG['G_BPftmag'][isoG['logAge'] == 7] + coefbp) - (isoG['G_RPmag'][isoG['logAge'] == 7] + coefrp),
             isoG['Gmag'][isoG['logAge'] == 7] + coefG, color='red', ls='-.', label='10 Myr')
axes[1].plot((zamsG['G_BPftmag'] + coefbp) - (zamsG['G_RPmag'] + coefrp),
             zamsG['Gmag'] + coefG, color='black', ls='--', label='ZAMS')

# --- Format Gaia plot ---
axes[1].set_xlim(-0.51, 4.99)
axes[1].set_ylim(15.99, 0.1)
axes[1].set_xlabel('bp-rp')
axes[1].set_ylabel('M$_G$ [mag]')
axes[1].set_title('GAIA')
axes[1].legend()

# --- Save the CMD plot with isochrones ---
plot0 = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('cmd iso.png', bbox_inches=plot0.expanded(0.85, 0.9))
#fig.savefig('cmd iso G3.png', bbox_inches=plot0.expanded(0.85, 0.9))

import matplotlib.gridspec as gridspec

# Create a 2x2 subplot layout for the accretion and CMD plots
fig = plt.figure(figsize=(24, 24))
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

# Correction factor for H-alpha calibration
corrHa = 0.22

# Filter out invalid ZAMS entries
zams_sorted = zamsO[zamsO['rmag'] > 0]

# Define subplots
ax00 = fig.add_subplot(spec[0, 0])  # r-i vs r-Ha (with accretion)
ax01 = fig.add_subplot(spec[0, 1])  # CMD (r-i vs Mr)
ax10 = fig.add_subplot(spec[1, :])  # spatial distribution

# --- Plot cluster stars (non-accretors) in color-color diagram ---
sns.scatterplot(
    x=(dfo['r'] - coefR) - (dfo['i'] - coefI),
    y=(dfo['r'] - coefR) - (dfo['ha'] - coefHa + corrHa),
    data=dfo, alpha=0.7, ax=ax00, s=100, color="orange",
    label="clustered non-accretors"
)
sns.scatterplot(
    x=(dfo2['r'] - coefR) - (dfo2['i'] - coefI),
    y=(dfo2['r'] - coefR) - (dfo2['ha'] - coefHa + corrHa),
    data=dfo2, alpha=0.7, ax=ax00, s=100, color="orange"
)

# Prepare model values from ZAMS as accretion baseline
ri0 = (dfo['r'] - coefR) - (dfo['i'] - coefI)
rHa0 = (dfo['r'] - coefR) - (dfo['ha'] - coefHa + corrHa)

ri1 = (dfo2['r'] - coefR) - (dfo2['i'] - coefI)
rHa1 = (dfo2['r'] - coefR) - (dfo2['ha'] - coefHa + corrHa)

ri_mod = pd.Series(zams_sorted['rmag'] - zams_sorted['imag'])
rha_mod = pd.Series(zams_sorted['rmag'] - zams_sorted['Hamag'])
mod = pd.concat([ri_mod, rha_mod], axis=1).sort_values(by=0)

ri_mod1 = ri_mod.copy()
rha_mod1 = rha_mod.copy()
mod1 = pd.concat([ri_mod1, rha_mod1], axis=1).sort_values(by=0)

# Optional: Overplot fitted model curve
ax00.plot(fitted['x'], fitted['y'] - corrHa, alpha=0.7, color='green')

# Plot ZAMS accretion baseline
ax00.plot(
    (zams_sorted['rmag'] - zams_sorted['imag']),
    (zams_sorted['rmag'] - zams_sorted['Hamag']),
    color='red', label='No-acc'
)

ax00.set_xlim(0.1, 2.8)
ax00.set_ylim(0.1, 1.75)
ax00.set_xlabel('r-i')
ax00.set_ylabel('r-H$_{\\alpha}$')
ax00.set_title('OmegaCam with H$_{\\alpha}$')
ax00.legend()

# --- Calculate Hα excess and Equivalent Width (EW) for accretion diagnostics ---

# Interpolate baseline from model to get predicted H-alpha color
rHa_predict = np.interp(ri0, mod[0], mod[1])
rHa_excess = rHa0 - rHa_predict
Eq = 107 * (1 - 10 ** (0.4 * rHa_excess))  # EqW in Angstroms

rHa_predict1 = np.interp(ri1, mod1[0], mod1[1])
rHa_excess1 = rHa1 - rHa_predict1
Eq1 = 107 * (1 - 10 ** (0.4 * rHa_excess1))

# Select stars by EW threshold: >10 Å and >20 Å
ri01 = ri0[Eq < -10]
rHa01 = rHa0[Eq < -10]
ri11 = ri1[Eq1 < -10]
rHa11 = rHa1[Eq1 < -10]

# Plot accretors with different EW levels
sns.scatterplot(x=ri01[Eq > -20], y=rHa01[Eq > -20], alpha=1, ax=ax00, s=150, label='Accretors ChaI >10$\AA$', marker="D", color="blue")
sns.scatterplot(x=ri11[Eq1 > -20], y=rHa11[Eq1 > -20], alpha=1, ax=ax00, s=150, marker="D", color="blue")

sns.scatterplot(x=ri0[Eq < -20], y=rHa0[Eq < -20], alpha=1, ax=ax00, s=400, label='Accretors ChaI >20$\AA$', marker="*", color="blue")
sns.scatterplot(x=ri1[Eq1 < -20], y=rHa1[Eq1 < -20], alpha=1, ax=ax00, s=400, marker="*", color="blue")

# --- CMD (Color-Magnitude Diagram) for accretion-marked stars ---
sns.scatterplot(x=dfo['r'] - dfo['i'], y=Rabs, data=dfo, alpha=0.7, ax=ax01, s=100, color="orange", label="clustered non-accretors")
sns.scatterplot(x=dfo2['r'] - dfo2['i'], y=Rabs1, data=dfo2, alpha=0.7, ax=ax01, s=100, color="orange")

x_newr = dfo['r'][Eq < -10]
x_newi = dfo['i'][Eq < -10]
y_new = Rabs[Eq < -10]

x_newr1 = dfo2['r'][Eq1 < -10]
x_newi1 = dfo2['i'][Eq1 < -10]
y_new1 = Rabs1[Eq1 < -10]

sns.scatterplot(x=x_newr[Eq > -20] - x_newi[Eq > -20], y=y_new[Eq > -20], alpha=1, ax=ax01, s=150, label='Accretors ChaI >10$\AA$', marker="D", color="blue")
sns.scatterplot(x=x_newr1[Eq1 > -20] - x_newi1[Eq1 > -20], y=y_new1[Eq1 > -20], alpha=1, ax=ax01, s=150, marker="D", color="blue")

sns.scatterplot(x=dfo['r'][Eq < -20] - dfo['i'][Eq < -20], y=Rabs[Eq < -20], data=dfo, alpha=1, ax=ax01, s=400, label='Accretors ChaI >20$\AA$', marker="*", color="blue")
sns.scatterplot(x=dfo2['r'][Eq1 < -20] - dfo2['i'][Eq1 < -20], y=Rabs1[Eq1 < -20], data=dfo2, alpha=1, ax=ax01, s=400, marker="*", color="blue")

# Isochrones in CMD
ax01.plot((isoO['rmag'][isoO['logAge'] == 6] + coefR) - (isoO['imag'][isoO['logAge'] == 6] + coefI), isoO['rmag'][isoO['logAge'] == 6] + coefR, color='red', label='1Myr')
ax01.plot((isoO['rmag'][isoO['logAge'] == 6.69897] + coefR) - (isoO['imag'][isoO['logAge'] == 6.69897] + coefI), isoO['rmag'][isoO['logAge'] == 6.69897] + coefR, color='red', label='5Myr', ls='--')
ax01.plot((isoO['rmag'][isoO['logAge'] == 7] + coefR) - (isoO['imag'][isoO['logAge'] == 7] + coefI), isoO['rmag'][isoO['logAge'] == 7] + coefR, color='red', label='10Myr', ls='-.')

# Format CMD
ax01.set_xlim(-0.49, 3.49)
ax01.set_ylim(15, 0.1)
ax01.set_xlabel('r-i')
ax01.set_ylabel('M$_r$ [mag]')
ax01.set_title('OmegaCam without H$_{\\alpha}$')
ax01.legend()

# --- RA/DEC spatial distribution of accretors ---
ax10.minorticks_on()
ax10.get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
ax10.get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True)
ax10.set_title('coordinates')

sns.scatterplot(x=whole['ra'][cond], y=whole['dec'][cond], data=whole, s=5, alpha=0.4, label="field SOs", ax=ax10, color="black")
sns.scatterplot(x=dfo['ra'], y=dfo['dec'], data=dfo, alpha=0.7, ax=ax10, s=100, color="orange", label="clustered non-accretors")
sns.scatterplot(x=dfo2['ra'], y=dfo2['dec'], data=dfo2, alpha=0.7, ax=ax10, s=100, color="orange")

x_newra = dfo['ra'][Eq < -10]
x_newdec = dfo['dec'][Eq < -10]
x_newra1 = dfo2['ra'][Eq1 < -10]
x_newdec1 = dfo2['dec'][Eq1 < -10]

sns.scatterplot(x=x_newra[Eq > -20], y=x_newdec[Eq > -20], data=dfo, alpha=1, ax=ax10, s=150, label='Accretors ChaI >10$\AA$', marker="D", color="blue")
sns.scatterplot(x=x_newra1[Eq1 > -20], y=x_newdec1[Eq1 > -20], data=dfo2, alpha=1, ax=ax10, s=150, marker="D", color="blue")

sns.scatterplot(x=dfo['ra'][Eq < -20], y=dfo['dec'][Eq < -20], data=dfo, alpha=1, ax=ax10, s=400, label='Accretors ChaI >20$\AA$', marker="*", color="blue")
sns.scatterplot(x=dfo2['ra'][Eq1 < -20], y=dfo2['dec'][Eq1 < -20], data=dfo2, alpha=1, ax=ax10, s=400, marker="*", color="blue")

ax10.set_xlim(147, 184.9)
ax10.set_ylim(-81.4, -73.4)
ax10.set_xlabel('ra [degree]')
ax10.set_ylabel('dec [degree]')
ax10.legend(loc=1)

# Save full figure
plot0 = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('Ha.png', bbox_inches=plot0.expanded(0.85, 0.9))
#fig.savefig('Ha G3.png', bbox_inches=plot0.expanded(0.85, 0.9))

# === H-alpha Equivalent Width (EW) Plot ===

# Create figure for visualizing H-alpha EW versus color
fig, axes = plt.subplots(1, figsize=(12, 12))
sns.set_context('paper', font_scale=2)

# Plot clustered non-accretors in orange
sns.scatterplot(x=ri0, y=-Eq, alpha=0.7, s=100, label='clustered non-accretors', color="orange")
sns.scatterplot(x=ri1, y=-Eq1, alpha=0.7, s=100, color="orange")

# Filter moderate accretors (EW < -10 Å) and strong accretors (EW < -20 Å)
ri01 = ri0[Eq > -20]
ri11 = ri1[Eq1 > -20]

# Plot moderate accretors as blue diamonds
sns.scatterplot(x=ri01[Eq < -10], y=-Eq[Eq < -10], alpha=1, s=150, label='Accretors ChaI >10$\AA$', marker="D", color="blue")
sns.scatterplot(x=ri11[Eq1 < -10], y=-Eq1[Eq1 < -10], alpha=1, s=150, marker="D", color="blue")

# Plot strong accretors as blue stars
sns.scatterplot(x=ri0[Eq < -20], y=-Eq[Eq < -20], alpha=1, s=400, marker='*', label='Accretors ChaI >20$\AA$', color="blue")
sns.scatterplot(x=ri1[Eq1 < -20], y=-Eq1[Eq1 < -20], alpha=1, s=400, marker='*', color="blue")

# Add horizontal threshold lines at 10Å and 20Å
axes.plot([0, 3], [20, 20], color="red", label='20$\AA$ limit')
axes.plot([0, 3], [10, 10], color="red", linestyle='--', label='10$\AA$ limit')

# Set plot formatting
axes.set_xlim(0.1, 2.69)
axes.set_ylim(-10, 199)
axes.set_xlabel('r-i')
axes.set_ylabel('EW(H$_{\\alpha}$) [$\\AA$]')
axes.set_title('H$_{\\alpha}$-emission')
axes.legend()

# Save plot to file
plot0 = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('Ha norm.png', bbox_inches=plot0.expanded(1, 0.9))

# === Polynomial Fit to Non-Accretor Sequence (Color-Color Diagram) ===

# Select stars with reliable photometry and reasonable colors in r, i, and Hα
dfno = dfo[
    (dfo['r'] < 90) & (dfo['i'] < 90) & (dfo['ha'] < 90) &
    ((dfo['r'] - dfo['i']) < 4) & ((dfo['r'] - dfo['i']) > -1) &
    ((dfo['r'] - dfo['ha']) > -1) & ((dfo['r'] - dfo['ha']) < 4)
]

# Apply similar filters to the full field sample with a parallax cut
cond_whole = whole[
    (whole['r'] < 90) & (whole['i'] < 90) & (whole['ha'] < 90) &
    ((whole['r'] - whole['i']) < 4) & ((whole['r'] - whole['i']) > -1) &
    ((whole['r'] - whole['ha']) > -1) & ((whole['r'] - whole['ha']) < 4) &
    (whole['parallax'] > 4)
]

# Define color indices for fitting
x = dfno['r'] - dfno['i']        # r-i color
y = dfno['r'] - dfno['ha']       # r-Hα color

x_w = cond_whole['r'] - cond_whole['i']
y_w = cond_whole['r'] - cond_whole['ha']

# Fit a 3rd-order polynomial to the color-color distribution (non-robust initial fit)
new_x = np.linspace(min(x), max(x), num=np.size(x))
coefs = np.polyfit(x, y, 3)
new_line = np.polyval(coefs, new_x)

# Plot initial fit for visual inspection
plt.scatter(x, y)
plt.scatter(new_x, new_line, c='g', marker='^', s=5)
plt.xlim(min(x) - 0.00001, max(x) + 0.00001)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show

# === Robust Polynomial Fit using Astropy ===

# Import fitting utilities and outlier rejection
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
import scipy.stats as stats  # already imported, but retained for completeness

# Initialize 3rd-degree polynomial models for both samples
g_init = models.Polynomial1D(3)   # for clustered sources (dfno)
w_init = models.Polynomial1D(3)   # for field sources (cond_whole)

# Define fitting object with outlier rejection:
# - Uses least-squares fitter
# - Rejects outliers using sigma-clipping (2σ threshold, 3 iterations)
fit = fitting.FittingWithOutlierRemoval(
    fitting.LevMarLSQFitter(), sigma_clip, niter=3, sigma=2.0
)

# Fit both models: returns (fitted_model, mask_of_good_points)
fitted_model, filtered_data = fit(g_init, x, y)
fitted_model_w, filtered_data_w = fit(w_init, x_w, y_w)

# === Visualize Fit Results ===

# Plot original points and inliers for the clustered sources
sns.scatterplot(x, y, label='All points')
sns.scatterplot(x[filtered_data], y[filtered_data], label='Filtered inliers')
nx = np.linspace(-0.2, 3, 100)
sns.lineplot(x=nx, y=fitted_model(nx), label='Polynomial fit')
plt.xlim(0, 5)

# Repeat for the field sources
sns.scatterplot(x_w, y_w, label='All field points')
sns.scatterplot(x_w[filtered_data_w], y_w[filtered_data_w], label='Field inliers')
nx_w = np.linspace(-0.2, 3, 100)
sns.lineplot(x=nx_w, y=fitted_model_w(nx_w), label='Field fit')

# === Save the Polynomial Fit to File ===

# Create array from the fitted model: [color index, predicted r-Hα]
X = np.array([nx, fitted_model(nx)])  # shape: (2, N)

# Convert to DataFrame with appropriate column labels
hdf = pd.DataFrame(X.T, columns=['x', 'y'])

# Sort by color index (just to ensure monotonic order for later interpolation)
hdf = hdf.sort_values(by=['x'])

# Save the model to CSV file (used later for interpolation of r-Hα baseline)
hdf.to_csv('datapath/Chaemeleon/fitted values/MYFIT.csv', index=False)

# (Optional legacy/test code – not used here)
# nx = pd.Series(fitted_model(x))
# ny = pd.Series(fitted_model(y))
# sel_stars = pd.DataFrame(columns=['nx', 'ny'])
# sel_stars['nx'] = nx
# sel_stars['ny'] = ny
# test = pd.array(nx, ny)
# fit_y = pd.Series(fitted_model(x))
# ndf = pd.DataFrame(columns=['X', 'Y'])
# ndf['X'] = x
# ndf['Y'] = fit_y
# ndf.to_csv('MYFIT.csv')

# === Plot the Final ZAMS Fit from Saved CSV ===

# Create side-by-side plot to visualize the model
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
sns.set_context('paper', font_scale=2)

# Plot the fitted r-i vs. r-Hα model (corrected for Hα offset)
axes[0].plot(hdf['x'], hdf['y'] - corrHa, alpha=0.7, color='green')
axes[0].set_xlim(0, 3)
axes[0].set_xlabel('r - i')
axes[0].set_ylabel('r - Hα')
axes[0].set_title('ZAMS Polynomial Fit (corrected)')

# Print the fit data (optional; could be commented out)
print(hdf)

# === Plot r - Hα Color vs. r Magnitude ===

# Create single-panel plot
fig, axes = plt.subplots(1, 1, figsize=(12, 8))
sns.set_context('paper', font_scale=2)

# Scatter plot: vertical axis is r magnitude, horizontal axis is Hα color
axes.scatter(
    (dfo['r']) - (dfo['ha']),  # x-axis: r - Hα
    dfo['r'],                  # y-axis: r magnitude
    color='red',
    label='No-acc'             # Label: presumed non-accreting stars
)

# Set axis limits and labels
axes.set_xlim(-2, 2)
axes.set_ylim(22, 9)  # Inverted y-axis: brighter stars at top
axes.set_xlabel('r$_0$ - Hα$_0$')  # Use subscript formatting for clarity
axes.set_ylabel('r$_0$')
axes.set_title('Color-Magnitude Diagram (r - Hα vs. r)')

# === Plot Histogram of Parallax with Parallax Error Weights ===

# Create a square figure for clarity
fig, axes = plt.subplots(1, 1, figsize=(9, 9))
sns.set_context('paper', font_scale=2)

# Plot histogram of parallaxes with weights = parallax_error
# This shows the contribution of uncertain measurements in a transparent way

# Outline-style histogram (only lines)
hist(
    dfo['parallax'],
    bins=10,
    weights=dfo['parallax_error'],
    histtype='step',
    color='blue'
)

# Filled histogram with slight transparency (for visual comparison)
hist(
    dfo['parallax'],
    bins=10,
    weights=dfo['parallax_error'],
    alpha=0.3,
    color='blue',
    ec='black'  # edge color
)

# You may add axis labels and title manually if needed, e.g.:
# axes.set_xlabel('Parallax [mas]')
# axes.set_ylabel('Weighted count')
# axes.set_title('Weighted Parallax Distribution')
