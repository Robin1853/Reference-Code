#!/usr/bin/env python
# coding: utf-8

# --- Libraries for data handling, visualization, and astrophysical calculations ---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from shapely.geometry import Point, Polygon
from matplotlib.path import Path
from scipy import stats
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
import scipy.stats as stats
from scipy import interpolate
import time

from pylab import *
from scipy.stats import kde
plt.style.use('classic')

# Enables inline plotting in Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# --- Function to compute projected coordinates and distance relative to a reference point ---
def coord(r, d, r0, d0):
    x = -(r - r0) * np.cos(d * np.pi / 180) * 3600  # RA difference in arcseconds
    y = (d - d0) * 3600                             # DEC difference in arcseconds
    dist = np.sqrt(x * x + y * y)                   # Euclidean distance
    return x, y, dist

# --- Coordinates of the reference center of the Chaemeleon-P103 region ---
ra0 = 189.6690230
dec0 = -51.1504379

# --- Function to clip points outside a given polygon shape (e.g., for sky region masks) ---
def clip_points(points, bounding_shape):
    if len(bounding_shape.interiors) > 0:
        mask = [bounding_shape.contains(Point(p)) for p in points]
    else:
        bounding_path = Path(np.array(bounding_shape.exterior.coords)[:, :2], closed=True)
        mask = bounding_path.contains_points(points[:, :2])
    clipped_points = points[mask]
    return clipped_points

# --- Load Gaia catalog data for Chaemeleon region ---
#Chaem = pd.read_csv('datapath/Chaemeleon/riHa_ChamP103_GAIAEDR3.csv')
Chaem = pd.read_csv('datapath/Chaemeleon/riHa_ChamP103_GAIA.csv')

# --- Display full catalog contents (or subset, if desired) ---
print(Chaem)
#Chaem.head(3,8)

# --- Compute relative positions (x/y) and angular distance from center ---
#x_Chaem, y_Chaem, d_Chaem = coord(Chaem['RA'], Chaem['DEC'], ra0, dec0)
x_Chaem, y_Chaem, d_Chaem = coord(Chaem['ra'], Chaem['dec'], ra0, dec0)

#1635941 Gaia
#1656049 GaiaEDR3

# --- Plot 1: Color-Magnitude diagram and Proper Motion diagram ---
fig, axes = plt.subplots(1, 2, figsize=(24, 12))
sns.set_context('paper', font_scale=2)

# --- Select stars with realistic parallax values (i.e., within ~25–250 pc) ---
cond = ((Chaem['parallax'] > 4.) & (Chaem['parallax'] < 40.))

# --- Color-Magnitude Diagram (CMD) using Gaia r, i, and parallax for absolute magnitude ---
axes[0].scatter(Chaem['r'][cond] - Chaem['i'][cond],
                Chaem['r'][cond] + 5 * np.log10(Chaem['parallax'][cond]) - 10,
                s=2, color='black', alpha=1, edgecolors='none')
axes[0].minorticks_on()
axes[0].get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
axes[0].get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True)
axes[0].set_xlim(0.1, 2.89)
axes[0].set_ylim(15.5, 4.1)
axes[0].set_xlabel('r-i')
axes[0].set_ylabel('r')
axes[0].set_title('color-magnitude-diagram')

# --- Proper Motion diagram, converted to velocity in km/s using parallax ---
axes[1].scatter(Chaem['pmra'][cond] * 4.74 / Chaem['parallax'][cond],
                Chaem['pmdec'][cond] * 4.74 / Chaem['parallax'][cond],
                s=2, color='black', alpha=1, edgecolors='none')
axes[1].minorticks_on()
axes[1].get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
axes[1].get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True)
axes[1].set_xlim(-69, 49)
axes[1].set_ylim(-49, 49)
axes[1].set_xlabel('pmra')
axes[1].set_ylabel('pmdec')
axes[1].set_title('proper motion')

# --- Plot 2: Sky positions and parallax histogram ---
fig, axes = plt.subplots(1, 2, figsize=(24, 12))
sns.set_context('paper', font_scale=2)

# --- Plot on-sky positions (RA vs DEC) for stars with good parallaxes ---
axes[0].scatter(Chaem['ra'][cond], Chaem['dec'][cond],
                s=2, color='black', alpha=1, edgecolors='none')
axes[0].minorticks_on()
axes[0].get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
axes[0].get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True)
axes[0].set_xlim(147, 184.9)
axes[0].set_ylim(-81.4, -73.4)
axes[0].set_xlabel('ra')
axes[0].set_ylabel('dec')
axes[0].set_title('angular position')

# --- Histogram of parallaxes (proxy for distance) ---
palx = Chaem['parallax']
palx[cond].hist(bins=50, ec='black')
z = len(palx[cond]) / 50
y = max(palx)
plt.xlim(0, 8)
plt.ylim(0, z / 8 * y + 50)
plt.xlabel('parsec=1000/parallax')

# --- Print number of selected sources ---
t = 0
cond = ((Chaem['parallax'] > 4.) & (Chaem['parallax'] < 40.))
ch_parlx = Chaem[cond]['parallax']
print(len(ch_parlx))

# --- Detailed 3-panel figure layout using GridSpec ---
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib import patches
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.patches as mpatches

# --- Define selection conditions ---
cond1 = ((Chaem['parallax'] > 4.) & (Chaem['parallax'] < 40.))
cond2 = ((Chaem['parallax'] > 4.) & (Chaem['parallax'] < 40.) &
         (Chaem['pmra'] > -30) & (Chaem['pmra'] < -15) &
         (Chaem['pmdec'] > -10) & (Chaem['pmdec'] < 10))

# --- Create a new figure with 3 subplots ---
fig = plt.figure(figsize=(12.0, 12.0))
fig.suptitle("full catalog")  # You can modify this to use cond1 or cond2 instead

spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

# --- Subplot 1: Color-Magnitude Diagram ---
ax00 = fig.add_subplot(spec[0, 0])
ax00.minorticks_on()
ax00.get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
ax00.get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True)
ax00.set_title('color-magnitude', fontsize=14, fontweight='bold')
ax00.scatter(Chaem['r'] - Chaem['i'], Chaem['r'],
             s=1, color='black', alpha=0.1, edgecolors='none')
ax00.set_xlim(-0.6, 1.99)
ax00.set_ylim(21.99, 10.1)
ax00.set_xlabel('r-i')
ax00.set_ylabel('r')
# You may add ellipses to highlight white dwarf region, MS, etc.

# --- Subplot 2: Proper motion (pmra vs pmdec) ---
ax01 = fig.add_subplot(spec[0, 1])
ax01.minorticks_on()
ax01.get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
ax01.get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True, labelright=True, labelleft=False)
ax01.set_title('velocity', fontsize=14, fontweight='bold')
ax01.scatter(Chaem['pmra'], Chaem['pmdec'],
             s=1, color='black', alpha=0.1, edgecolors='none')
ax01.set_xlim(-59, 39)
ax01.set_ylim(-39, 59)
ax01.set_xlabel('pmra')
ax01.set_ylabel('pmdec')
ax01.yaxis.set_label_position("right")

# --- Subplot 3: RA vs DEC (sky position) ---
ax10 = fig.add_subplot(spec[1, :])
ax10.minorticks_on()
ax10.get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
ax10.get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True)
ax10.set_title('coordinates', fontsize=14, fontweight='bold')
ax10.scatter(Chaem['ra'], Chaem['dec'],
             s=2, color='black', alpha=0.08, edgecolors='none')
ax10.set_xlim(147, 184.9)
ax10.set_ylim(-81.4, -73.4)
ax10.set_xlabel('ra [degree]')
ax10.set_ylabel('dec [degree]')

# --- Save final figure to PNG ---
fig.savefig('no limitations.png', bbox_inches='tight')
#fig.savefig('plx lim marked.png', bbox_inches='tight')
#fig.savefig('plx lim.png', bbox_inches='tight')
#fig.savefig('plx & pm lim.png', bbox_inches='tight')
