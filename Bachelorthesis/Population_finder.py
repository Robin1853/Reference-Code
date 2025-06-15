#!/usr/bin/env python
# coding: utf-8

"""
This script performs clustering on stellar data using the DBSCAN algorithm.
It includes data normalization, cluster detection, and a variety of visualizations,
including CMD diagrams, velocity space, and parallax-based distances.
"""

# Imports
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from astropy.visualization import hist
import seaborn as sns

# Input parameters for DBSCAN
db_scan1 = float(input('eps='))
db_scan2 = int(input('min stars='))

# Normalization function for input array
def in_parlx(parin):
    parin = parin - np.mean(parin)
    parout = parin / np.std(parin)
    return parout

# Load data
Chaem = pd.read_csv('datapath/Chaemeleon/riHa_ChamP103_GAIA.csv')

# Apply conditions on parallax and proper motion
par_cond = ((Chaem['parallax'] > 4.) & (Chaem['parallax'] < 40.))
pm_cond = ((Chaem['pmra'] > -30) & (Chaem['pmra'] < -15) & (Chaem['pmdec'] > -10) & (Chaem['pmdec'] < 10))

# Filter data
cg = Chaem[par_cond & pm_cond]
data_raw = cg['ra'], cg['dec'], cg['parallax'], cg['pmra'] * 4.74 / cg['parallax'], cg['pmdec'] * 4.74 / cg['parallax']
data_raw = np.array(data_raw).astype(float).T

# Normalize selected columns for clustering
dist_norm = in_parlx(1000 / cg['parallax'])
pmra_norm = in_parlx(cg['pmra'] * 4.74 / cg['parallax'])
pmdec_norm = in_parlx(cg['pmdec'] * 4.74 / cg['parallax'])
data_norm = np.array([dist_norm, pmra_norm, pmdec_norm]).astype(float).T

# Apply DBSCAN clustering
db = DBSCAN(eps=db_scan1, min_samples=db_scan2).fit(data_norm)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Count number of clusters and noise points
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Counted number of clusters: %d' % n_clusters_)

# Save individual clusters and combined clusters
n = 0
u = {}
cluster = {}
for cl in set(labels):
    if cl >= 0:
        globals()['X%s' % cl] = cg[labels == cl]
        globals()['X%s' % cl].to_csv(f'datapath/Chaemeleon/Cluster/CLRobSt_{db_scan1}+{db_scan2}_{cl}.csv', index=False)
        u[cl] = cg[labels == cl]
        print(f'---->Cluster {cl}', len(globals()['X%s' % cl]))
        n += len(globals()['X%s' % cl])

print('total ---->', n)

XALL = cg[labels >= 0]
XALL.to_csv('ClRobSt1_ALL.csv', index=False)

# Plot raw data with cluster labels
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        zorder = 0
        col = [0, 0, 0, 1]  # black for noise
    else:
        zorder = 1
    class_member_mask = (labels == k)
    xy = data_raw[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6, zorder=zorder)

plt.title('Clusters')
plt.show()

for inx in range(n_clusters_):
    #cluster=[0,0,0,0,0,0,0]
    cluster[inx]=pd.merge(Chaem, (u[inx])['n'], on='n')
    print(cluster[inx])

ra0=189.6690230
dec0=-51.1504379

def coord(r,d,r0,d0):
    x=-(r-r0)*np.cos(d*np.pi/180)*3600
    y=(d-d0)*3600
    dist=np.sqrt(x*x+y*y)
    return x,y,dist
x_Chaem,y_Chaem,d_Chaem=coord(Chaem['ra'],Chaem['dec'],ra0,dec0)

# Plot: Color-Magnitude Diagram (CMD) and Proper Motion Diagram
fig, axes = plt.subplots(1, 2, figsize=(24, 12))
sns.set_context('paper', font_scale=2)

# Filter condition for plotting
cond = ((Chaem['parallax'] > 4.) & (Chaem['parallax'] < 40.))

# CMD plot
axes[0].scatter(
    Chaem['r'][cond] - Chaem['i'][cond],
    Chaem['r'][cond] + 5 * np.log10(Chaem['parallax'][cond]) - 10,
    s=2, color='black', label="field SOs", alpha=1, edgecolors='none')

paint = ['blue', 'green', 'orange', 'red', 'yellow']
axes[0].scatter(
    cluster[0]['r'] - cluster[0]['i'],
    cluster[0]['r'] + 5 * np.log10(cluster[0]['parallax']) - 10,
    s=60, color='blue', label="clustered SOs", alpha=1, edgecolors='none')
axes[0].scatter(
    cluster[1]['r'] - cluster[1]['i'],
    cluster[1]['r'] + 5 * np.log10(cluster[1]['parallax']) - 10,
    s=60, color='blue', alpha=1, edgecolors='none')

axes[0].minorticks_on()
axes[0].tick_params(direction='in', width=1, which='both', bottom=True, top=True, left=True, right=True)
axes[0].set_xlim(0.1, 2.99)
axes[0].set_ylim(15.5, 3.1)
axes[0].set_xlabel('r-i')
axes[0].set_ylabel('M$_r$ [mag]')
axes[0].set_title('color-magnitude diagram')
plot0 = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
axes[0].legend(loc=1, fontsize='small')
fig.savefig('cmd marked 1.png', bbox_inches=plot0.expanded(1.2, 1.2))

# Proper motion plot
axes[1].scatter(
    Chaem['pmra'][cond] * 4.74 / Chaem['parallax'][cond],
    Chaem['pmdec'][cond] * 4.74 / Chaem['parallax'][cond],
    s=5, color='black', label="field SOs", alpha=1, edgecolors='none')
axes[1].scatter(
    cluster[0]['pmra'] * 4.74 / cluster[0]['parallax'],
    cluster[0]['pmdec'] * 4.74 / cluster[0]['parallax'],
    s=60, color='blue', label="clustered SOs", alpha=1, edgecolors='none')
axes[1].scatter(
    cluster[1]['pmra'] * 4.74 / cluster[1]['parallax'],
    cluster[1]['pmdec'] * 4.74 / cluster[1]['parallax'],
    s=60, color='blue', alpha=1, edgecolors='none')

axes[1].minorticks_on()
axes[1].tick_params(direction='in', width=1, which='both', bottom=True, top=True, left=True, right=True)
axes[1].set_xlim(-29.9, -0.1)
axes[1].set_ylim(-14.9, 14.9)
axes[1].set_xlabel('v$_{\alpha}$ [km/s]')
axes[1].set_ylabel('v$_{\delta}$ [km/s]')
axes[1].set_title('velocity')
plot1 = axes[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
axes[1].legend(loc=1, fontsize='small', framealpha=1)
fig.savefig('velo marked 1.png', bbox_inches=plot1.expanded(1.25, 1.2))

fig, axes = plt.subplots(1, 2, figsize=(24, 12))
sns.set_context('paper', font_scale=2)

# Proper motion scatter plot
axes[0].scatter(
    Chaem['pmra'][cond] * 4.74 / Chaem['parallax'][cond],
    Chaem['pmdec'][cond] * 4.74 / Chaem['parallax'][cond],
    s=5, color='black', label="field SOs", alpha=1, edgecolors='none')
axes[0].scatter(
    cluster[0]['pmra'] * 4.74 / cluster[0]['parallax'],
    cluster[0]['pmdec'] * 4.74 / cluster[0]['parallax'],
    s=60, color=paint[0], label="clustered SOs", alpha=1, edgecolors='none')
axes[0].scatter(
    cluster[1]['pmra'] * 4.74 / cluster[1]['parallax'],
    cluster[1]['pmdec'] * 4.74 / cluster[1]['parallax'],
    s=60, color=paint[0], alpha=1, edgecolors='none')

axes[0].minorticks_on()
axes[0].tick_params(direction='in', width=1, which='both', bottom=True, top=True, left=True, right=True)
axes[0].set_xlim(-29.9, -0.1)
axes[0].set_ylim(-14.9, 14.9)
axes[0].set_xlabel('v$_{\alpha}$ [km/s]')
axes[0].set_ylabel('v$_{\delta}$ [km/s]')
axes[0].set_title('velocity')
axes[0].legend(loc=1, fontsize='small')

# KDE (density) plot for velocity space
level = ([0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1], [0, 0.2, 0.3])
cmap = ["Blues", "Greens"]
cluster[0] = cluster[0].append(cluster[1])
den = sns.kdeplot(
    cluster[0]['pmra'] * 4.74 / cluster[0]['parallax'],
    cluster[0]['pmdec'] * 4.74 / cluster[0]['parallax'],
    ax=axes[1], cmap=cmap[0], common_norm=False, shade=True,
    alpha=0.7, levels=level[0])
axes[1].scatter(
    cluster[0]['pmra'] * 4.74 / cluster[0]['parallax'],
    cluster[0]['pmdec'] * 4.74 / cluster[0]['parallax'],
    s=40, color=paint[0], label="clustered SOs", alpha=1, edgecolors='none')

axes[1].set_xlabel('v$_{\alpha}$ [km/s]')
axes[1].set_ylabel('v$_{\delta}$ [km/s]')
axes[1].set_title('velocity density')
axes[1].set_xlim(-22.4, -18.6)
axes[1].set_ylim(-1.4, 2.9)
axes[1].legend(loc=1, fontsize='small')
fig.savefig('velo density 1.png', bbox_inches='tight')

# Coordinate and distance histogram plots
fig, axes = plt.subplots(1, 2, figsize=(24, 12))
sns.set_context('paper', font_scale=2)
axes[0].scatter(
    Chaem['ra'][cond], Chaem['dec'][cond], s=2,
    color='black', label="field SOs", alpha=1, edgecolors='none')
axes[0].scatter(
    cluster[0]['ra'], cluster[0]['dec'], s=30,
    color=paint[0], label="clustered SOs", alpha=1, edgecolors='none')
axes[0].minorticks_on()
axes[0].tick_params(direction='in', width=1, which='both', bottom=True, top=True, left=True, right=True)
axes[0].set_xlim(147, 184.9)
axes[0].set_ylim(-81.4, -73.4)
axes[0].set_xlabel('ra [degree]')
axes[0].set_ylabel('dec [degree]')
axes[0].set_title('coordinates')
plot0 = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
axes[0].legend(loc=1, fontsize='small', framealpha=1)
fig.savefig('coord marked 1.png', bbox_inches=plot0.expanded(1.25, 1.2))

binwidth = 2.2
bin = [8, 8]
dist = 1000 / cluster[0]['parallax']
dist_error = dist * (cluster[0]['parallax_error'] / cluster[0]['parallax'])
axes[1].hist(
    dist, color=paint[0], weights=dist_error,
    ec='black', label="clustered SOs", alpha=0.5, bins=bin[0])
axes[1].set_title('histogram distance')
plot1 = axes[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
axes[1].legend(loc=1, fontsize='small', framealpha=1)
fig.savefig('dist 1.png', bbox_inches=plot1.expanded(1.2, 1.2))

# Parallax error analysis
a, b = 0, 0
for j in range(n_clusters_):
    plxer = cluster[j]['parallax_error'] / cluster[j]['parallax']
    a += len(plxer)
    b += sum(plxer)

# KS-Test (normal data)
from scipy import stats
stats.ks_2samp(1000 / cluster[0]['parallax'], 1000 / cluster[1]['parallax'])

# Histogram and Gaussian fit of parallax-based distance
fig, axes = plt.subplots(1, 2, figsize=(24, 12))
binwidth = 2.2
dist = 1000 / cluster[0]['parallax']
dist_error = dist * (cluster[0]['parallax_error'] / cluster[0]['parallax'])
cond3 = (cluster[0]['parallax_error'] / cluster[0]['parallax']) < 0.015
ndist = dist[cond3]
ndist_error = dist_error[cond3]
_, bins, _ = axes[1].hist(
    ndist, bins=np.arange(min(ndist), max(ndist) + binwidth, binwidth),
    weights=ndist_error, density=True,
    label="clustered SOs", alpha=0.5, color=paint[0], ec='black')
mu, sigma = norm.fit(ndist)
gaussian_fit = norm.pdf(bins, mu, sigma)
axes[1].plot(bins, gaussian_fit, label="fit")
axes[1].set_xlim(180.1, 201.9)
axes[1].set_ylim(0.001, 0.17)
axes[1].set_xlabel('distance [pc]')
axes[1].set_title('histogram distance & gaussian fit')
axes[1].legend(loc=1, fontsize='small', framealpha=1)
plot = axes[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('dist fit 1.png', bbox_inches=plot.expanded(1.2, 1.2))

# KS-Test (fitted data)
co1 = (cluster[0]['parallax_error'] / cluster[0]['parallax']) < 0.015
co2 = (cluster[1]['parallax_error'] / cluster[1]['parallax']) < 0.015
c1 = (1000 / cluster[0]['parallax'])[co1]
c2 = (1000 / cluster[1]['parallax'])[co2]
stats.ks_2samp(c1, c2)

# Sequence plot (color-color diagram)
fig, axes = plt.subplots(1, 2, figsize=(24, 12))
sns.set_context('paper', font_scale=2)
cond = ((Chaem['parallax'] > 4.) & (Chaem['parallax'] < 40.))
axes[0].scatter(
    Chaem['r'][cond] - Chaem['i'][cond],
    Chaem['r'][cond] - Chaem['ha'][cond],
    s=2, color='black', alpha=1, edgecolors='none')
for t in range(n_clusters_):
    axes[0].scatter(
        cluster[t]['r'] - cluster[t]['i'],
        cluster[t]['r'] - cluster[t]['ha'],
        s=60, color=paint[t], alpha=1, edgecolors='none')
axes[0].minorticks_on()
axes[0].tick_params(direction='in', width=1, which='both', bottom=True, top=True, left=True, right=True)
axes[0].set_xlim(-0.4, 3)
axes[0].set_ylim(-0.4, 2)
axes[0].set_xlabel('R-I')
axes[0].set_ylabel('R-Ha')
axes[0].set_title('sequence')
