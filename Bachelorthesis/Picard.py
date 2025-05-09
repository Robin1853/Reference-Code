#!/usr/bin/env python
# coding: utf-8

# In[50]:


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

get_ipython().run_line_magic('matplotlib', 'inline')

def coord(r,d,r0,d0):
    x=-(r-r0)*np.cos(d*np.pi/180)*3600
    y=(d-d0)*3600
    dist=np.sqrt(x*x+y*y)
    return x,y,dist

####centrum of cham103
ra0=189.6690230
dec0=-51.1504379


def clip_points(points, bounding_shape):
    if len(bounding_shape.interiors)>0:
        mask=[bounding_shape.contains(Point(p)) for p in points]
    else:
        bounding_path=Path(np.array(bounding_shape.exterior.coords)[:,:2],closed=True)
        mask=bounding_path.contains_points(points[:,:2])
    clipped_points=points[mask]
    return clipped_points


# In[51]:


#Chaem=pd.read_csv('datapath/Chaemeleon/riHa_ChamP103_GAIAEDR3.csv')
Chaem=pd.read_csv('datapath/Chaemeleon/riHa_ChamP103_GAIA.csv')


# In[52]:


print(Chaem)
#Chaem.head(3,8)


# In[53]:


#x_Chaem,y_Chaem,d_Chaem=coord(Chaem['RA'],Chaem['DEC'],ra0,dec0)
x_Chaem,y_Chaem,d_Chaem=coord(Chaem['ra'],Chaem['dec'],ra0,dec0)
#1635941 Gaia
#1656049 GaiaEDR3


# In[55]:


#plot figure: figuresize=()
fig, axes =plt.subplots(1,2,figsize=(24,12))
sns.set_context('paper',font_scale=2)
#sns.scatterplot(x=Chaem['r']-Chaem['i'],y=Chaem['v'],data=Chaem, marker='.',ax=axes,s=30, color'black')
cond = ((Chaem['parallax']>4.) & (Chaem['parallax']<40.))
#axes[0].scatter(Chaem['r'][cond]-Chaem['i'][cond],Chaem['r'][cond],s=2,color='black',alpha=1,edgecolors='none')
axes[0].scatter(Chaem['r'][cond]-Chaem['i'][cond],Chaem['r'][cond] + 5 * np.log10(Chaem['parallax'][cond]) - 10,s=2,color='black',alpha=1,edgecolors='none')
axes[0].minorticks_on()
axes[0].get_xaxis().set_tick_params(direction='in',width=1,which='both',bottom=True, top=True)
axes[0].get_yaxis().set_tick_params(direction='in',width=1,which='both',left=True,right=True)
axes[0].set_xlim(0.1,2.89)
axes[0].set_ylim(15.5,4.1)
axes[0].set_xlabel('r-i')
axes[0].set_ylabel('r')
axes[0].set_title('color-magnitude-diagram')

#propermotion
#axes[1].scatter(Chaem['pmra'][cond],Chaem['pmdec'][cond],s=2,color='black',alpha=1,edgecolors='none')
axes[1].scatter(Chaem['pmra'][cond]*4.74/Chaem['parallax'][cond],Chaem['pmdec'][cond]*4.74/Chaem['parallax'][cond],s=2,color='black',alpha=1,edgecolors='none')
axes[1].minorticks_on()
axes[1].get_xaxis().set_tick_params(direction='in',width=1,which='both',bottom=True, top=True)
axes[1].get_yaxis().set_tick_params(direction='in',width=1,which='both',left=True,right=True)
axes[1].set_xlim(-69,49)
axes[1].set_ylim(-49,49)
axes[1].set_xlabel('pmra')
axes[1].set_ylabel('pmdec')
axes[1].set_title('proper motion')


# In[56]:


fig, axes =plt.subplots(1,2,figsize=(24,12))
sns.set_context('paper',font_scale=2)

#ra-dec original picture
cond = ((Chaem['parallax']>4.) & (Chaem['parallax']<40.))
#axes[0].scatter(Chaem['RA'][cond],Chaem['DEC'][cond],s=2,color='black',alpha=1,edgecolors='none')
axes[0].scatter(Chaem['ra'][cond],Chaem['dec'][cond],s=2,color='black',alpha=1,edgecolors='none')
axes[0].minorticks_on()
axes[0].get_xaxis().set_tick_params(direction='in',width=1,which='both',bottom=True, top=True)
axes[0].get_yaxis().set_tick_params(direction='in',width=1,which='both',left=True,right=True)
axes[0].set_xlim(147,184.9)
axes[0].set_ylim(-81.4,-73.4)
axes[0].set_xlabel('ra')
axes[0].set_ylabel('dec')
axes[0].set_title('angular position')

#histogramm parallax
palx=Chaem['parallax']
#palx.rdiv(1000)[cond].hist(bins=50, ec='black')
palx[cond].hist(bins=50, ec='black')
z=len(palx[cond])/50
y=max(palx)
plt.xlim(0,8)
plt.ylim(0,z/8*y+50)
plt.xlabel('parsec=1000/parallax')


# In[82]:


#print(Chaem.iloc[0, 0])
#print(len(Chaem['n']))
#print(Chaem[cond])


# In[426]:


t=0
cond=((Chaem['parallax']>4.) & (Chaem['parallax']<40.))
# & (Chaem['pmra']>-30) & (Chaem['pmra']<-15) & (Chaem['pmdec']>-10) & (Chaem['pmdec']<10)

ch_parlx=Chaem[cond]['parallax']
print(len(ch_parlx))


# In[425]:


import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib import patches
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.patches as mpatches

cond1=((Chaem['parallax']>4.) & (Chaem['parallax']<40.))
cond2=((Chaem['parallax']>4.) & (Chaem['parallax']<40.) & (Chaem['pmra']>-30) & (Chaem['pmra']<-15) & (Chaem['pmdec']>-10) & (Chaem['pmdec']<10))

#------------------------------------------------------------------------------
fig = plt.figure(figsize=(12.0,12.0))
#no-cond
fig.suptitle("full catalog")
#cond1
#fig.suptitle("4<plx<40")
#cond2
#fig.suptitle("4<plx<40; $-30<v_{\\alpha}<-15$; $-10<v_{\\delta}<10$")
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

ax00 = fig.add_subplot(spec[0, 0])
ax00.minorticks_on()
ax00.get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
ax00.get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True)
ax00.set_title('color-magnitude', fontsize=14, fontweight='bold')
ax00.scatter(Chaem['r']-Chaem['i'],Chaem['r'],s=1,color='black',alpha=0.1,edgecolors='none')
#ax00.scatter(Chaem['r'][cond1]-Chaem['i'][cond1],Chaem['r'][cond1] + 5 * np.log10(Chaem['parallax'][cond1]) - 10,s=1,color='black',alpha=1,edgecolors='none')
#ax00.scatter(Chaem['r'][cond2]-Chaem['i'][cond2],Chaem['r'][cond2] + 5 * np.log10(Chaem['parallax'][cond2]) - 10,s=1,color='black',alpha=1,edgecolors='none')
ax00.set_xlim(-0.6,1.99)
ax00.set_ylim(21.99,10.1)
ax00.set_xlabel('r-i')
ax00.set_ylabel('r')
#ax00.set_ylabel('M$_r$ [mag]')


ellipse1 = Ellipse(xy=(0.27, 0.1), width=0.6, height=0.15, transform=ax00.transAxes, label='white dwarf area', edgecolor='r', fc='None', lw=2)
#ax00.add_patch(ellipse1)
ellipse2 = Ellipse(xy=(0.4, 0.43), width=1.1, height=0.13, angle=-50, transform=ax00.transAxes, label='MS', edgecolor='y', fc='None', lw=2)
#ax00.add_patch(ellipse2)
#ax00.legend(loc=1, fontsize='small')

#ax00.annotate('PMS', xy=(1.6, 9.5),  xycoords='data', xytext=(0.8, 0.7), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='right', verticalalignment='top')

ax01 = fig.add_subplot(spec[0, 1])
ax01.minorticks_on()
ax01.get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
ax01.get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True, labelright=True, labelleft=False)
ax01.set_title('velocity', fontsize=14, fontweight='bold')
ax01.scatter(Chaem['pmra'],Chaem['pmdec'],s=1,color='black',alpha=0.1,edgecolors='none')
#ax01.scatter(Chaem['pmra'][cond1]*4.74/Chaem['parallax'][cond1],Chaem['pmdec'][cond1]*4.74/Chaem['parallax'][cond1],s=1,color='black',alpha=1,edgecolors='none')
#ax01.scatter(Chaem['pmra'][cond2]*4.74/Chaem['parallax'][cond2],Chaem['pmdec'][cond2]*4.74/Chaem['parallax'][cond2],s=1,color='black',alpha=1,edgecolors='none')
ax01.set_xlim(-59,39)
ax01.set_ylim(-39,59)
#ax01.set_xlim(-69,49)
#ax01.set_ylim(-49,49)
#ax01.set_xlim(-29.9,-10.1)
#ax01.set_ylim(-9.9,9.9)
ax01.set_xlabel('pmra')
ax01.set_ylabel('pmdec')
#ax01.set_xlabel('v$_\\alpha$ [km/s]')
#ax01.set_ylabel('v$_\\delta$ [km/s]')
ax01.yaxis.set_label_position("right")

ellipse3 = Ellipse(xy=(0.4, 0.5), width=0.1, height=0.1, transform=ax01.transAxes, edgecolor='r', fc='None', lw=2)
#ax01.add_patch(ellipse3)

ax10 = fig.add_subplot(spec[1, :])
ax10.minorticks_on()
ax10.get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
ax10.get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True)
ax10.set_title('coordinates', fontsize=14, fontweight='bold')
ax10.scatter(Chaem['ra'],Chaem['dec'],s=2,color='black',alpha=0.08,edgecolors='none')
#ax10.scatter(Chaem['ra'][cond1],Chaem['dec'][cond1],s=2,color='black',alpha=1,edgecolors='none')
#ax10.scatter(Chaem['ra'][cond2],Chaem['dec'][cond2],s=2,color='black',alpha=1,edgecolors='none')
ax10.set_xlim(147,184.9)
ax10.set_ylim(-81.4,-73.4)
ax10.set_xlabel('ra [degree]')
ax10.set_ylabel('dec [degree]')

ellipse4 = Ellipse(xy=(0.53, 0.55), width=0.1, height=0.4, transform=ax10.transAxes, edgecolor='r', fc='None', lw=2)
#ax10.add_patch(ellipse4)

fig.savefig('no limitations.png', bbox_inches='tight')
#fig.savefig('plx lim marked.png', bbox_inches='tight')
#fig.savefig('plx lim.png', bbox_inches='tight')
#fig.savefig('plx & pm lim.png', bbox_inches='tight')


# In[ ]:





# In[ ]:




