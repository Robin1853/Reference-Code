#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot

from astropy.visualization import hist

init_notebook_mode(connected=True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


#GAIA old data
dfo = pd.read_csv('datapath/Chaemeleon/Cluster/CLRobSt_0.14+10_0.csv')
dfo2 = pd.read_csv('datapath/Chaemeleon/Cluster/CLRobSt_0.14+10_1.csv')
whole=pd.read_csv('datapath/Chaemeleon/riHa_ChamP103_GAIA.csv')
#dfo=dfo.append(dfo2)

#GAIA3 data
#dfo = pd.read_csv('datapath/Chaemeleon/Cluster/CLRobSt_0.19+15_0.csv')
#dfo.rename(columns={'RA':'ra','DEC':'dec'}, inplace=True)
#dfo2 = pd.read_csv('datapath/Chaemeleon/Cluster/CLRobSt_0.19+15_1.csv')
#dfo2.rename(columns={'RA':'ra','DEC':'dec'}, inplace=True)
#whole=pd.read_csv('datapath/Chaemeleon/riHa_ChamP103_GAIAEDR3.csv')
#whole.rename(columns={'RA':'ra','DEC':'dec'}, inplace=True)

isoG = np.genfromtxt('datapath/Chaemeleon/Omega-Gaia/isoG.dat',skip_header=12,names=True,dtype=None)
isoO = np.genfromtxt('datapath/Chaemeleon/Omega-Gaia/iso.dat',skip_header=12,names=True,dtype=None)

zamsO = np.genfromtxt('datapath/Chaemeleon/Omega-Gaia/zams.dat',skip_header=12,names=True,dtype=None)
zamsG = np.genfromtxt('datapath/Chaemeleon/Omega-Gaia/zamsG.dat',skip_header=12,names=True,dtype=None)


# In[ ]:





# In[51]:


fig, axes = plt.subplots(1, figsize=(24,12))
sns.set_context('paper',font_scale=2)
#sns.set_style("white")
sns.kdeplot(dfo['ra'],dfo['dec'], cmap="Blues", shade=True, common_norm=False, shade_lowest=False)
#sns.kdeplot(dfo2['ra'],dfo2['dec'], cmap="Greens", shade=True, shade_lowest=False)
#sns.kdeplot(dfo2['ra'],dfo2['dec'],cbar=True,shade=False, alpha=0.8)
sns.scatterplot(x='ra',y='dec',data=dfo, s=100, alpha=1, color="blue", label="ChaI population")
#sns.scatterplot(x='ra',y='dec',data=dfo, s=100, alpha=1, color="blue", label="blue cluster")
#sns.scatterplot(x='ra',y='dec',data=dfo2, s=100, alpha=1, color="green", label="green cluster")
plt.title('Cluster star density')
plt.xlabel('ra [degree]')
plt.ylabel('dec [degree]')

#old data
plt.xlim(162.1, 170.5)
plt.ylim(-78.3, -75.2)


#G3 data
#plt.xlim(165.1, 170.5)
#plt.ylim(-77.9, -76.1)

plt.legend(loc=1,fontsize='small', framealpha=1)

plot0=bbox_inches=bbox_inches=fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('cluster density', bbox_inches=plot0.expanded(0.9,0.9))
#fig.savefig('cluster density G3.png', bbox_inches=plot0.expanded(0.9,0.9))


# In[100]:


#OmegaCAM filters and Gaia filters
#Av = 3.1* 0.5
Av=1
coefR = 0.854 * Av
coefI = 0.68128 * Av
coefHa = 0.80881 * Av
coefG = 0.86209 * Av
coefbp = 1.07198 * Av
coefrp = 0.64648 * Av

Rabs = dfo['r']+5*np.log10(dfo['parallax'])-10
Rabs1 = dfo2['r']+5*np.log10(dfo2['parallax'])-10
RabsW = whole['r']+5*np.log10(whole['parallax'])-10
Gabs = dfo['phot_g_mean_mag']+5*np.log10(dfo['parallax'])-10
Gabs1 = dfo2['phot_g_mean_mag']+5*np.log10(dfo2['parallax'])-10
GabsW = whole['phot_g_mean_mag']+5*np.log10(whole['parallax'])-10

#fit
cond = (whole['parallax']>4)


# In[67]:




#fit
cond = (whole['parallax']>4)


#plt.figure()
fig, axes = plt.subplots(1,2, figsize=(24,12))

###OmegaCAM filters
sns.scatterplot(x=whole['r'][cond]-whole['i'][cond],y=(RabsW[cond]),data=whole, s=5, alpha=0.2, label="field SOs", ax=axes[0], color="black")
sns.scatterplot(x=dfo['r']-dfo['i'],y=(Rabs),data=dfo, alpha=1, s=250, marker="*", label="ChaI population", ax=axes[0], color="blue")
#sns.scatterplot(x=dfo['r']-dfo['i'],y=(Rabs),data=dfo, alpha=1, s=250, marker="*", label="blue cluster", ax=axes[0], color="blue")
#sns.scatterplot(x=dfo2['r']-dfo2['i'],y=(Rabs1),data=dfo2, alpha=1, s=250, marker="*",label="green cluster", ax=axes[0], color="green")

#ZAMS and isochrones
axes[0].plot((isoO['rmag'][isoO['logAge']==6]+coefR)-(isoO['imag'][isoO['logAge']==6]+coefI),             isoO['rmag'][isoO['logAge']==6]+coefR,color='red',label='1Myr')

axes[0].plot((isoO['rmag'][isoO['logAge']==6.69897]+coefR)-(isoO['imag'][isoO['logAge']==6.69897]+coefI),             isoO['rmag'][isoO['logAge']==6.69897]+coefR,color='red',label='5Myr',ls='--')

axes[0].plot((isoO['rmag'][isoO['logAge']==7]+coefR)-(isoO['imag'][isoO['logAge']==7]+coefI),             isoO['rmag'][isoO['logAge']==7]+coefR,color='red',label='10Myr',ls='-.')

axes[0].plot((zamsO['rmag']+coefR)-(zamsO['imag']+coefI),             zamsO['rmag']+coefR,color='black',label='ZAMS',ls='--')


###Gaia filters
sns.scatterplot(x=whole['bp_rp'][cond],y=(GabsW[cond]),data=whole, s=5, alpha=0.2, label="field SOs", ax=axes[1], color="black")
sns.scatterplot(x=dfo['bp_rp'],y=(Gabs),data=dfo, alpha=1, s=250, marker="*", label="ChaI population", ax=axes[1], color="blue")
#sns.scatterplot(x=dfo['bp_rp'],y=(Gabs),data=dfo, alpha=1, s=250, marker="*", label="blue cluster", ax=axes[1], color="blue")
#sns.scatterplot(x=dfo2['bp_rp'],y=(Gabs1),data=dfo2, alpha=1, s=250, marker="*", label="green cluster", ax=axes[1], color="green")

#ZAMS and isochrones
axes[1].plot((isoG['G_BPftmag'][isoO['logAge']==6]+coefbp)-(isoG['G_RPmag'][isoO['logAge']==6]+coefrp),             isoG['Gmag'][isoO['logAge']==6]+coefG,color='red',label='1Myr')

axes[1].plot((isoG['G_BPftmag'][isoG['logAge']==6.69897]+coefbp)-(isoG['G_RPmag'][isoG['logAge']==6.69897]+coefrp),             isoG['Gmag'][isoG['logAge']==6.69897]+coefG,color='red',label='5Myr',ls='--')

axes[1].plot((isoG['G_BPftmag'][isoG['logAge']==7]+coefbp)-(isoG['G_RPmag'][isoG['logAge']==7]+coefrp),             isoG['Gmag'][isoG['logAge']==7]+coefG,color='red',label='10Myr',ls='-.')

axes[1].plot((zamsG['G_BPftmag']+coefbp)-(zamsG['G_RPmag']+coefrp),             zamsG['Gmag']+coefG,color='black',label='ZAMS',ls='--')



#plot settings
axes[0].set_xlim(-0.49,3.49)
axes[0].set_ylim(15.99,0.1)
axes[0].set_xlabel('r-i')
axes[0].set_ylabel('M$_r$ [mag]')
axes[0].set_title('OmegaCam')
axes[0].legend()

axes[1].set_xlim(-0.51,4.99)
axes[1].set_ylim(15.99,0.1)
axes[1].set_xlabel('bp-rp')
axes[1].set_ylabel('M$_G$ [mag]')
axes[1].set_title('GAIA')
axes[1].legend()


plot0=bbox_inches=bbox_inches=fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('cmd iso.png', bbox_inches=plot0.expanded(0.85,0.9))
#fig.savefig('cmd iso G3.png', bbox_inches=plot0.expanded(0.85,0.9))


# In[ ]:





# In[102]:


import matplotlib.gridspec as gridspec

#fig,axes=plt.subplots(1,2,3,figsize=(24,12))
#sns.set_context('paper',font_scale=2)
fig=plt.figure(figsize=(24,24))
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

corrHa=0.22
zams_sorted=zamsO[zamsO['rmag']>0]

ax00 = fig.add_subplot(spec[0, 0])
ax01 = fig.add_subplot(spec[0, 1])
ax10 = fig.add_subplot(spec[1, :])

sns.scatterplot(x=(dfo['r']-coefR)-(dfo['i']-coefI),y=(dfo['r']-coefR)-(dfo['ha']-coefHa+corrHa),data=dfo,alpha=0.7,ax=ax00, s=100, color="orange", label="clustered non-accretors")
sns.scatterplot(x=(dfo2['r']-coefR)-(dfo2['i']-coefI),y=(dfo2['r']-coefR)-(dfo2['ha']-coefHa+corrHa),data=dfo2,alpha=0.7,ax=ax00, s=100, color="orange")



#read fitted data
#fitted=pd.read_csv('datapath/Chaemeleon/fitted values/MYFIT.csv')

ri0=(dfo['r']-coefR)-(dfo['i']-coefI)
rHa0=(dfo['r']-coefR)-(dfo['ha']-coefHa+corrHa)
ri_mod=pd.Series(zams_sorted['rmag']-zams_sorted['imag'])
#ri_mod=pd.Series(fitted['x'])
rha_mod=pd.Series(zams_sorted['rmag']-zams_sorted['Hamag'])
#rha_mod=pd.Series(fitted['y'])
mod=pd.concat([ri_mod, rha_mod],axis=1).sort_values(by=0,axis=0)

ri1=(dfo2['r']-coefR)-(dfo2['i']-coefI)
rHa1=(dfo2['r']-coefR)-(dfo2['ha']-coefHa+corrHa)
ri_mod1=pd.Series(zams_sorted['rmag']-zams_sorted['imag'])
rha_mod1=pd.Series(zams_sorted['rmag']-zams_sorted['Hamag'])
mod1=pd.concat([ri_mod1, rha_mod1],axis=1).sort_values(by=0,axis=0)

ax00.plot(fitted['x'],fitted['y']-corrHa, alpha=0.7, color='green')

#plot zams magnitudes
#zams_pd=pd.DataFrame(zamsO)

ax00.plot((zams_sorted['rmag'])-(zams_sorted['imag']),zams_sorted['rmag']-zams_sorted['Hamag'],color='red',label='No-acc')

ax00.set_xlim(0.1,2.8)
ax00.set_ylim(0.1,1.75)
ax00.set_xlabel('r-i')
ax00.set_ylabel('r-H$_{\\alpha}$')
ax00.set_title('OmegaCam with H$_{\\alpha}$')
ax00.legend()



rHa_predict=np.interp(ri0, mod[0], mod[1], left=None ,right=None)
rHa_excess=rHa0-rHa_predict
Eq=107*(1-10**(0.4*rHa_excess))

rHa_predict1=np.interp(ri1, mod1[0], mod1[1], left=None ,right=None)
rHa_excess1=rHa1-rHa_predict1
Eq1=107*(1-10**(0.4*rHa_excess1))

#sns.scatterplot()
ri01=ri0[Eq<-10]
rHa01=rHa0[Eq<-10]
ri11=ri1[-10>Eq1]
rHa11=rHa1[-10>Eq1]

#sns.scatteplots first two lines are DR2 data & second two are EDR3 data

sns.scatterplot(x=ri01[Eq>-20],y=rHa01[Eq>-20],alpha=1,ax=ax00,s=150, label='Accretors ChaI >10$\AA$', marker="D", color="blue")
sns.scatterplot(x=ri11[Eq1>-20],y=rHa11[Eq1>-20],alpha=1,ax=ax00,s=150, marker="D", color="blue")
#sns.scatterplot(x=ri01[Eq>-20],y=rHa01[Eq>-20],alpha=1,ax=ax00,s=150, label='blue Accretors >10$\AA$', marker="D", color="blue")
#sns.scatterplot(x=ri11[Eq1>-20],y=rHa11[Eq1>-20],alpha=1,ax=ax00,s=150, label='green Accretors >10$\AA$', marker="D", color="green")

sns.scatterplot(x=ri0[Eq<-20],y=rHa0[Eq<-20],alpha=1,ax=ax00,s=400, label='Accretors ChaI >20$\AA$', marker="*", color="blue")
sns.scatterplot(x=ri1[Eq1<-20],y=rHa1[Eq1<-20],alpha=1,ax=ax00,s=400, marker="*", color="blue")
#sns.scatterplot(x=ri0[Eq<-20],y=rHa0[Eq<-20],alpha=1,ax=ax00,s=400, label='blue Accretors >20$\AA$', marker="*", color="blue")
#sns.scatterplot(x=ri1[Eq1<-20],y=rHa1[Eq1<-20],alpha=1,ax=ax00,s=400, label='green Accretors >20$\AA$', marker="*", color="green")

###OmegaCam filters
sns.scatterplot(x=dfo['r']-dfo['i'],y=(Rabs),data=dfo, alpha=0.7, ax=ax01,s=100, color="orange", label="clustered non-accretors")
sns.scatterplot(x=dfo2['r']-dfo2['i'],y=(Rabs1),data=dfo2, alpha=0.7, ax=ax01,s=100, color="orange")

x_newr=dfo['r'][Eq<-10]
x_newi=dfo['i'][Eq<-10]
y_new=Rabs[Eq<-10]
x_newr1=dfo2['r'][Eq1<-10]
x_newi1=dfo2['i'][Eq1<-10]
y_new1=Rabs1[Eq1<-10]

sns.scatterplot(x=x_newr[Eq>-20]-x_newi[Eq>-20],y=y_new[Eq>-20],alpha=1,ax=ax01,s=150, label='Accretors ChaI >10$\AA$', marker="D", color="blue")
sns.scatterplot(x=x_newr1[Eq1>-20]-x_newi1[Eq1>-20],y=y_new1[Eq1>-20],alpha=1,ax=ax01,s=150, marker="D", color="blue")
#sns.scatterplot(x=x_newr[Eq>-20]-x_newi[Eq>-20],y=y_new[Eq>-20],alpha=1,ax=ax01,s=150, label='blue Accretors >10$\AA$', marker="D", color="blue")
#sns.scatterplot(x=x_newr1[Eq1>-20]-x_newi1[Eq1>-20],y=y_new1[Eq1>-20],alpha=1,ax=ax01,s=150, label='green Accretors >10$\AA$', marker="D", color="green")

sns.scatterplot(x=dfo['r'][Eq<-20]-dfo['i'][Eq<-20],y=Rabs[Eq<-20],data=dfo,alpha=1,ax=ax01,s=400,label='Accretors ChaI >20$\AA$', marker="*", color="blue")
sns.scatterplot(x=dfo2['r'][Eq1<-20]-dfo2['i'][Eq1<-20],y=Rabs1[Eq1<-20],data=dfo2,alpha=1,ax=ax01,s=400, marker="*", color="blue")
#sns.scatterplot(x=dfo['r'][Eq<-20]-dfo['i'][Eq<-20],y=Rabs[Eq<-20],data=dfo,alpha=1,ax=ax01,s=400,label='blue Accretors >20$\AA$', marker="*", color="blue")
#sns.scatterplot(x=dfo2['r'][Eq1<-20]-dfo2['i'][Eq1<-20],y=Rabs1[Eq1<-20],data=dfo2,alpha=1,ax=ax01,s=400,label='green Accretors >20$\AA$', marker="*", color="green")

ax01.plot((isoO['rmag'][isoO['logAge']==6]+coefR)-(isoO['imag'][isoO['logAge']==6]+coefI),             isoO['rmag'][isoO['logAge']==6]+coefR,color='red',label='1Myr')

ax01.plot((isoO['rmag'][isoO['logAge']==6.69897]+coefR)-(isoO['imag'][isoO['logAge']==6.69897]+coefI),             isoO['rmag'][isoO['logAge']==6.69897]+coefR,color='red',label='5Myr',ls='--')

ax01.plot((isoO['rmag'][isoO['logAge']==7]+coefR)-(isoO['imag'][isoO['logAge']==7]+coefI),             isoO['rmag'][isoO['logAge']==7]+coefR,color='red',label='10Myr',ls='-.')

ax01.set_xlim(-0.49,3.49)
ax01.set_ylim(15,0.1)
#ax01.set_xlim(-2,4)
#ax01.set_ylim(20,0)
ax01.set_xlabel('r-i')
ax01.set_ylabel('M$_r$ [mag]')
ax01.set_title('OmegaCam without H$_{\\alpha}$')
ax01.legend()


ax10.minorticks_on()
ax10.get_xaxis().set_tick_params(direction='in', width=1, which='both', bottom=True, top=True)
ax10.get_yaxis().set_tick_params(direction='in', width=1, which='both', left=True, right=True)
ax10.set_title('coordinates')
sns.scatterplot(x=whole['ra'][cond],y=whole['dec'][cond],data=whole,s=5, alpha=0.4, label="field SOs", ax=ax10, color="black")
sns.scatterplot(x=dfo['ra'],y=dfo['dec'],data=dfo, alpha=0.7, ax=ax10,s=100, color="orange", label="clustered non-accretors")
sns.scatterplot(x=dfo2['ra'],y=dfo2['dec'],data=dfo2, alpha=0.7, ax=ax10,s=100, color="orange")

x_newra=dfo['ra'][Eq<-10]
x_newdec=dfo['dec'][Eq<-10]
x_newra1=dfo2['ra'][Eq<-10]
x_newdec1=dfo2['dec'][Eq<-10]
sns.scatterplot(x=x_newra[Eq>-20],y=x_newdec[Eq>-20],data=dfo,alpha=1,ax=ax10,s=150,label='Accretors ChaI >10$\AA$', marker="D", color="blue")
sns.scatterplot(x=x_newra1[Eq1>-20],y=x_newdec1[Eq1>-20],data=dfo2,alpha=1,ax=ax10,s=150, marker="D", color="blue")
#sns.scatterplot(x=x_newra[Eq>-20],y=x_newdec[Eq>-20],data=dfo,alpha=1,ax=ax10,s=150,label='blue Accretors >10$\AA$', marker="D", color="blue")
#sns.scatterplot(x=x_newra1[Eq1>-20],y=x_newdec1[Eq1>-20],data=dfo2,alpha=1,ax=ax10,s=150,label='green Accretors >10$\AA$', marker="D", color="green")

sns.scatterplot(x=dfo['ra'][Eq<-20],y=dfo['dec'][Eq<-20],data=dfo,alpha=1,ax=ax10,s=400,label='Accretors ChaI >20$\AA$', marker="*", color="blue")
sns.scatterplot(x=dfo2['ra'][Eq1<-20],y=dfo2['dec'][Eq1<-20],data=dfo2,alpha=1,ax=ax10,s=400, marker="*", color="blue")
#sns.scatterplot(x=dfo['ra'][Eq<-20],y=dfo['dec'][Eq<-20],data=dfo,alpha=1,ax=ax10,s=400,label='blue Accretors >20$\AA$', marker="*", color="blue")
#sns.scatterplot(x=dfo2['ra'][Eq1<-20],y=dfo2['dec'][Eq1<-20],data=dfo2,alpha=1,ax=ax10,s=400,label='green Accretors >20$\AA$', marker="*", color="green")

ax10.set_xlim(147,184.9)
ax10.set_ylim(-81.4,-73.4)
ax10.set_xlabel('ra [degree]')
ax10.set_ylabel('dec [degree]')
ax10.legend(loc=1)

plot0=bbox_inches=bbox_inches=fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('Ha.png', bbox_inches=plot0.expanded(0.85,0.9))
#fig.savefig('Ha G3.png', bbox_inches=plot0.expanded(0.85,0.9))



# In[103]:


fig, axes=plt.subplots(1,figsize=(12,12))
sns.set_context('paper',font_scale=2)

sns.scatterplot(x=ri0,y=-Eq,alpha=0.7, s=100, label='clustered non-accretors', color="orange")
sns.scatterplot(x=ri1,y=-Eq1,alpha=0.7, s=100, color="orange")

ri01=ri0[Eq>-20]
rHa01=rHa0[Eq>-20]
ri11=ri1[Eq1>-20]
rHa11=rHa1[Eq1>-20]

Eqn=-Eq

sns.scatterplot(x=ri01[-10>Eq],y=-Eq[-10>Eq],alpha=1,s=150, label='Accretors ChaI >10$\AA$', marker="D", color="blue")
sns.scatterplot(x=ri11[-10>Eq1],y=-Eq1[-10>Eq1],alpha=1,s=150, marker="D", color="blue")
#sns.scatterplot(x=ri01[-10>Eq],y=-Eq[-10>Eq],alpha=1,s=150, label='blue Accretors >10$\AA$', marker="D", color="blue")
#sns.scatterplot(x=ri11[-10>Eq1],y=-Eq1[-10>Eq1],alpha=1,s=150, label='green Accretors >10$\AA$', marker="D", color="green")

sns.scatterplot(x=ri0[Eq<-20],y=-Eq[Eq<-20],alpha=1,s=400, marker='*', label='Accretors ChaI >20$\AA$', color="blue")
sns.scatterplot(x=ri1[Eq1<-20],y=-Eq1[Eq1<-20],alpha=1,s=400, marker='*', color="blue")
#sns.scatterplot(x=ri0[Eq<-20],y=-Eq[Eq<-20],alpha=1,s=400, marker='*', label='blue Accretors >20$\AA$', color="blue")
#sns.scatterplot(x=ri1[Eq1<-20],y=-Eq1[Eq1<-20],alpha=1,s=400, marker='*',label='green Accretors >20$\AA$', color="green")

x_line=[0,3]
y_line=[20,20]
axes.plot(x_line, y_line, color="red", label='20$\AA$ limit' )

x_line1=[0,3]
y_line1=[10,10]
axes.plot(x_line1, y_line1, color="red", ls='--', label='10$\AA$ limit' )

axes.set_xlim(0.1,2.69)
axes.set_ylim(-10,199)
axes.set_xlabel('r-i')
axes.set_ylabel('EW(H$_{\\alpha}$) [$\AA$]')

axes.set_title('H$_{\\alpha}$-emission')
axes.legend()

plot0=bbox_inches=bbox_inches=fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('Ha norm.png', bbox_inches=plot0.expanded(1,0.9))
#fig.savefig('Ha norm G3.png', bbox_inches=plot0.expanded(1,0.9))


# #### print(rha_mod)

# In[107]:


Eq.count(<-10)


# In[21]:


#fig,axes =plt.subplots(1,2, figsize=(12,6))
#sns.set_context('paper',font_scale=2)


# In[22]:


dfno=dfo[(dfo['r']<90) & (dfo['i']<90) & (dfo['ha']<90) & (dfo['r']-dfo['i']<4) &        (dfo['r']-dfo['i']>-1) & (dfo['r']-dfo['ha']>-1) & (dfo['r']-dfo['ha']<4)]


# In[23]:


cond_whole=whole[(whole['r']<90) & (whole['i']<90) & (whole['ha']<90) & (whole['r']-whole['i']<4) &        (whole['r']-whole['i']>-1) & (whole['r']-whole['ha']>-1) & (whole['r']-whole['ha']<4) &        (whole['parallax']>4)]
x_w=cond_whole['r']-cond_whole['i']
y_w=cond_whole['r']-cond_whole['ha']


# In[24]:


#whole.info()


# In[25]:


x=dfno['r']-dfno['i']
y=dfno['r']-dfno['ha']
new_x=np.linspace(min(x),max(x),num=np.size(x))
coefs=np.polyfit(x,y,3)
new_line=np.polyval(coefs, new_x)

plt.scatter(x,y)
plt.scatter(new_x,new_line,c='g',marker='^', s=5)
plt.xlim(min(x)-0.00001,max(x)+0.00001)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show


# In[26]:


from scipy.optimize import curve_fit


# In[27]:


sns.scatterplot(x,y)
sns.lineplot(new_x,new_line,color='red')


# In[28]:


from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
import scipy.stats as stats


# In[29]:


g_init=models.Polynomial1D(3)
fit=fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(),sigma_clip,niter=3,                                      sigma=2.0)


# In[30]:


w_init=models.Polynomial1D(3)
fit=fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(),sigma_clip,niter=3,                                      sigma=2.0)


# In[31]:


fitted_model_w,filtered_data_w=fit(w_init,x_w,y_w)


# In[32]:


fitted_model,filtered_data=fit(g_init,x,y)


# In[33]:


#sns.scatterplot(x,y)
#sns.scatterplot(x[filtered_data],y[filtered_data])
#sns.lineplot(x,fitted_model(x))

sns.scatterplot(x,y)
sns.scatterplot(x[filtered_data],y[filtered_data])
nx = np.linspace(-0.2,3,100)
filt_plot=sns.lineplot(nx,fitted_model(nx))
filt_plot.set(xlim=(0,5))


# In[34]:


sns.scatterplot(x_w,y_w)
sns.scatterplot(x_w[filtered_data_w],y_w[filtered_data_w])
nx_w = np.linspace(-0.2,3,100)
sns.lineplot(nx_w,fitted_model_w(nx_w))


# In[35]:


X=np.array([nx,fitted_model(nx)])
#np.savetxt('datapath/Chaemeleon/fitted values/MYFIT.dat',X.T)

#ndf = pd.DataFrame(X.T,columns=['x','y'])
#ndf=ndf.sort_values(by=['x'])
#ndf.to_csv('datapath/Chaemeleon/fitted values/MYFIT.csv')


nx=pd.Series(fitted_model(x))
ny=pd.Series(fitted_model(y))
sel_stars=pd.DataFrame(columns=['nx','ny'])
sel_stars['nx']=nx
sel_stars['ny']=ny
#print(selected_stars)

hdf = pd.DataFrame(X.T,columns=['x','y'])
hdf=hdf.sort_values(by=['x'])
hdf.to_csv('datapath/Chaemeleon/fitted values/MYFIT.csv')

#test=pd.array(nx,ny)
#print(test)
#fit_y = pd.Series(fitted_model(x))
#ndf = pd.DataFrame(columns=['X', 'Y'])
#ndf['X'] = x
#ndf['Y'] = fit_y
#ndf.to_csv('MYFIT.csv')


# In[36]:


fig,axes=plt.subplots(1,2,figsize=(12,8))
sns.set_context('paper',font_scale=2)

axes[0].plot(hdf['x'],hdf['y']-corrHa, alpha=0.7, color='green')
axes[0].set_xlim(0,3)

print(hdf)


# In[37]:


print(fitted)


# In[38]:


fig, axes=plt.subplots(1,1, figsize=(12,8))
sns.set_context('paper',font_scale=2)

axes.scatter((dfo['r'])-(dfo['ha']),dfo['r'],color='red',label='No-acc')

axes.set_xlim(-2,2)
axes.set_ylim(22,9)
axes.set_xlabel('r$_0$-Halpha$_0$')
axes.set_ylabel('r$_0$')


# In[39]:


fig, axes = plt.subplots(1, 1, figsize=(9, 9))
sns.set_context('paper',font_scale=2)
hist(dfo['parallax'], bins=10, weights=dfo['parallax_error'], histtype='step', color='blue')
hist(dfo['parallax'], bins=10, weights=dfo['parallax_error'], alpha=0.3, color='blue', ec='black')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




