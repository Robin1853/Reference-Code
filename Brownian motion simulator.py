import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

from Drift_field import drift_dynamics
from Trajectory_calculator import trajectory
#from Welch_denoising import plot_welch_spectra, lowpass_filter_with_reflection, welch_smoothed, enforce_endpoints
import local_filtered_estimator as lfe

###simulation parameters
dim = 3                             #dimension
T = 10.0                            #overall time
dt = 0.01                           #time steps
N = int(T/dt)                       #number of time steps
D = 0.1                             #diffusion constant
M = 100                             #number of trajectories

###Unsupervised local approach parameters
radius = 0.27                       #radius of local trajectories for drift estimation
ab_ratio = 0.02                  #Sets the alpha/beta ratio of the denoising loss function: alpha * d + beta * c
                                    # where d(ifference) is the difference between the denoised trajectory and original and c(urvature) is the roughness of the denoised trajectory

grid_step = 0.4
plot_range = [-1, 1]

def drift_func(x):
    ###linear drift
    #return np.array([0.2, -0.26, 0.18])     #homogeneous 3D drift
    ### Ornstein-Uhlenbeck process
    #k = 1.0 # constant for inhomogeneous Ornstein-Uhlenbeck process
    #return -k*x
    ###rotation field
    #return np.array([-x[1], x[0], 0.0])
    ##rotation with wave field
    return np.array([-x[1], x[0], np.sin(np.pi*np.sqrt(x[0]**2 + x[1]**2))])

np.random.seed(0)
#print(trajectory(drift_func, D, dim, dt, N).shape)
signal = trajectory(drift_func, D, dim, dt, N, M)
true_drifts = drift_dynamics(drift_func, grid_step, plot_range)
lfe.plot_welch_spectra(signal[0], dt)

#####Unsupervised statistical approach
lfe.local_filtered_estimator(signal, true_drifts, grid_step, radius, ab_ratio)

#plt.show()