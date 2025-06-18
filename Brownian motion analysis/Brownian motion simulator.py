import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

from Drift_field import drift_dynamics
from Trajectory_calculator import trajectory
import lowpassfilter_approach as lfa
import local_denoising_approach as lda

###simulation parameters
dim = 3                             #dimension
T = 10.0                            #overall time
dt = 0.01                           #time steps
N = int(T/dt)                       #number of time steps
D = 0.1                             #diffusion constant
M = 1000                             #number of trajectories
grid_step = 0.4                     #steps between local evaluation centers (along all axes)
plot_range = [-1, 1]                #area of concern along x, y and z axis
drift_test_point = (2,2,2)          #Point of lokal drift test

###lowpass filter denoising approach parameters
radius = 0.27 #0.27                       #radius of local trajectories for drift estimation
ab_ratio = 0.012 #0.01241 -0.01242                     #Sets the alpha/beta ratio of the denoising loss function: alpha * d + beta * c
                                    # where d(ifference) is the difference between the denoised trajectory and original and c(urvature) is the roughness of the denoised trajectory
###local denoising approach  parameters
radii_ratio = 1                     #factor for evaluation distance around grid points (should be ten times grid_step)

def drift_func(x):
    ###linear drift
    return np.array([2.2, -0.76, 1.18])     #homogeneous 3D drift
    ### Ornstein-Uhlenbeck process
    #k = 1.0 # constant for inhomogeneous Ornstein-Uhlenbeck process
    #return -k*x
    ###rotation field
    #return np.array([-x[1], x[0], 0.0])
    ##rotation with wave field
    #return np.array([-x[1], x[0], np.sin(np.pi*np.sqrt(x[0]**2 + x[1]**2))])

#fix random seed
np.random.seed(0)

#Create Trjectories and original drift field
signal = trajectory(drift_func, D, dim, dt, N, M)
true_drifts, x_drift, y_drift, z_drift = drift_dynamics(drift_func, grid_step, plot_range)
print(x_drift[drift_test_point], y_drift[drift_test_point], z_drift[drift_test_point])

#####Unsupervised approach estimating local drift from denoised trajectories (lowpass filter)
###Estimating local diffusion from reverse filter
#lfa.plot_welch_spectra(signal[0], dt)
#lfa.local_filtered_estimator(signal, true_drifts, grid_step, radius, ab_ratio, dt)

####Unsupervised approach estimating local drift from local gaussian weighted movement
est_drift, U, V, W =lda.local_drift_estimator(signal, dt = dt, space_borders=plot_range, grid_step=grid_step, radii_ratio=radii_ratio)
print([np.mean(U), np.mean(V), np.mean(W)], lda.score(true_drifts, est_drift))


plt.show()