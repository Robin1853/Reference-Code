import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#####Drift Dynamic###
###original drift field
def drift_dynamics(drift_func, step, plot_range=(-1, 1)):
    #grid points
    x_vals = y_vals = z_vals = np.arange(plot_range[0], plot_range[1]+step, step)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    true_drifts = []

    ###Driftfield
    U, V, W = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            for k in range(Z.shape[0]):
                pos = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                drift = drift_func(pos)
                U[i, j, k], V[i, j, k], W[i, j, k] = drift
                true_drifts.append(drift)

    ###Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(X, Y, Z, U, V, W, length = 0.1, normalize=False, color='darkblue', linewidth=0.5)
    ax.set_xlim(plot_range)
    ax.set_ylim(plot_range)
    ax.set_zlim(plot_range)
    ax.set_title('Drift Dynamics')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return np.array(true_drifts)


