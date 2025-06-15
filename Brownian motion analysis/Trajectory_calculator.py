from signal import signal

import numpy as np
import matplotlib.pyplot as plt


def trajectory(drift_func, D, dim, dt, N, M):
    x = np.zeros((M, N, dim))

    ###Loop over multiple trajectories with multiple starting positions
    for j in range(M):
        x[j, 0] = np.random.uniform(-1, 1, size = dim)
        ###Simulation
        for i in range(1, N):
            v_drift = drift_func(x[j, i-1])
            noise = np.random.normal(0, D, dim)
            x[j, i] = x[j, i-1] + v_drift * dt + np.sqrt(2*D*dt)*noise

    ###3D Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ###First Trajectory
    ax.plot(x[0, :,0], x[0, :,1], x[0, :,2], lw=1)
    ax.scatter(x[0, 0, 0], x[0, 0, 1], x[0, 0, 2], color = 'r', s = 50)
    ax.scatter(x[0, -1, 0], x[0, -1, 1], x[0, -1, 2], color = 'g', s = 50)
    #ax.quiver(x[0, 0], x[0, 1], x[0, 2], v_drift[0], v_drift[1], v_drift[2], color = 'r')      ###for homogeneous drift vector
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Brownian motion 3D')
    plt.grid(True)
    #plt.show()

    return x

