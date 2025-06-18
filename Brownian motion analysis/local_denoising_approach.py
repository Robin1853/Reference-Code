import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sympy.codegen.ast import continue_


###Define Grid
def Grid(space_borders, grid_step = 0.1):
    if space_borders is None:
        space_borders = [-1, 1]

    x_val = y_val = z_val = np.linspace(space_borders[0], space_borders[1], int((space_borders[1] - space_borders[0]) / grid_step) + 1)
    X, Y, Z = np.meshgrid(x_val, y_val, z_val)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    W = np.zeros_like(Z)

    return X, Y, Z, U, V, W

###Slice trajectory in steps
def trajectory_steps(signal):
    M, N, dim = signal.shape

    all_starts = []
    all_steps = []
    for i in range(M):
        starts = signal[i, :-1]
        steps = signal[i, 1:] - signal[i, :-1]
        all_starts.append(starts)
        all_steps.append(steps)
    all_starts = np.vstack(all_starts)
    all_steps = np.vstack(all_steps)

    return all_starts, all_steps

###define drift Neighborhood
def KDTree_neighbors(tree, local_point, steps, starts, radius):
    indices = tree.query_ball_point(local_point, r = radius)
    # if len(indices) == 0:
    #     return np.zeros(steps.shape[1]), [], [], []

    if len(indices) == 0:
        dim = steps.shape[1]
        return np.zeros(dim), 0.0

    neighbor_steps = steps[indices]
    neighbor_starts = starts[indices]

    return neighbor_steps, neighbor_starts

###drift field plot
def drift_field_plot(X, Y, Z, U, V, W):
    # magnitude = np.sqrt(U ** 2 + V ** 2 + W ** 2)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=False, color='darkblue', linewidth=0.5)
    # ax.quiver(X, Y, Z, U, V, W, length=0.15, normalize=True, cmap='plasma', linewidth=0.8, arrow_length_ratio=0.5, array = magnitude.flatten())
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title("Gaussian averaged local drifts")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()

###drift estimation
def drift_estimation(neighbor_starts, neighbor_steps, local_point, radius, dt, U,V, W, i, j, k):
    dists = np.linalg.norm(neighbor_starts - local_point, axis=1)
    weighting = np.exp(-(dists ** 2) / (2 * (radius / 2) ** 2))
    weight_normed = weighting / np.sum(weighting)
    weighted_drift = np.sum(neighbor_steps * weight_normed[:, None], axis=0) / dt
    U[i, j, k], V[i, j, k], W[i, j, k] = weighted_drift
    return U, V, W, weighted_drift

###diffusion field plot
def diffusion_field_plot(X, Y, Z, D_field):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=D_field.flatten(), cmap='plasma', s=30)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    fig.colorbar(sc, ax=ax, label='estimated diffusion')
    ax.set_title("Gaussian averaged local diffusion")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()

###First lokal drift estimation with KDTree
def local_estimator(signal, dt = 0.1, space_borders = None, grid_step = 0.1, radii_ratio = 1):
    X, Y, Z, U, V, W = Grid(space_borders=space_borders, grid_step=grid_step)
    D_field = np.zeros_like(X)

    radius = radii_ratio * np.sqrt(3 * grid_step ** 2)

    starts, steps = trajectory_steps(signal)

    tree = KDTree(starts)
    all_drifts = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                local_point = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])

                neighbor_steps, neighbor_starts = KDTree_neighbors(tree, local_point, steps, starts, radius)

                #drift estimation
                U, V, W, weighted_drift = drift_estimation(neighbor_starts, neighbor_steps, local_point, radius, dt, U, V, W, i, j, k)
                all_drifts.append(weighted_drift)

                #diffusion estimation
                noise = neighbor_steps - weighted_drift * dt
                squared_noise = np.sum(noise**2, axis=1)
                D_local = np.mean(squared_noise) / (2 * dt**2 * 3)
                D_field[i, j, k] = D_local

    all_drifts = np.array(all_drifts)
    drift_field_plot(X, Y, Z, U, V, W)
    drift_mean = np.linalg.norm(np.mean(all_drifts, axis = 0))
    drift_std = np.linalg.norm(np.std(all_drifts, axis = 0))
    if drift_std/drift_mean > 0.1:
        print(f"Global drift is inhomogeneous: {drift_mean:.4f} ± {drift_std:.4f}")
    else:
        print(f"Global drift appears homogeneous: {drift_mean:.4f} ± {drift_std:.4f}")

    diffusion_field_plot(X, Y, Z, D_field)
    D_mean = np.mean(D_field)
    D_std = np.std(D_field)
    if D_std/D_mean > 0.1:
        print(f"Diffusion is inhomogeneous: {D_mean:.4f} ± {D_std:.4f}")
    else:
        print(f"Diffusion is homogeneous: {D_mean:.4f} ± {D_std:.4f}")
    return np.array(all_drifts), U, V, W


###Denoising with wiener filter

###Meassuring accuracy of estimates
def score(true_drifts, est_drifts):
    mse = np.mean((true_drifts - est_drifts) ** 2)
    print(f"MSE: {mse:.4f}")
    return mse