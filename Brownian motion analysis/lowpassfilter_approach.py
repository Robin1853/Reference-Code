from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def plot_welch_spectra(signal, dt, dim_labels=('x', 'y', 'z')):
    """
    signal: np.ndarray, shape (T, 3)
    dt: float, timestep
    dim_labels: axis naming of dimensions
    """
    T, dim = signal.shape
    assert dim == 3, "3 dimesnions for signal requested"

    fs = 1 / dt  # Sampling Rate
    nperseg = min(256, T)  # Segmentlength

    plt.figure(figsize=(10, 6))

    for d in range(dim):
        f, Pxx = welch(signal[:, d], fs=fs, nperseg=nperseg)
        plt.loglog(f, Pxx, label=f'{dim_labels[d]}-Axis')

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [Signal²/Hz]")
    plt.title("Power Spectrum with Welch-method")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    #plt.show()


def lowpass_filter_with_reflection(signal, cutoff_ratio=0.1):

    M, T, D = signal.shape
    all_smoothed = []

    for i in range(M):
        single_signal = signal[i]
        smoothed = np.zeros_like(single_signal)

        for d in range(D):
            x = single_signal[:, d]

            # Reflection
            x_pad = np.concatenate([x[::-1], x, x[::-1]])

            # FFT & low-pass
            Xf = np.fft.rfft(x_pad)
            cutoff = int(len(Xf) * cutoff_ratio)
            Xf[cutoff:] = 0

            # Backtransformation
            x_smooth = np.fft.irfft(Xf, n=len(x_pad))

            # Cut out the middle part
            smoothed[:, d] = x_smooth[T:2 * T]
        all_smoothed.append(smoothed)

    return np.array(all_smoothed)


def compute_curvature_penalty(smoothed):
    """Calculate curvature (roughness) of trajectory"""
    return np.sum(np.linalg.norm(smoothed[2:] - 2 * smoothed[1:-1] + smoothed[:-2], axis=1) ** 2)

def combined_score(traj, smoothed, alpha=1.0, beta=1.0):
    """Combined Score of difference to original trajectory and roughness"""
    scores = []
    for i in range(traj.shape[0]):
        deviation = np.mean(np.linalg.norm(traj[i] - smoothed[i], axis=1))
        # deviation = np.sum(np.linalg.norm(traj[i] - smoothed[i], axis=1))
        curvature = compute_curvature_penalty(smoothed[i])
        scores.append(alpha * deviation + beta * curvature)

    return np.mean(scores)

def denoising_score(cutoff_values, combined_scores):
    # Plot: Score vs. Cutoff-Ratio
    plt.figure(figsize=(8, 5))
    plt.plot(cutoff_values, combined_scores, marker='o')
    plt.xlabel("Cutoff Ratio")
    plt.ylabel("Combined Score (Difference + Roughness)")
    plt.title("Optimisation of denoising")
    plt.grid(True)
    plt.tight_layout()

###KDTree function for local drift estimation of multiple trajectories
def KDTree_drift(local_point, tree, starts, steps, radius, dt):
    indices = tree.query_ball_point(local_point, r = radius)
    if len(indices) == 0:
        return np.zeros(steps.shape[1]), [], [], []

    neighbor_steps = steps[indices]
    neighbor_positions = starts[indices]

    dists = np.linalg.norm(neighbor_positions - local_point, axis=1)
    weights = np.exp(- (dists**2) / (2 * (radius/2)**2))
    weights /= np.sum(weights)


    weighted_drift = np.sum(neighbor_steps * weights[:, None], axis=0)/dt
    return weighted_drift, neighbor_steps, neighbor_positions, weights

###estimated local drift field
def plot_drift_field(tree, starts, steps, dt, radius=0.3, step=0.2):
    # Create 3D-Gridpoints between [-1,1]³
    x_vals = y_vals = z_vals = np.arange(-1, 1 + step, step)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    est_drifts = []

    # Arrays for drift vectors
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    W = np.zeros_like(Z)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                query = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                drift, _, _, _ = KDTree_drift(query, tree, starts, steps, radius, dt)
                U[i, j, k], V[i, j, k], W[i, j, k] = drift
                est_drifts.append(drift)


    magnitude = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    # Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=False, color='darkblue', linewidth=0.5)
    # ax.quiver(X, Y, Z, U, V, W, length=0.15, normalize=True, cmap='plasma', linewidth=0.8, arrow_length_ratio=0.5, array = magnitude.flatten())
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title("Driftfield estimated with KDTree")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()

    return np.array(est_drifts)


def drift_estimation_error(true_drifts, est_drifts):
    mse = np.mean((true_drifts - est_drifts) ** 2)
    return mse

###smoothed trajectory
def denoised_trajectory(traj, best_smoothed, best_cutoff):
    all_positions = []
    all_steps = []
    M, N, dim = traj.shape

    for i in range(M):
        steps = traj[i, 1:] - traj[i, :-1]
        starts = traj[i, :-1]

        all_positions.append(starts)
        all_steps.append(steps)

    ###Plot first denoised trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[0, :, 0], traj[0, :, 1], traj[0, :, 2], lw=1, label="Original (noisy)", color='gray', alpha=0.5)
    ax.plot(best_smoothed[0, :, 0], best_smoothed[0, :, 1], best_smoothed[0, :, 2], lw=2,
            label=f"Denoised (cutoff = {best_cutoff:.2f})", color='blue')
    ax.scatter(traj[0, 0, 0], traj[0, 0, 1], traj[0, 0, 2], color='red', s=50, label="Start")
    ax.scatter(traj[0, -1, 0], traj[0, -1, 1], traj[0, -1, 2], color='green', s=50, label="End")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D-Trajektory: Original vs. Optimally denoised')
    ax.legend()
    plt.tight_layout()

    return np.vstack(all_positions), np.vstack(all_steps)

###estimate diffusion from reverse filtering
def diffusion_estimation(signal, smoothed, dt):
    D_vals = []
    M = signal.shape[0]
    for i in range(M):
        noise = signal[i, 1:] - smoothed[i, 1:]
        squared = np.sum(noise ** 2, axis=1)
        D_i = np.mean(squared) / (2 * dt)
        D_vals.append(D_i)

    D_vals = np.array(D_vals)
    mean_D = np.mean(D_vals)
    std_D = np.std(D_vals)

    #Homogeneity classification
    if std_D / mean_D < 0.1:
        print(f"Diffusion appears homogeneous: D = {mean_D:.4f} ± {std_D:.4f}")
    else:
        print(f"Diffusion is probably inhomogeneous: D = {mean_D:.4f} ± {std_D:.4f}")

    #Histogram plot
    plt.figure(figsize=(8, 4))
    plt.hist(D_vals, bins=15, color='skyblue', edgecolor='k')
    plt.axvline(mean_D, color='r', linestyle='--', label=f"Mean D = {mean_D:.3f}")
    plt.xlabel("Estimated D per trajectory")
    plt.ylabel("Frequency")
    plt.title("Histogram of Diffusion Estimates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return mean_D, std_D, D_vals

#####Unsupervised statistical approach
def local_filtered_estimator(signal, true_drifts, grid_step, radius, ab_ratio, dt):
    ###Cutoff optimization
    # Parameter
    cutoff_values = np.linspace(0.01, 1.0, 10)
    combined_scores = []
    smoothed_versions = []

    # Loop over all Cutoff-Values
    for cutoff in cutoff_values:
        smoothed = lowpass_filter_with_reflection(signal, cutoff_ratio=cutoff)
        score = combined_score(signal, smoothed, alpha=ab_ratio, beta=1.0)
        combined_scores.append(score)
        smoothed_versions.append(smoothed)

    # Find best result
    best_idx = np.argmin(combined_scores)
    best_smoothed = smoothed_versions[best_idx]
    best_cutoff = cutoff_values[best_idx]

    ###Denoised results
    denoising_score(cutoff_values, combined_scores)

    ###local drift estimation
    starts, steps = denoised_trajectory(signal, best_smoothed, best_cutoff)
    tree = KDTree(starts)

    est_drifts = plot_drift_field(tree, starts, steps, dt, radius, grid_step)
    diffusion_estimation(signal, best_smoothed, dt)
    print(f"Mean squared error of drift field: {drift_estimation_error(true_drifts, est_drifts):.3f}")
    print(f"Best Cutoff: {best_cutoff:.3f}")
    return est_drifts