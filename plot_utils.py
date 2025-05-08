# plot_utils.py – plotting training/validation metrics

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_results(qid, output_dir, num_epochs):
    """Plot training and validation loss/accuracy with confidence intervals."""
    train_loss = np.load(os.path.join(output_dir, f"train_loss_q{qid}.npy"))
    val_loss = np.load(os.path.join(output_dir, f"val_loss_q{qid}.npy"))
    train_acc = np.load(os.path.join(output_dir, f"train_acc_q{qid}.npy"))
    val_acc = np.load(os.path.join(output_dir, f"val_acc_q{qid}.npy"))

    x = np.arange(num_epochs)

    def plot_metric(ax, train_metric, val_metric, ylabel):
        train_avg = np.mean(train_metric, axis=0)
        train_std = np.std(train_metric, axis=0)
        val_avg = np.mean(val_metric, axis=0)
        val_std = np.std(val_metric, axis=0)

        ax.fill_between(x, train_avg - train_std, train_avg + train_std, alpha=0.2, label='Train ±1σ')
        ax.fill_between(x, val_avg - val_std, val_avg + val_std, alpha=0.2, label='Val ±1σ')
        ax.plot(x, train_avg, label='Train Avg')
        ax.plot(x, val_avg, label='Val Avg')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"QNode {qid} Training Metrics")
    plot_metric(ax1, train_loss, val_loss, "Loss")
    plot_metric(ax2, train_acc, val_acc, "Accuracy")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"metrics_q{qid}.png"))
    plt.close()
