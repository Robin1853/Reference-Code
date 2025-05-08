# main.py – entry point to run training and plotting

from train_loop import run_training_loop
from plot_utils import plot_training_results
import os

def main():
    data_source = "breastmnist" #"MNIST"  or "5_7_MNIST"
    # Current project directory
    project_dir = os.path.abspath(os.path.dirname(__file__))

    # Path to data directories
    data_dir = os.path.join(project_dir, "data")

    output_dir = f"./results_{data_source}"

    batch_size = 16
    train_size = 100
    num_epochs = 10
    num_seeds = 5

    run_training_loop(
        data_source=data_source,
        batch_size=batch_size,
        train_size=train_size,
        num_epochs=num_epochs,
        num_seeds=num_seeds,
        data_dir=data_dir,
        output_dir=output_dir,
    )

    # Plot for all QNodes in list
    for qid in range(16):  # extended range to 16 QNodes
        plot_training_results(qid, output_dir, num_epochs)


if __name__ == "__main__":
    main()