# train_loop.py – loop over qnodes and seeds with saving logic

import torch
import os
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import pennylane as qml
from quantum_classifier import load_dataset, HybridQuantumModel, train_one_epoch, evaluate
from qnodes import qnodes, WEIGHT_SHAPES


def run_training_loop(
    data_source: str,
    batch_size: int,
    train_size: int,
    num_epochs: int,
    num_seeds: int,
    data_dir: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    base_seeds = torch.randint(0, 100, (num_seeds,), dtype=torch.int)

    for qid, qnode in enumerate(qnodes):
        print(f"Training QNode {qid}")

        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes=WEIGHT_SHAPES, init_method={"weights": nn.init.normal_})

        all_train_losses = []
        all_val_losses = []
        all_train_accs = []
        all_val_accs = []

        for seed in base_seeds:
            torch.manual_seed(seed.item())
            model = HybridQuantumModel(qlayer)

            train_loader, val_loader = load_dataset(data_source, batch_size, train_size, data_dir)

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            epoch_train_losses = []
            epoch_val_losses = []
            epoch_train_accs = []
            epoch_val_accs = []

            for epoch in range(num_epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc = evaluate(model, val_loader, criterion)

                print(f"Seed {seed.item()} | Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_loss)
                epoch_train_accs.append(train_acc)
                epoch_val_accs.append(val_acc)

            all_train_losses.append(epoch_train_losses)
            all_val_losses.append(epoch_val_losses)
            all_train_accs.append(epoch_train_accs)
            all_val_accs.append(epoch_val_accs)

        # Save results for current QNode
        np.save(os.path.join(output_dir, f"train_loss_q{qid}.npy"), np.array(all_train_losses))
        np.save(os.path.join(output_dir, f"val_loss_q{qid}.npy"), np.array(all_val_losses))
        np.save(os.path.join(output_dir, f"train_acc_q{qid}.npy"), np.array(all_train_accs))
        np.save(os.path.join(output_dir, f"val_acc_q{qid}.npy"), np.array(all_val_accs))
