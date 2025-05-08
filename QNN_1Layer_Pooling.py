Masterthesis/.gitkeep

#Imports
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from math import pi

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

import pennylane as qml

import timer

# Normalization of inputs to qubits to 0 - 2pi
def norm01(Tensor):
    normed_tensor = Tensor / 255 * (2 * pi)
    return normed_tensor

def norm02(Tensor):
    normed_tensor = Tensor * (2 * pi)
    return normed_tensor

# Current project directory
project_dir = os.path.abspath(os.path.dirname(__file__))

# Path to data directories
data_dir = os.path.join(project_dir, "data")

#Loading and normalizing choosen dataset according to batchsize & trainsize
def data_loading(datasource, batchsize, trainsize):
    if datasource == "MNIST":
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                              download=True,
                                              transform=transforms.Compose([torchvision.transforms.ToTensor(), norm02]))
        testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                             download=True,
                                             transform=transforms.Compose([torchvision.transforms.ToTensor(), norm02]))

        trainload = torch.utils.data.Subset(trainset, range(trainsize))
        testload = data_utils.Subset(testset, range(int(trainsize / 6)))

        trainloader = torch.utils.data.DataLoader(trainload, batch_size=batchsize, shuffle=True)
        testloader = torch.utils.data.DataLoader(testload, batch_size=batchsize, shuffle=False)

    elif datasource == "breastmnist":

        trainset = np.load(os.path.join(data_dir, 'breastmnist', 'train_images.npy'))
        # # trainset_at_eval = np.load('C:\Users\FaierR\models\data\breastmnist/val_images.npy')
        testset = np.load(os.path.join(data_dir, 'breastmnist', 'test_images.npy'))

        trainlab = np.load(os.path.join(data_dir, 'breastmnist', 'train_labels.npy'))
        # # trainlab_at_eval = np.load('C:\Users\FaierR\models\data\breastmnist/val_labels.npy')
        testlab = np.load(os.path.join(data_dir, 'breastmnist', 'test_labels.npy'))

        trainset = torch.Tensor(norm01(trainset))
        trainlab = torch.Tensor(trainlab)
        testset = torch.Tensor(norm01(testset))
        testlab = torch.Tensor(testlab)

        trainset = torch.utils.data.TensorDataset(trainset, trainlab)
        testset = torch.utils.data.TensorDataset(testset, testlab)

        trainload = torch.utils.data.Subset(trainset, range(trainsize))
        testload = data_utils.Subset(testset, range(int(trainsize / 6)))

        trainloader = torch.utils.data.DataLoader(trainload, batch_size=batchsize, shuffle=True)
        testloader = torch.utils.data.DataLoader(testload, batch_size=batchsize, shuffle=False)

    elif datasource == "5_7_MNIST":
        trainset = np.load(os.path.join(data_dir, '5_7_MNIST_reduced', 'Resizing_trainset_5_7_MNIST.npy'))
        testset = np.load(os.path.join(data_dir, '5_7_MNIST_reduced', 'Resizing_testset_5_7_MNIST.npy'))
        trainlab = np.load(os.path.join(data_dir, '5_7_MNIST_reduced', 'trainlab_5_7_MNIST.npy'))
        testlab = np.load(os.path.join(data_dir, '5_7_MNIST_reduced', 'testlab_5_7_MNIST.npy'))

        trainset = torch.Tensor(norm02(trainset))
        trainlab = torch.Tensor(trainlab)
        testset = torch.Tensor(norm02(testset))
        testlab = torch.Tensor(testlab)

        trainset = torch.utils.data.TensorDataset(trainset, trainlab)
        testset = torch.utils.data.TensorDataset(testset, testlab)

        trainload = torch.utils.data.Subset(trainset, range(trainsize))
        testload = torch.utils.data.Subset(testset, range(int(trainsize / 6)))

    else:
        sys.exit("Error: no source given")

    return trainload, testload

#Defining 16 different Quantum circuits for simulation
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def qnode1(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.CNOT(wires=[1, 0])
    qml.RX(weights[1][0], wires=0)
    qml.CNOT(wires=[2, 0])
    qml.RX(weights[1][1], wires=0)
    qml.CNOT(wires=[3, 0])
    qml.RX(weights[1][2], wires=0)

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode2(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.Toffoli(wires=[2, 1, 0])
    qml.RX(weights[1][0], wires=0)
    qml.Toffoli(wires=[3, 2, 0])
    qml.RX(weights[1][1], wires=0)
    qml.Toffoli(wires=[1, 3, 0])
    qml.RX(weights[1][2], wires=0)

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode3(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.MultiControlledX(wires=[3, 2, 1, 0])
    qml.RX(weights[1][0], wires=0)

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode4(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.RX(weights[0][0], wires=0)
    qml.RX(weights[0][1], wires=1)
    qml.RX(weights[0][2], wires=2)
    qml.RX(weights[0][3], wires=3)

    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode5(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.RX(weights[0][0], wires=0)
    qml.RX(weights[0][1], wires=1)
    qml.RX(weights[0][2], wires=2)
    qml.RX(weights[0][3], wires=3)

    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode6(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.RX(weights[0][0], wires=0)
    qml.RX(weights[0][1], wires=1)
    qml.RX(weights[0][2], wires=2)
    qml.RX(weights[0][3], wires=3)

    qml.MultiControlledX(wires=[3, 2, 1, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode7(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode8(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode9(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.MultiControlledX(wires=[3, 2, 1, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode10(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])

    qml.MultiControlledX(wires=[3, 2, 1, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode11(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])

    qml.MultiControlledX(wires=[3, 2, 1, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode12(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])

    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode13(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])

    qml.Toffoli(wires=[2, 1, 0])
    qml.Toffoli(wires=[3, 2, 0])
    qml.Toffoli(wires=[1, 3, 0])

    qml.MultiControlledX(wires=[3, 2, 1, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode14(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.CRZ(weights[0][0], wires=[1, 0])
    qml.PauliX(wires=[1])
    qml.CRX(weights[0][1], wires=[1, 0])

    qml.CRZ(weights[0][0], wires=[3, 2])
    qml.PauliX(wires=[3])
    qml.CRX(weights[0][1], wires=[3, 2])

    qml.CRZ(weights[0][0], wires=[2, 0])
    qml.PauliX(wires=[2])
    qml.CRX(weights[0][1], wires=[2, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode15(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.RY(weights[0][0], wires=[0])
    qml.RY(weights[0][1], wires=[1])
    qml.CNOT(wires=[1, 0])
    qml.CRZ(weights[0][2], wires=[1, 0])
    qml.PauliX(wires=[1])
    qml.CRX(weights[0][3], wires=[1, 0])

    qml.RY(weights[0][0], wires=[2])
    qml.RY(weights[0][1], wires=[3])
    qml.CNOT(wires=[3, 2])
    qml.CRZ(weights[0][2], wires=[3, 2])
    qml.PauliX(wires=[3])
    qml.CRX(weights[0][3], wires=[3, 2])

    qml.RY(weights[0][0], wires=[0])
    qml.RY(weights[0][1], wires=[2])
    qml.CNOT(wires=[2, 0])
    qml.CRZ(weights[0][2], wires=[2, 0])
    qml.PauliX(wires=[2])
    qml.CRX(weights[0][3], wires=[2, 0])

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev, interface="torch")
def qnode16(inputs, weights):
    qml.AngleEmbedding(inputs, range(n_qubits))

    qml.Hadamard(wires=[0])
    qml.Hadamard(wires=[1])
    qml.CZ(wires=[1, 0])
    qml.RX(weights[0][0], wires=[0])
    qml.RX(weights[0][1], wires=[1])
    qml.CRZ(weights[0][2], wires=[1, 0])
    qml.PauliX(wires=[1])
    qml.CRX(weights[0][3], wires=[1, 0])

    qml.Hadamard(wires=[2])
    qml.Hadamard(wires=[3])
    qml.CZ(wires=[3, 2])
    qml.RX(weights[0][0], wires=[2])
    qml.RX(weights[0][1], wires=[3])
    qml.CRZ(weights[0][2], wires=[3, 2])
    qml.PauliX(wires=[3])
    qml.CRX(weights[0][3], wires=[3, 2])

    qml.Hadamard(wires=[0])
    qml.Hadamard(wires=[2])
    qml.CZ(wires=[2, 0])
    qml.RX(weights[0][0], wires=[0])
    qml.RX(weights[0][1], wires=[2])
    qml.CRZ(weights[0][2], wires=[2, 0])
    qml.PauliX(wires=[2])
    qml.CRX(weights[0][3], wires=[2, 0])

    return qml.expval(qml.PauliZ(wires=0))


n_layers = 1
weight_shapes = {"weights": (2, 4)}


# In[6]:

#Reshaping images for input to Quantum Circuits
def filtering(x):
    pic_bat = x
    bat = pic_bat.size()[0]
    pic_bat = pic_bat.reshape(bat, 28, 28)
    pic_bat = pic_bat.reshape(bat, 14, 2, 14, 2)
    pic_bat = np.swapaxes(pic_bat, 2, 3)
    pic_bat = pic_bat.reshape(bat, 14, 14, 4)
    ###5_7_MNIST (14x14)
    # pic_bat = pic_bat.reshape(bat, 14, 14)
    # pic_bat = pic_bat.reshape(bat, 7, 2, 7, 2)
    # pic_bat = np.swapaxes(pic_bat, 2, 3)
    # pic_bat = pic_bat.reshape(bat, 7, 7, 4)

    x = pic_bat
    return x

#Defining overall network architecture
class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.qlayer_1 = qlayer
        self.flat = nn.Flatten()
        ###in_features 196 for 5_7_MNIST: 49
        self.fc1 = nn.Linear(7 * 7 * 4, 30)
        # self.clayer_1 = nn.Linear(784, 30)
        # MNIST(14x14, 32):
        # self.fc3 = nn.Linear(6272, 30)
        # Cifar10:
        # self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        # self.drop3 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        y = x.size()[0]
        x = filtering(x)

        x = self.qlayer_1(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        return x

#Initialization of weights for reproducibility
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)


def initialization(j):
    print("Seed:", j)
    model.apply(weights_init)

#Definition of Parameter Training with Adam optimizer
def train(net_model, data_loader):
    optimizer = optim.Adam(net_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    global loss

    step_loss = []
    count = 0
    t_acc = 0
    train_acc = []

    for data, labels in tqdm.tqdm(data_loader):
        optimizer.zero_grad()
        labels = labels.long()
        pred = net_model(data)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        step_loss.append(loss.item())
        t_acc += (torch.argmax(pred, 1) == labels).float().sum()
        train_acc.append(t_acc)
        count += len(labels)
    t_acc /= count

    print(f'Accuracy: {t_acc * 100:.2f}%, Loss: {loss.item():.4f}')
    # return np.array(step_loss).mean(), np.array(train_acc).mean()
    return torch.mean(torch.Tensor(step_loss)).numpy(), torch.mean(torch.tensor(train_acc)).numpy()

#Definition of Parameter testing
def test(model, data_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    global val_loss

    acc = 0
    count = 0
    val_step_loss = []
    val_acc = []

    with torch.no_grad():
        for data, labels in tqdm.tqdm(data_loader):
            labels = labels.long()
            pred = model(data)
            val_loss = criterion(pred, labels)

            val_step_loss.append(val_loss.item())
            acc += (torch.argmax(pred, 1) == labels).float().sum()
            val_acc.append(acc)
            count += len(labels)
    acc /= count
    print(f'Val Accuracy: {acc * 100:.2f}%, Validation Loss: {val_loss.item():.4f}')
    # return np.array(val_step_loss).mean(), np.array(val_acc).mean()
    return torch.mean(torch.Tensor(val_step_loss)).numpy(), torch.mean(torch.tensor(val_acc)).numpy()

#Set base seed
torch.manual_seed(0)
#Generate different seeds for parameter seeding
base_seed = torch.randint(0, 100, (20,), dtype=torch.int)

# ##############################################
# ##input parameters to change##################
# ##############################################
batch_size = 16

train_size = 100  # 5_7_MNIST: 400 #MNIST: 10000

data_source = "breastmnist"  # "5_7_MNIST" #"MNIST" #"MNIST"

n_epochs = 10
n_seeds = 10

identity = "mixed_poolings"

directory: str = f"/ifxhome/faierr/projects/qml-cybrex/models/Paper/Pooling_paper/{data_source}_{identity}/"
os.makedirs(os.path.dirname(directory), exist_ok=True)

#Quantum Circuits to be used in different training runs
qnodes = [qnode1, qnode2, qnode3, qnode4, qnode5, qnode6, qnode7, qnode8, qnode9, qnode10, qnode11, qnode12, qnode13,
          qnode14, qnode15, qnode16]
# qnodes = [qnode7, qnode8, qnode9]

#Main code for running trainings of different Quantum Circuit networks with different parameter
#initialization seeds (n_seeds) and epochs (n_epochs) & plotting of training results with error, etc.

timer.tic()
for qname, qnode in enumerate(qnodes):  # [9:], start=10):

    #Drawing of used Quantum Circuit (has to be fitted for each circuit individually)
    qml.draw_mpl(qnode, decimals=2)(inputs=['inp1', 'inp2', 'inp3', 'inp4'],
                                    weights=[['w11', 'w12', 'w13', 'w14'], ['w21', 'w22', 'w23', 'w24']])
    # qml.draw_mpl(qnode2, decimals=2)(inputs=['inp1', 'inp2', 'inp3', 'inp4'], weights=[['w11', 'w12', 'w13', 'w14'],['w21', 'w22', 'w23', 'w24']])
    # weights=[['w11', 'w12', 'w13', 'w14'],['w21', 'w22', 'w23', 'w24']])
    # weights=[['w1', 'w2', 'w3', 'w4']])
    # weights=[[['w111', 'w112', 'w113'],['w121', 'w122', 'w123'],['w131', 'w132', 'w133'],['w141', 'w142', 'w143']],[['w211', 'w212', 'w213'],['w221', 'w222', 'w223'],['w231', 'w232', 'w233'],['w241', 'w242', 'w243']]])

    # qml.draw_mpl(qnode2, decimals=2)(inputs=['inp1', 'inp2', 'inp3', 'inp4'], weights=[['w11', 'w12', 'w13', 'w14'],['w21', 'w22', 'w23', 'w24']])

    plt.savefig(f"{directory}{data_source}Circuit_{qname}.png")

    size = train_size

    parameter_in = []
    parameter_out = []
    plot_data_sum = []
    plot_data = []

    trainingEpoch_loss_sum = []
    validationEpoch_loss_sum = []
    train_accuracy_sum = []
    val_accuracy_sum = []

    if __name__ == '__main__':
        for seed in range(n_seeds):

            trainingEpoch_loss = []
            validationEpoch_loss = []
            train_accuracy = []
            val_accuracy = []

            torch.manual_seed(base_seed[seed])
            print(torch.manual_seed(base_seed[seed]))

            init_method = {"weights": nn.init.normal_}
            # qlayer = qml.qnn.TorchLayer(qnode2, weight_shapes={"weights": (1, 4)}, init_method=init_method)
            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes, init_method=init_method)
            model = HybridModel()
            initialization(seed)
            params_i = []
            params_o = []
            [params_i.append(parameter) for parameter in model.parameters()]

            dataset, testset = data_loading(data_source, batch_size, size)

            for epoch in range(n_epochs):
                print(f"Qnode: {str(qname)} Seed:{seed} Epoch {epoch + 1}/{n_epochs}")
                #   ###Training###
                model.train()
                data_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size)
                mean_step_loss, mean_train_acc = train(model, data_loader)

                trainingEpoch_loss.append(mean_step_loss)
                train_accuracy.append(mean_train_acc)

                ###Testing
                model.eval()
                data_loader = DataLoader(
                    dataset=testset,
                    batch_size=batch_size)
                mean_val_step_loss, mean_val_train_acc = test(model, data_loader)

                validationEpoch_loss.append(mean_val_step_loss)
                val_accuracy.append(mean_val_train_acc)

                # torch.save(model.state_dict(), f'{directory}model_save_{seed}_{bs}.pth')
                timer.tac()

            [params_o.append(parameter) for parameter in model.parameters()]

            # plotting_training_cycle(n_epochs)
            trainingEpoch_loss_sum.append(trainingEpoch_loss)
            validationEpoch_loss_sum.append(validationEpoch_loss)
            train_accuracy_sum.append(train_accuracy)
            val_accuracy_sum.append(val_accuracy)

            plot_data = np.array([trainingEpoch_loss,
                                  validationEpoch_loss,
                                  train_accuracy,
                                  val_accuracy])

            plot_data = np.array(plot_data)
            # torch.save(params_i, f"{directory}{size}_param_in_save_{identity}_{bs}_block{seed}.pt")
            # torch.save(params_o, f"{directory}{size}_param_out_save_{identity}_{bs}_block{seed}.pt")
            # np.save(f"{directory}{size}_Plot_data_raw_{identity}_{bs}_block{seed}.npy", plot_data)

            parameter_in.append(params_i)
            parameter_out.append(params_o)

            print(plot_data.shape)

        plot_data_sum = np.array([trainingEpoch_loss_sum,
                                  validationEpoch_loss_sum,
                                  train_accuracy_sum,
                                  val_accuracy_sum])

        print(plot_data_sum.shape)

        plot_data_sum = np.array(plot_data_sum)
        torch.save(parameter_in, f"{directory}{size}_param_in_sum_save_{identity}_{qname}.pt")
        torch.save(parameter_out, f"{directory}{size}_param_out_sum_save_{identity}_{qname}.pt")
        np.save(f"{directory}{size}_Plot_data_raw_{identity}_{qname}.npy", plot_data_sum)

    c = np.load(f"{directory}{size}_Plot_data_raw_{identity}_{qname}.npy")

    # print(c.shape)

    trainingEpoch_loss_sum = c[0]
    validationEpoch_loss_sum = c[1]
    train_accuracy_sum = c[2]
    val_accuracy_sum = c[3]

    overall_training_loss = []
    overall_validation_loss = []
    overall_training_acc = []
    overall_validation_acc = []

    overall_training_loss_avg = []
    overall_training_loss_std = []
    overall_training_loss_min = []
    overall_training_loss_max = []

    overall_validation_loss_avg = []
    overall_validation_loss_std = []
    overall_validation_loss_min = []
    overall_validation_loss_max = []

    overall_training_acc_avg = []
    overall_training_acc_std = []
    overall_training_acc_min = []
    overall_training_acc_max = []

    overall_validation_acc_avg = []
    overall_validation_acc_std = []
    overall_validation_acc_min = []
    overall_validation_acc_max = []

    for i in range(n_epochs):
        # overall_training_loss = []
        # overall_validation_loss = []
        # overall_training_acc = []
        # overall_validation_acc = []

        overall_training_loss = trainingEpoch_loss_sum[:, [i]].reshape(-1)
        overall_validation_loss = validationEpoch_loss_sum[:, [i]].reshape(-1)
        overall_training_acc = train_accuracy_sum[:, [i]].reshape(-1)
        overall_validation_acc = val_accuracy_sum[:, [i]].reshape(-1)

        # for j in range(5):
        #    overall_training_loss.append(trainingEpoch_loss_sum[j][i])
        #    overall_validation_loss.append(validationEpoch_loss_sum[j][i])
        #    overall_training_acc.append(train_accuracy_sum[j][i])
        #    overall_validation_acc.append(val_accuracy_sum[j][i])

        overall_training_loss_avg.append(sum(overall_training_loss) / len(overall_training_loss))
        overall_training_loss_std.append(np.std(overall_training_loss))
        overall_training_loss_min.append(min(overall_training_loss))
        overall_training_loss_max.append(max(overall_training_loss))

        overall_validation_loss_avg.append(sum(overall_validation_loss) / len(overall_validation_loss))
        overall_validation_loss_std.append(np.std(overall_validation_loss))
        overall_validation_loss_min.append(min(overall_validation_loss))
        overall_validation_loss_max.append(max(overall_validation_loss))

        overall_training_acc_avg.append(sum(overall_training_acc) / len(overall_training_acc))
        overall_training_acc_std.append(np.std(overall_training_acc))
        overall_training_acc_min.append(min(overall_training_acc))
        overall_training_acc_max.append(max(overall_training_acc))

        overall_validation_acc_avg.append(sum(overall_validation_acc) / len(overall_validation_acc))
        overall_validation_acc_std.append(np.std(overall_validation_acc))
        overall_validation_acc_min.append(min(overall_validation_acc))
        overall_validation_acc_max.append(max(overall_validation_acc))

    # plot_data_overall = [[[overall_training_loss_avg, overall_training_loss_min, overall_training_loss_max],
    #                      [overall_validation_loss_avg, overall_validation_loss_min, overall_validation_loss_max]],
    #                      [[overall_training_acc_avg, overall_training_acc_min, overall_training_acc_max],
    #                      [overall_validation_acc_avg, overall_validation_acc_min, overall_validation_acc_max]]]

    dif = sum(overall_validation_loss_avg) / len(overall_validation_loss_avg) - sum(overall_training_loss_avg) / len(
        overall_training_loss_avg)

    plot_data_overall = np.array([[[overall_training_loss_avg, overall_training_loss_std],
                                   [overall_validation_loss_avg, overall_validation_loss_std]],
                                  [[overall_training_acc_avg, overall_training_acc_std],
                                   [overall_validation_acc_avg, overall_validation_acc_std]],
                                  [dif]], dtype=object, )

    # df2 = pd.DataFrame(plot_data_overall)
    # df2.to_csv(f"/home/faierr/projects/qml-cybrex/models/angenc_basent/tr_sz_{size}/{size}_Plot_data_overall.csv")
    # df2.to_csv(f"/home/faierr/data/angenc_basent/{size}_Plot_data_overall.csv")

    np.save(f"{directory}{size}_Plot_data_overall_{identity}_{qname}.npy",
            plot_data_overall)

    x = np.arange(0, n_epochs, 1)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.suptitle(f'{size} Circuit: {identity} {qname}')
    ax1.fill_between(x, np.array(overall_training_loss_avg) - np.array(overall_training_loss_std),
                     np.array(overall_training_loss_avg) + np.array(overall_training_loss_std), alpha=0.2, color='blue')
    ax1.fill_between(x, np.array(overall_validation_loss_avg) - np.array(overall_validation_loss_std),
                     np.array(overall_validation_loss_avg) + np.array(overall_validation_loss_std), alpha=0.2,
                     color='orange')
    ax1.plot(x, overall_training_loss_avg, label=f'train_loss_avg:{overall_training_loss_avg[-1]:.4f}', color='blue')
    ax1.plot(x, overall_validation_loss_avg, label=f'val_loss_avg:{overall_validation_loss_avg[-1]:.4f}',
             color='orange')
    # ax1.plot(x, overall_validation_loss_min, label='val_loss_min', color = 'orange')
    # ax1.plot(x, overall_validation_loss_max, label='val_loss_max', color = 'orange')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.set_xlim(-1, 11)
    # ax1.set_ylim(1.75,2.4)
    ax1.legend(loc='upper right')

    ax2.fill_between(x, np.array(overall_training_acc_avg) - np.array(overall_training_acc_std),
                     np.array(overall_training_acc_avg) + np.array(overall_training_acc_std), alpha=0.2, color='blue')
    ax2.fill_between(x, np.array(overall_validation_acc_avg) - np.array(overall_validation_acc_std),
                     np.array(overall_validation_acc_avg) + np.array(overall_validation_acc_std), alpha=0.2,
                     color='orange')
    ax2.plot(overall_training_acc_avg, label=f'train_acc:{overall_training_acc_avg[-1]:.4f}')
    ax2.plot(overall_validation_acc_avg, label=f'val_acc:{overall_validation_acc_avg[-1]:.4f}')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('epoch')
    ax2.set_xlim(-1, 11)
    # ax2.set_ylim(0.6, 0.80)
    ax2.legend(loc='lower right')

    generalization_error = []

    # dif1000 = sum(overall_validation_loss_avg)/len(overall_validation_loss_avg) - sum(overall_training_loss_avg)/len(overall_training_loss_avg)
    # generalization_error.append(dif)
    # ax3.plot(generalization_error)

    f.savefig(f"{directory}{size}_plot_{identity}_{qname}.png")
    # f.tight_layout()
    # f.show()
    plt.close()
