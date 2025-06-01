import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# === Config ===
epochs = 20
learning_rates = [0.15, 0.1, 0.075, 0.05]
duration = 30  # interval for logging loss/grad/beta
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)

# === Device Setup ===
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# === Reproducibility ===
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# === Training Function with Gradient/Parameter Tracking ===
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=20, duration=30):
    model.to(device)
    loss_values, grad_distances, beta_values = [], [], []
    iteration, accumulated_loss = 0, 0
    previous_grad, previous_param = None, None

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        for x, y in train_loader:
            iteration += 1
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            accumulated_loss += loss.item()

            if iteration % duration == 0:
                current_grad = model.classifier[-1].weight.grad.detach().clone()
                current_param = model.classifier[-1].weight.detach().clone()

                if previous_grad is not None:
                    grad_distance = torch.dist(current_grad, previous_grad).item()
                    grad_distances.append(grad_distance)

                if previous_param is not None:
                    param_distance = torch.dist(current_param, previous_param).item()
                    beta = grad_distance / (param_distance + 1e-3)
                    beta_values.append(beta)

                previous_grad = current_grad
                previous_param = current_param

                loss_values.append(accumulated_loss / duration)
                accumulated_loss = 0

    return loss_values, grad_distances, beta_values

# === Plotting Functions ===
def plot_loss_landscape(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
    min_vgg, max_vgg = np.min(list_vgg, axis=0), np.max(list_vgg, axis=0)
    steps = np.arange(len(min_vgg)) * duration
    ax.plot(steps, min_vgg, 'g-', alpha=0.8, label=label_vgg)
    ax.plot(steps, max_vgg, 'g-', alpha=0.8)
    ax.fill_between(steps, min_vgg, max_vgg, color='g', alpha=0.4)

    min_vgg_bn, max_vgg_bn = np.min(list_vgg_bn, axis=0), np.max(list_vgg_bn, axis=0)
    steps = np.arange(len(min_vgg_bn)) * duration
    ax.plot(steps, min_vgg_bn, 'r-', alpha=0.8, label=label_vgg_bn)
    ax.plot(steps, max_vgg_bn, 'r-', alpha=0.8)
    ax.fill_between(steps, min_vgg_bn, max_vgg_bn, color='r', alpha=0.4)
    ax.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax.legend()

def plot_gradient_distance(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
    plot_loss_landscape(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration)

def plot_beta_smoothness(ax, list_vgg, list_vgg_bn, title, ylabel, label_vgg, label_vgg_bn, duration):
    max_vgg = np.max(np.asarray(list_vgg), axis=0)
    max_vgg_bn = np.max(np.asarray(list_vgg_bn), axis=0)
    steps = np.arange(len(max_vgg)) * duration
    ax.plot(steps, max_vgg, 'g-', alpha=0.8, label=label_vgg)
    ax.plot(steps, max_vgg_bn, 'r-', alpha=0.8, label=label_vgg_bn)
    ax.set(title=title, ylabel=ylabel, xlabel='Iterations')
    ax.legend()

# === Main ===
if __name__ == '__main__':
    set_random_seeds(seed_value=2020, device=device)
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)

    grad_list_vgg, loss_list_vgg, beta_list_vgg = [], [], []
    grad_list_vgg_bn, loss_list_vgg_bn, beta_list_vgg_bn = [], [], []

    for lr in learning_rates:
        print(f'Training VGG_A with lr={lr}')
        model = VGG_A()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss, grad, beta = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs, duration=duration)
        loss_list_vgg.append(loss)
        grad_list_vgg.append(grad)
        beta_list_vgg.append(beta)

    for lr in learning_rates:
        print(f'Training VGG_A_BatchNorm with lr={lr}')
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss, grad, beta = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs, duration=duration)
        loss_list_vgg_bn.append(loss)
        grad_list_vgg_bn.append(grad)
        beta_list_vgg_bn.append(beta)

    # === Plot Results ===
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
    plot_loss_landscape(ax, np.array(loss_list_vgg), np.array(loss_list_vgg_bn), 'Loss Landscape', 'Loss', 'VGG', 'VGG+BN', duration)
    plt.savefig(os.path.join(output_dir, 'Loss_Landscape.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
    plot_gradient_distance(ax, np.array(grad_list_vgg), np.array(grad_list_vgg_bn), 'Gradient Predictiveness', 'Gradient Distance', 'VGG', 'VGG+BN', duration)
    plt.savefig(os.path.join(output_dir, 'Gradient_Distance.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
    plot_beta_smoothness(ax, np.array(beta_list_vgg), np.array(beta_list_vgg_bn), 'Beta Smoothness', 'Beta', 'VGG', 'VGG+BN', duration)
    plt.savefig(os.path.join(output_dir, 'Beta_Smoothness.png'))
    plt.close()