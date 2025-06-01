import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import os
from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# === Config ===
epochs = 30
lr = 0.1
output_dir = './figs'
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Data ===
train_loader = get_cifar_loader(train=True)
test_loader = get_cifar_loader(train=False)

# === Evaluation ===
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

# === Train Function ===
def train_and_record(model_class, label):
    print(f"Training {label}...")
    model = model_class().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    test_accs = []
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"{label} Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        test_acc = evaluate(model, test_loader)
        test_accs.append(test_acc)
        print(f"[{label}] Epoch {epoch+1:2d} | Test Accuracy: {test_acc:.4f}")
    return test_accs

# === Train both models ===
test_accs_vgg = train_and_record(VGG_A, "VGG_A")
test_accs_bn  = train_and_record(VGG_A_BatchNorm, "VGG_A_BatchNorm")

# === Plotting ===
plt.figure(figsize=(9, 6), dpi=300)
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, test_accs_vgg, label='VGG', marker='o')
plt.plot(epochs_range, test_accs_bn, label='VGG + BatchNorm', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Curve: VGG vs. VGG+BatchNorm")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'VGG_vs_BN_Accuracy_Curve.png'))
plt.close()