import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from models.my_cnn import MyCNN_baseline, MyCNN_Enhanced, MyCNN_Enhanced_mish, MyCNN_Enhanced_mish_dropoutmore, ResNet18, MyCNN_Enhanced_Activatable, MyCNN_Scalable, MyCNN_Deep
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import torch.nn.functional as F

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 保证 cudnn 结果确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(name, model_params):
    if name == 'SGD':
        return optim.SGD(model_params, lr=0.001, momentum=0.9, weight_decay=5e-4)
    elif name == 'Adam':
        return optim.Adam(model_params, lr=0.001)
    elif name == 'RMSprop':
        return optim.RMSprop(model_params, lr=0.001)
    elif name == 'AdamW':
        return optim.AdamW(model_params, lr=0.001)
    else:
        raise ValueError(f"Unknown optimizer: {name}")



# 可选 focal loss 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(logp, targets, reduction='none')
        p = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def loss_comparison(model_cls, loss_fn_dict, trainloader, valloader, num_epochs=20, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for loss_name, loss_fn in loss_fn_dict.items():
        print(f"\n=== Loss Function: {loss_name} ===")
        set_seed(42)
        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        val_accs = []
        for ep in range(num_epochs):
            model.train()
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

            acc = evaluate(model, valloader)
            val_accs.append(acc)
            scheduler.step(acc)
            print(f"Epoch {ep+1:2d} | Val Acc: {acc:.2f}%")

        results[loss_name] = val_accs

    return results

def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()

class MultiClassHingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        targets_onehot = one_hot(targets, num_classes=inputs.size(1)).to(inputs.device)
        pos_scores = (inputs * targets_onehot).sum(dim=1, keepdim=True)
        margin_mat = self.margin + inputs - pos_scores
        margin_mat = margin_mat * (1 - targets_onehot)
        loss = margin_mat.clamp(min=0).sum(dim=1).mean()
        return loss

# 更新 loss 函数字典
loss_fn_dict = {
    "CrossEntropy": nn.CrossEntropyLoss(),
    "CE + Smoothing": nn.CrossEntropyLoss(label_smoothing=0.1),
    "NLLLoss + LogSoftmax": lambda inputs, targets: F.nll_loss(F.log_softmax(inputs, dim=1), targets),
    "FocalLoss": FocalLoss(alpha=1, gamma=2),
    "MSELoss (OneHot)": lambda inputs, targets: F.mse_loss(F.softmax(inputs, dim=1), one_hot(targets, inputs.size(1)).to(inputs.device)),
    "HingeLoss": MultiClassHingeLoss(margin=1.0),
}

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

full_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# split: 45k training + 5k validation
train_size = int(0.95 * len(full_trainset))  # 45000
val_size = len(full_trainset) - train_size  # 5000
trainset, valset = random_split(full_trainset, [train_size, val_size])


# Dataloaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
valloader   = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
testloader  = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate on any dataloader
def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


scale_configs = {
    "Small":  {"conv1_out": 32, "block2_out": 64,  "fc_hidden": 128},
    "Base":   {"conv1_out": 64, "block2_out": 128, "fc_hidden": 256},
    "Large":  {"conv1_out": 96, "block2_out": 192, "fc_hidden": 384},
    "XLarge": {"conv1_out": 128,"block2_out": 256, "fc_hidden": 512},
}

def optimizer_comparison(model_cls, optimizer_names, trainloader, valloader, testloader, num_epochs=20):
    results = {}
    for opt_name in optimizer_names:
        print(f"\n===== Optimizer: {opt_name} =====")
        set_seed(42)

        model = model_cls().to(device)
        optimizer = get_optimizer(opt_name, model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=False
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        val_accs = []
        for ep in range(num_epochs):
            model.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            val_acc = evaluate(model, valloader)
            val_accs.append(val_acc)
            scheduler.step(val_acc)
            print(f"Epoch {ep+1:2d} | Val Acc: {val_acc:.2f}%")

        results[opt_name] = val_accs
    return results

# Evaluate on any dataloader
def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

deep_configs = {
    "Base":     {"n_blocks1": 1, "n_blocks2": 1, "n_blocks3": 1},
    "Deep":     {"n_blocks1": 2, "n_blocks2": 2, "n_blocks3": 1},
    "VeryDeep": {"n_blocks1": 3, "n_blocks2": 3, "n_blocks3": 2},
}

def depth_comparison(configs, trainloader, valloader, model_base_kwargs, num_epochs=20):
    results = {}
    for name, block_cfg in configs.items():
        print(f"\n=== Model Depth: {name} ===")
        set_seed(42)
        model = MyCNN_Deep(**model_base_kwargs, **block_cfg).to(device)
        summary(model, (3, 32, 32))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        val_accs = []
        for ep in range(num_epochs):
            model.train()
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            acc = evaluate(model, valloader)
            val_accs.append(acc)
            scheduler.step(acc)
            print(f"Epoch {ep+1:2d} | Val Acc: {acc:.2f}%")

        results[name] = val_accs
    return results

def get_scheduler(name, optimizer):
    if name == 'None':
        return None
    elif name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif name == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)
    elif name == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def scheduler_comparison(model_cls, scheduler_names, trainloader, valloader, num_epochs=20):
    results = {}

    for sched_name in scheduler_names:
        print(f"\n===== Scheduler: {sched_name} =====")
        set_seed(42)
        model = model_cls().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = get_scheduler(sched_name, optimizer)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        val_accs = []

        for ep in range(num_epochs):
            model.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            val_acc = evaluate(model, valloader)
            val_accs.append(val_acc)
            print(f"Epoch {ep+1:2d} | Val Acc: {val_acc:.2f}%")

            # 对不同 scheduler 调用不同的 step
            if sched_name == 'ReduceLROnPlateau':
                scheduler.step(val_acc)
            elif sched_name != 'None':
                scheduler.step()

        results[sched_name] = val_accs

    return results

def activation_comparison(
    model_cls,
    activation_fn_dict,
    trainloader,
    valloader,
    num_epochs=20,
    save_path="reports/figures/activation_comparison.png",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for name, act_fn in activation_fn_dict.items():
        print(f"\n===== Activation Function: {name} =====")
        set_seed(42)
        model = model_cls(activation_fn=act_fn).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=False
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        val_accs = []

        for ep in range(num_epochs):
            model.train()
            loop = tqdm(trainloader, desc=f"[{name}] Epoch {ep+1}/{num_epochs}", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())

            val_acc = evaluate(model, valloader)
            val_accs.append(val_acc)
            scheduler.step(val_acc)
            print(f"Epoch {ep+1:2d} | Val Acc: {val_acc:.2f}%")

        results[name] = val_accs

    # 绘图
    plt.figure(figsize=(10, 6))
    for name, accs in results.items():
        plt.plot(range(1, num_epochs + 1), accs, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Activation Function Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n📊 Result plot saved to: {save_path}")
    plt.show()

    return results

def filter_scaling_experiment(configs, trainloader, valloader, num_epochs=20):
    results = {}
    for name, cfg in configs.items():
        print(f"\n=== Model Scale: {name} ===")
        set_seed(42)
        model = MyCNN_Scalable(**cfg, activation_fn=nn.ReLU).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        val_accs = []
        for ep in range(num_epochs):
            model.train()
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            val_acc = evaluate(model, valloader)
            val_accs.append(val_acc)
            scheduler.step(val_acc)
            print(f"Epoch {ep+1:2d} | Val Acc: {val_acc:.2f}%")

        results[name] = val_accs
    return results

# # 运行调度器对比实验
# scheduler_list = ['None', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau']
# results = scheduler_comparison(MyCNN_Enhanced_mish_dropoutmore, scheduler_list, trainloader, valloader, num_epochs=20)

# # 绘图
# plt.figure(figsize=(10, 6))
# for name, accs in results.items():
#     plt.plot(range(1, len(accs)+1), accs, label=name)
# plt.xlabel("Epoch")
# plt.ylabel("Validation Accuracy (%)")
# plt.title("Validation Accuracy vs. Epoch for Different LR Schedulers")
# plt.legend()
# plt.grid(True)
# plt.savefig("reports/figures/lr_scheduler_comparison.png")
# plt.show()

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))
    
activation_fn_dict = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "Mish": Mish,
}

# # if you want to compare activation functions
# activation_comparison(
#     model_cls=MyCNN_Enhanced_Activatable,
#     activation_fn_dict=activation_fn_dict,
#     trainloader=trainloader,
#     valloader=valloader,
#     num_epochs=20,
#     save_path="reports/figures/activation_comparison.png"
# )

# # if you want to compare different number of neurons:
# results = filter_scaling_experiment(scale_configs, trainloader, valloader)

# plt.figure(figsize=(10,6))
# for name, accs in results.items():
#     plt.plot(range(1, len(accs)+1), accs, label=name)
# plt.xlabel("Epoch")
# plt.ylabel("Validation Accuracy (%)")
# plt.title("Effect of Filter/Neuron Size on Performance")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("reports/figures/filters_comparison.png")
# plt.show()

# # if you wish to compare different type of loss
# results = loss_comparison(
#     model_cls=lambda: MyCNN_Enhanced_mish_dropoutmore(), 
#     loss_fn_dict=loss_fn_dict,
#     trainloader=trainloader,
#     valloader=valloader,
#     num_epochs=20
# )

# plt.figure(figsize=(10, 6))
# for name, accs in results.items():
#     plt.plot(range(1, len(accs)+1), accs, label=name)
# plt.xlabel("Epoch")
# plt.ylabel("Validation Accuracy (%)")
# plt.title("Loss Function Comparison on CIFAR-10")
# plt.legend()
# plt.grid(True)
# plt.savefig("reports/figures/loss_comparison.png")
# plt.show()

results = depth_comparison(
    configs=deep_configs,
    trainloader=trainloader,
    valloader=valloader,
    model_base_kwargs={
        "conv1_out": 64,
        "block2_out": 128,
        "fc_hidden": 256,
        "activation_fn": nn.ReLU
    }
)


plt.figure(figsize=(10,6))
for name, accs in results.items():
    plt.plot(range(1, len(accs)+1), accs, label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Effect of Model Depth on CIFAR-10")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/figures/depth_comparison.png")
plt.show()