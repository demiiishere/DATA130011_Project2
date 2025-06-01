import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from models.my_cnn import MyCNN_baseline, MyCNN_Enhanced, MyCNN_Enhanced_mish, MyCNN_Enhanced_mish_dropoutmore, ResNet18, MyCNN_Enhanced_mish_dropoutmore_Large, MyCNN_Enhanced_mish_dropoutmore_deeper
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR



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

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, base_scheduler=None):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    if base_scheduler is None:
        base_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

    return warmup_scheduler, base_scheduler
# # Transform
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))
# ])

# # original train set
# full_trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform
# )

# # use the original test set
# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform
# )

# more specific trainsform with data augmentation:
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

# no split
trainset = full_trainset



# Dataloaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader  = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyCNN_baseline().to(device)
# model = MyCNN_Enhanced().to(device)
# model = MyCNN_Enhanced_mish().to(device)
model = MyCNN_Enhanced_mish_dropoutmore().to(device)
# model = MyCNN_Enhanced_mish_dropoutmore_Large().to(device)
# model = MyCNN_Enhanced_mish_dropoutmore_deeper().to(device)
# model = SimpleDLA().to(device)
# model = MyCNN_tl().to(device)
# model = ResNet18().to(device)
# model = SEMXResNeXt29_16x2d(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.1)

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=20, eta_min=1e-5
# )

# warmup_epochs = 5
# total_epochs = 50

# warmup_scheduler, cosine_scheduler = get_warmup_cosine_scheduler(
#     optimizer, warmup_epochs=warmup_epochs, total_epochs=total_epochs
# )
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


# adjusted train
def train(model, trainloader, testloader, criterion, optimizer, scheduler, epoch=200):
    model.train()
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_acc = 0.0
    best_model_path = "./best_model.pth"
    record_every = 50
    iter_count = 0

    for ep in range(epoch):
        model.train()
        correct_train = 0
        total_train = 0
        running_loss = 0.0

        loop = tqdm(trainloader, desc=f"Epoch [{ep+1}/{epoch}]", leave=False)
        for inputs, labels in loop:
            iter_count += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算 train acc（逐 batch 加）
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(trainloader)
        train_acc_epoch = 100 * correct_train / total_train
        test_acc_epoch = evaluate(model, testloader)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc_epoch)
        test_accuracies.append(test_acc_epoch)

        print(f"Epoch {ep+1}: Train Loss: {avg_loss:.4f} | Train Acc: {train_acc_epoch:.2f}% | Test Acc: {test_acc_epoch:.2f}%")

        # 保存 best 模型
        if test_acc_epoch > best_acc:
            best_acc = test_acc_epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with test acc: {best_acc:.2f}%")

        scheduler.step(test_acc_epoch)

    # 可视化
    os.makedirs("reports/figures", exist_ok=True)
    x_axis = list(range(1, epoch + 1))

    plt.figure()
    plt.plot(x_axis, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig("reports/figures/train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(x_axis, train_accuracies, label='Train Accuracy')
    plt.plot(x_axis, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train & Test Accuracy')
    plt.legend()
    plt.savefig("reports/figures/train_test_acc.png")
    plt.close()

if __name__ == '__main__':
    set_seed(42) 
    summary(model, (3, 32, 32))
    train(model, trainloader, testloader, criterion, optimizer, scheduler)
    # # if we add warmup to the model and use cosine decay in LR
    # train(model, trainloader, valloader, criterion, optimizer, warmup_scheduler, cosine_scheduler)
    
    model.load_state_dict(torch.load("./best_model.pth"))
    test_acc = evaluate(model, testloader)
    print(f"Final Test Accuracy (Best Model): {test_acc:.2f}%")
