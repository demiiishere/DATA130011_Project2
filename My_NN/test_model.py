from models.my_cnn import MyCNN, MyCNN_baseline
from train_eval import evaluate
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from models.my_cnn import MyCNN, MyCNN_baseline
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyCNN_baseline().to(device)
    
    # load best model
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    
    # Transform and TestLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    test_acc = evaluate(model, testloader)
    print(f"Test Accuracy of Loaded Best Model: {test_acc:.2f}%")