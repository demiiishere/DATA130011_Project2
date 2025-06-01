import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
import numpy as np

classes = ['plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']


def get_dataloaders(batch_size=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def build_model(model_name):
    if model_name == 'ResNet18':
        return ResNet18()
    elif model_name == 'ResNet34':
        return ResNet34()
    elif model_name == 'ResNet18_filtermul':
        return ResNet18_filtermul()
    elif model_name == 'ResNet18_dropout':
        return ResNet18_dropout()
    elif model_name == 'ResNet18_Mish':
        return ResNet18_Mish()
    elif model_name == 'ResNet18_se':
        return ResNet18_se()
    else:
        raise ValueError(f'Unknown model name: {model_name}')


def train_model(model, trainloader, criterion, optimizer, device, epoch, total_epochs, metrics):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(trainloader, desc=f'Training Epoch {epoch+1}/{total_epochs}', ncols=95)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'Tr_loss': f'{train_loss/(batch_idx+1):.3f}',
                         'Tr_acc': f'{100.*correct/total:.3f}%'})
    logging.info(pbar)
    avg_train_loss = train_loss / len(trainloader)
    avg_train_acc = 100. * correct / total
    metrics['train_loss'].append(avg_train_loss)
    metrics['train_acc'].append(avg_train_acc)
    return avg_train_acc

def test_model(model, testloader, criterion, device, epoch, total_epochs, metrics):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(testloader, desc=f'Testing Epoch {epoch+1}/{total_epochs}', ncols=95)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {'Te_loss': f'{test_loss/(batch_idx+1):.3f}', 'Te_acc': f'{100.*correct/total:.3f}%'})
        logging.info(pbar)
    acc = 100.*correct/total
    avg_test_loss = test_loss / len(testloader)
    metrics['test_loss'].append(avg_test_loss)
    metrics['test_acc'].append(acc)
    return acc, avg_test_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', type=str, default='ResNet18', help='model name (default: ResNet18)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader, testloader = get_dataloaders()
    num_epochs = 200

    model_name = args.model
    
    print(f'==> Building and training model: {model_name} ..')
    # Setup logger
    setup_logging(model_name)
    net = build_model(model_name)
    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True
    
    total_params = count_parameters(net)
    print(f'Total parameters: {total_params}')
    logging.info(f'Total parameters: {total_params}')
    
    if args.resume:
        # Load checkpoint.
        print(f'==> Resuming {model_name} from checkpoint..')
        checkpoint_path = f'./checkpoints/{model_name}_ckpt.pth'
        assert os.path.isfile(
            checkpoint_path), f'Error: no checkpoint directory found for {model_name}!'
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        best_acc = 0
        start_epoch = 0

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, cooldown=5, min_lr=1e-6)

    metrics = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}
    
    total_train_time = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_epoch_time = time.time()
        acc_train = train_model(net, trainloader, criterion, optimizer, device, epoch, num_epochs, metrics)
        acc, avg_test_loss = test_model(net, testloader, criterion, device, epoch, num_epochs, metrics)
        scheduler.step(avg_test_loss) 
        epoch_time = time.time() - start_epoch_time
        total_train_time += epoch_time
        
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        
        if acc > best_acc:
            print('Saving..')
            logging.info('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(state, f'./checkpoints/{model_name}_ckpt.pth')
            best_acc = acc
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Epoch time: {epoch_time:.2f}s, Test accuracy: {acc:.2f}%')

    avg_epoch_time = total_train_time / num_epochs
    logging.info(f'Train accuracy for {model_name}: {acc_train:.2f}')
    logging.info(f'Best test accuracy for {model_name}: {best_acc:.2f}%')
    logging.info(f'Total parameters: {total_params}')
    logging.info(f'Total training time: {total_train_time:.2f}s, Average epoch time: {avg_epoch_time:.2f}s')
    print(f'Best accuracy for {model_name}: {best_acc:.2f}%')
    visualize_metrics(metrics, model_name)


if __name__ == '__main__':
    main()
