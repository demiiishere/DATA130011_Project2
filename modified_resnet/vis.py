import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models import ResNet18, ResNet18_se


os.makedirs("./figs", exist_ok=True)

checkpoint = torch.load('./checkpoints/ResNet18_se_ckpt.pth', map_location=torch.device('cpu'))
net = ResNet18_se()
net.load_state_dict(checkpoint['net'])
net.eval()

dog = Image.open('plane.jpg').convert('RGB')

transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor()
])
input_tensor = transform(dog).unsqueeze(0)  # [1, 3, 32, 32]

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

net.conv1.register_forward_hook(get_activation('Conv1 Output'))
net.layer1[0].conv1.register_forward_hook(get_activation('Layer1 Conv1 Output'))

with torch.no_grad():
    net(input_tensor)

def visualize_feature_map(tensor, title, save_path, num_rows=4):
    tensor = tensor.squeeze().cpu()  # [C, H, W]
    num_channels = tensor.shape[0]
    num_cols = (num_channels + num_rows - 1) // num_rows

    plt.figure(figsize=(num_cols * 2, num_rows * 2), dpi=200)
    for i in range(num_channels):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(tensor[i], cmap='Spectral')
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

visualize_feature_map(activation['Conv1 Output'], 'Conv1 Feature Map', './figs/conv1_output.png')
visualize_feature_map(activation['Layer1 Conv1 Output'], 'Layer1 Conv1 Feature Map', './figs/layer1_conv1_output.png')