import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import os

from models.my_cnn import MyCNN_Enhanced_mish_dropoutmore

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并恢复参数
model = MyCNN_Enhanced_mish_dropoutmore().to(device)
model.load_state_dict(torch.load("best_model_9130.pth", map_location=device))
model.eval()

# 用于保存中间层激活的字典
activation = {}

# 注册 hook 的函数
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# 注册你想观察的层（可以根据模型结构做调整）
model.stage1[0].register_forward_hook(get_activation('Stage1 Conv'))
model.resblock2.conv_bn_relu[0].register_forward_hook(get_activation('ResBlock2 Conv1'))

# 加载和预处理输入图像
img = Image.open('plane.jpg').convert('RGB')  # 请替换为你的图片路径
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 输入尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
input_tensor = transform(img).unsqueeze(0).to(device)

# 前向传播以触发 hook
with torch.no_grad():
    model(input_tensor)

# 可视化特征图函数
def visualize_feature_map(feature_map, title='Feature Map', save_path=None, num_rows=4):
    feature_map = feature_map.squeeze().cpu()  # [C, H, W]
    num_channels = feature_map.shape[0]
    num_cols = (num_channels + num_rows - 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten()

    for i in range(num_channels):
        fmap = feature_map[i]
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)  # Normalize
        axes[i].imshow(fmap, cmap='Spectral')
        axes[i].axis('off')

    for j in range(num_channels, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()

# 可视化 Stage1 的输出
visualize_feature_map(
    activation['Stage1 Conv'],
    title='Stage1 Conv Output',
    save_path='reports/figures/stage1_activation.png'
)

# 可视化 ResBlock2 Conv1 的输出
visualize_feature_map(
    activation['ResBlock2 Conv1'],
    title='ResBlock2 Conv1 Output',
    save_path='reports/figures/resblock2_activation.png'
)