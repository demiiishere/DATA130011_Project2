import torch
import torch.nn as nn
import torch.nn.functional as F



class MyCNN_baseline(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNN_baseline, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # [3,32,32] → [64,32,32]
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # [64,32,32] → [64,16,16]

        # 替代 ResidualBlock(64, 128, downsample=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # downsample by stride
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # [128,8,8] → [128,4,4]

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # [128,4,4] → [128,2,2]

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

# SE block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Basic Residual Block with SE and Dropout
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.downsample = downsample or in_channels != out_channels

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            Mish(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.se = SEBlock(out_channels)

    def forward(self, x):
        out = self.conv_bn_relu(x)
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

# CNN Model
class MyCNN_Enhanced(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNN_Enhanced, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.resblock1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)  # [64,32,32] → [64,16,16]

        self.resblock2 = ResidualBlock(64, 128, downsample=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # [128,16,16] → [128,8,8]

        self.resblock3 = ResidualBlock(128, 128)
        self.pool3 = nn.MaxPool2d(2, 2)  # [128,8,8] → [128,4,4]

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.resblock1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, width_per_group, downsample=False):
        super().__init__()
        width = int((out_channels / 64.0) * width_per_group) * groups  # base_width fixed to 64

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.act1 = Mish()

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.act2 = Mish()

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = Mish()

        self.se = SEBlock(out_channels)

        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # nn.AvgPool2d(kernel_size=2, stride=stride),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.act3(self.bn3(self.conv3(out)))
        out = self.se(out)
        if out.shape != identity.shape:
            print(f"[Debug] Shape mismatch: out={out.shape}, identity={identity.shape}")

        return F.relu(out + identity)

# CNN Model
class MyCNN_Enhanced_mish(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNN_Enhanced_mish, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish()
        )

        self.resblock1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)  # [64,32,32] → [64,16,16]

        self.resblock2 = ResidualBlock(64, 128, downsample=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # [128,16,16] → [128,8,8]

        self.resblock3 = ResidualBlock(128, 128)
        self.pool3 = nn.MaxPool2d(2, 2)  # [128,8,8] → [128,4,4]

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.resblock1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

# CNN Model
class MyCNN_Enhanced_mish_dropoutmore(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNN_Enhanced_mish_dropoutmore, self).__init__()
        # self.stage1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     Mish()
        # )
        # consider adding dropout in input layer
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish(),
            nn.Dropout2d(0.1)
        )

        self.resblock1 = ResidualBlock(64, 64)
        # # consider adding dropout here
        self.dropout1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2, 2)  # [64,32,32] → [64,16,16]

        self.resblock2 = ResidualBlock(64, 128, downsample=True)
        # # consider adding dropout here
        self.dropout2 = nn.Dropout2d(0.3)
        self.pool2 = nn.MaxPool2d(2, 2)  # [128,16,16] → [128,8,8]

        self.resblock3 = ResidualBlock(128, 128)
        self.pool3 = nn.MaxPool2d(2, 2) 

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.resblock1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        # x = self.dropout2(x)
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    


class MyCNN_Enhanced_mish_dropoutmore_Large(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNN_Enhanced_mish_dropoutmore_Large, self).__init__()
        # 加宽：conv1_out = 96
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            Mish(),
            nn.Dropout2d(0.1)
        )

        # ResBlock1: 96 → 96
        self.resblock1 = ResidualBlock(96, 96)
        self.dropout1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2, 2)

        # ResBlock2: 96 → 192
        self.resblock2 = ResidualBlock(96, 192, downsample=True)
        self.dropout2 = nn.Dropout2d(0.3)
        self.pool2 = nn.MaxPool2d(2, 2)

        # ResBlock3: 192 → 192
        self.resblock3 = ResidualBlock(192, 192)
        self.pool3 = nn.MaxPool2d(2, 2)

        # fc_hidden = 384
        self.classifier = nn.Sequential(
            nn.Linear(192 * 2 * 2, 384),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.resblock1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    


class MyCNN_Enhanced_mish_dropoutmore_deeper(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNN_Enhanced_mish_dropoutmore_deeper, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish(),
            nn.Dropout2d(0.1)
        )

        self.resblock1 = ResidualBlock(64, 64)
        self.dropout1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2, 2)  # [64,32,32] → [64,16,16]

        self.resblock2 = ResidualBlock(64, 128, downsample=True)
        self.dropout2 = nn.Dropout2d(0.3)
        self.pool2 = nn.MaxPool2d(2, 2)  # [128,16,16] → [128,8,8]

        self.resblock3 = ResidualBlock(128, 128)
        self.dropout3 = nn.Dropout2d(0.3)

        self.resblock4 = ResidualBlock(128, 128)
        self.dropout4 = nn.Dropout2d(0.2)

        self.pool3 = nn.MaxPool2d(2, 2)  # [128,8,8] → [128,4,4]

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.resblock1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.resblock2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        x = self.resblock3(x)
        x = self.dropout3(x)
        x = self.resblock4(x)
        x = self.dropout4(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # CIFAR-style stem
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)
    



# class SimpleDLA(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleDLA, self).__init__()
#         self.base = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True)
#         )
#         self.stage1 = self._make_layer(16, 16, num_blocks=1)
#         self.stage2 = self._make_layer(16, 32, num_blocks=1, stride=2)
#         self.stage3 = self._make_layer(32, 64, num_blocks=1, stride=2)
#         self.stage4 = self._make_layer(64, 128, num_blocks=1, stride=2)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(128, num_classes)

#     def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
#         layers = []
#         for i in range(num_blocks):
#             layers.append(BasicBlock(in_channels if i == 0 else out_channels, out_channels, stride if i == 0 else 1))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.base(x)
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         x = self.pool(x).flatten(1)
#         return self.fc(x)


class ResidualBlock_modified(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn=nn.ReLU, downsample=False):
        super(ResidualBlock_modified, self).__init__()
        self.activation = activation_fn()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.residual_transform = None
        if downsample or in_channels != out_channels:
            self.residual_transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_transform is not None:
            identity = self.residual_transform(x)

        out += identity
        out = self.activation(out)

        return out
    



class MyCNN_Enhanced_Activatable(nn.Module):
    def __init__(self, activation_fn=nn.ReLU, num_classes=10):
        super(MyCNN_Enhanced_Activatable, self).__init__()
        self.activation = activation_fn()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.activation,
            nn.Dropout2d(0.1)
        )

        self.resblock1 = ResidualBlock_modified(64, 64)
        self.dropout1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.resblock2 = ResidualBlock_modified(64, 128, downsample=True)
        self.dropout2 = nn.Dropout2d(0.3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.resblock3 = ResidualBlock_modified(128, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.resblock1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        # x = self.dropout2(x)  # 可选 dropout
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock_ac(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, activation_fn=nn.ReLU):
        super(ResidualBlock_ac, self).__init__()
        self.activation = activation_fn()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.residual_transform = None
        if downsample or in_channels != out_channels:
            self.residual_transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_transform is not None:
            identity = self.residual_transform(x)

        out += identity
        out = self.activation(out)

        return out


class MyCNN_Scalable(nn.Module):
    def __init__(self, conv1_out=64, block2_out=128, fc_hidden=256, activation_fn=nn.ReLU, num_classes=10):
        super(MyCNN_Scalable, self).__init__()
        self.activation = activation_fn()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, conv1_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv1_out),
            self.activation,
            nn.Dropout2d(0.1)
        )

        self.resblock1 = ResidualBlock_ac(conv1_out, conv1_out, activation_fn=activation_fn)
        self.dropout1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.resblock2 = ResidualBlock_ac(conv1_out, block2_out, activation_fn=activation_fn, downsample=True)
        self.dropout2 = nn.Dropout2d(0.3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.resblock3 = ResidualBlock_ac(block2_out, block2_out, activation_fn=activation_fn)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(block2_out * 2 * 2, fc_hidden),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.resblock1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class MyCNN_Deep(nn.Module):
    def __init__(self, conv1_out=64, block2_out=128, fc_hidden=256,
                 n_blocks1=1, n_blocks2=1, n_blocks3=1, activation_fn=nn.ReLU, num_classes=10):
        super(MyCNN_Deep, self).__init__()
        self.activation = activation_fn()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, conv1_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv1_out),
            self.activation,
            nn.Dropout2d(0.1)
        )

        self.block1 = nn.Sequential(*[
            ResidualBlock_ac(conv1_out, conv1_out, activation_fn=activation_fn)
            for _ in range(n_blocks1)
        ])
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block2 = nn.Sequential(*[
            ResidualBlock_ac(conv1_out if i == 0 else block2_out, block2_out,
                          activation_fn=activation_fn, downsample=(i == 0))
            for i in range(n_blocks2)
        ])
        self.pool2 = nn.MaxPool2d(2, 2)

        self.block3 = nn.Sequential(*[
            ResidualBlock_ac(block2_out, block2_out, activation_fn=activation_fn)
            for _ in range(n_blocks3)
        ])
        self.pool3 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(block2_out * 2 * 2, fc_hidden),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
