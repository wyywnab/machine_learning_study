import torch
import torch as t
from torch import nn
from torch.nn import functional as F

from models.attention import ChannelAttention, SpatialAttention


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 更稳定的实现
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = (random_tensor < keep_prob).float()
        output = x.div(keep_prob) * random_tensor
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None, use_cbam=False, drop_prob=None):
        super(ResidualBlock, self).__init__()
        self.use_cbam = use_cbam

        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.right = shortcut

        self.drop_path = DropPath(drop_prob) if drop_prob is not None else nn.Identity()

        if self.use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        out = self.left(x)

        if self.use_cbam:
            out = self.cbam(out)

        residual = x if self.right is None else self.right(x)
        out = out + self.drop_path(residual)
        return F.relu(out)


class ResNet34(nn.Module):
    """
    实现主module:ResNet34
    ResNet34包含多个layer，每个layer又包含多个residual block
    用子model实现residual block，用__make_layer__函数实现layer
    """
    def __init__(self, num_classes=10, cbam_enabled=False, drop_prob=None):
        """
        构建ResNet34网络的各层结构
        :param num_classes:
        """
        super(ResNet34, self).__init__()
        self.drop_prob = drop_prob

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self.__make_layer__(64, 64, 3, use_cbam=False)  # 第一个layer不改变通道数
        self.layer2 = self.__make_layer__(64, 128, 4, stride=2, use_cbam=cbam_enabled)
        self.layer3 = self.__make_layer__(128, 256, 6, stride=2, use_cbam=cbam_enabled)
        self.layer4 = self.__make_layer__(256, 512, 3, stride=2, use_cbam=cbam_enabled)
        self.fc = nn.Linear(512, num_classes)

    def __make_layer__(self, in_channels, out_channels, block_num, stride=1, use_cbam=False):
        """
        构建layer，包含多个residual block
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param block_num: 块数量
        :param stride: 步长
        :param use_cbam: 是否使用CBAM
        :return:
        """
        shortcut = None
        # 只有当步长不为1或者输入输出通道数不同时，才需要shortcut卷积
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        # 第一个块可能改变通道数和尺寸
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut, use_cbam=use_cbam, drop_prob=self.drop_prob))
        
        # 后续的块保持通道数和尺寸不变
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels, use_cbam=use_cbam, drop_prob=self.drop_prob))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))  # 使用自适应池化，适应不同输入尺寸
        x = x.view(x.size(0), -1)
        return self.fc(x)