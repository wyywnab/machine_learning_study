import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(1, in_planes // ratio)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始卷积层
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 7, 1, 3),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU()
        )

        # 第一个注意力块
        self.ca1 = ChannelAttention(4)
        self.sa1 = SpatialAttention(7)

        # 第二个卷积层
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, 1, 1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU()
        )

        # 第二个注意力块
        self.ca2 = ChannelAttention(4)
        self.sa2 = SpatialAttention(3)

        # 第三个卷积层
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, 1, 1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU()
        )

        # 第三个注意力块
        self.ca3 = ChannelAttention(4)
        self.sa3 = SpatialAttention(3)

        # 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 28 * 28, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 27)
        )

    def forward(self, x):
        # 第一层
        out = self.conv1(x)

        # 应用注意力并残差连接
        ca_out = self.ca1(out)
        sa_out = self.sa1(out)
        out = out * ca_out + out * sa_out  # 结合两种注意力
        out = torch.nn.functional.relu(out)
        out = torch.nn.functional.dropout(out)

        # 第二层
        #identity = out
        out = self.conv2(out)

        ca_out = self.ca2(out)
        sa_out = self.sa2(out)
        out = out * ca_out + out * sa_out  # 结合两种注意力
        out = torch.nn.functional.dropout(out)

        # 第三层
        out = self.conv3(out)

        ca_out = self.ca3(out)
        sa_out = self.sa3(out)
        out = out * ca_out + out * sa_out  # 结合两种注意力
        out = torch.nn.functional.dropout(out)

        # 分类
        out = self.classifier(out)
        return out