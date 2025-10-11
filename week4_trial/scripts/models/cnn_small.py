import torch
from torch import nn

from week3.scripts.models.attention import ChannelAttention, SpatialAttention


class SmallCNNAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始卷积层
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 7, 1, 3),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU()
        )

        # 第一个注意力块
        self.ca1 = ChannelAttention(4)
        self.sa1 = SpatialAttention(7)

        # 第二个卷积层
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 3, 1, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU()
        )

        # 第二个注意力块
        self.ca2 = ChannelAttention(8)
        self.sa2 = SpatialAttention(3)

        # 第三个卷积层
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )

        # 第三个注意力块
        self.ca3 = ChannelAttention(16)
        self.sa3 = SpatialAttention(3)

        # 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 32 * 32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x, attention_enabled=False):
        # 第一层
        out = self.conv1(x)

        if attention_enabled:
            # 应用注意力并残差连接
            ca_out = self.ca1(out)
            sa_out = self.sa1(out)
            out = out * ca_out + out * sa_out  # 结合两种注意力
            out = torch.nn.functional.relu(out)
        out = torch.nn.functional.dropout(out)

        # 第二层
        #identity = out
        out = self.conv2(out)

        if attention_enabled:
            ca_out = self.ca2(out)
            sa_out = self.sa2(out)
            out = out * ca_out + out * sa_out  # 结合两种注意力
        out = torch.nn.functional.dropout(out)

        # 第三层
        out = self.conv3(out)

        if attention_enabled:
            ca_out = self.ca3(out)
            sa_out = self.sa3(out)
            out = out * ca_out + out * sa_out  # 结合两种注意力
        out = torch.nn.functional.dropout(out)

        # 分类
        out = torch.nn.functional.log_softmax(self.classifier(out), dim=1)
        return out

class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 7, 1, 3),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv2d(4, 4, 3, 1, 1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Conv2d(4, 4, 3, 1, 1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
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
        x = torch.nn.functional.log_softmax(self.fc(x), dim=1)
        return x