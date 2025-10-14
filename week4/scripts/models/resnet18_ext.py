# -*- coding: utf-8 -*-
import torch.nn as nn
import torchvision

from week4.scripts.models.attention import ChannelAttention
from week4.scripts.models.attention import SpatialAttention


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class Resnet18Ext(nn.Module):
    def __init__(self, num_classes=10, cbam_enabled=True):
        super(Resnet18Ext, self).__init__()
        self.cbam_enabled = cbam_enabled

        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = nn.Sequential()

        if cbam_enabled:
            self.cbam1 = CBAM(64)
            self.cbam2 = CBAM(128)
            self.cbam3 = CBAM(256)
            self.cbam4 = CBAM(512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        if self.cbam_enabled:
            x = self.model.layer1(x)
            x = self.cbam1(x)
            x = self.model.layer2(x)
            x = self.cbam2(x)
            x = self.model.layer3(x)
            x = self.cbam3(x)
            x = self.model.layer4(x)
            x = self.cbam4(x)
        else:
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.squeeze(2).squeeze(2)

        x = self.classifier(x)

        return x
