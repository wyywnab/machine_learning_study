import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def SmallResNet():
    # 加载预训练模型
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 修改第一层卷积 (conv1)
    # 原始: kernel_size=7, stride=2, padding=3
    # 改为: kernel_size=3, stride=1, padding=1
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # 移除 maxpool 层（替换为恒等映射）
    model.maxpool = nn.Identity()

    # 验证修改
    #print(model)  # 查看网络结构
    return model