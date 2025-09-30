import torch


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