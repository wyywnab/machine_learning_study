import torch


class Net64(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 64),
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

class Net64_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
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

class NetConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)

        self.c0 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 7, 1, 3),
            torch.nn.MaxPool2d(2)
        )
        self.m1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, 1, 1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 3, 1, 1),
            torch.nn.BatchNorm2d(4)
        )
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 3, 1, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, 1, 1),
            torch.nn.BatchNorm2d(8)
        )
        self.proj1 = torch.nn.Conv2d(4, 8, 1)
        self.m2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, 3, 1, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, 1, 1),
            torch.nn.BatchNorm2d(8)
        )
        self.c2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(16)
        )
        self.proj2 = torch.nn.Conv2d(8, 16, 1)
        self.m3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(16)
        )
        self.c3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32)
        )
        self.proj3 = torch.nn.Conv2d(16, 32, 1)
        self.m4 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32)
        )
        self.c4 = torch.nn.AvgPool2d(2)
        self.c5 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*7*7, 32),
            torch.nn.Linear(32,27)
        )

    def basics(self, callback, steps, inp):
        out = inp
        for i in range(steps):
            identity = out
            out = callback(out)
            out = out + identity
            out = self.relu(out)
        return out

    def trans(self, m, pr, inp):
        identity = inp
        out = m(inp)
        identity = pr(identity)
        out = out + identity
        out = self.relu(out)
        return out

    def forward(self, x):
        out = self.c0(x)

        out = self.basics(self.m1, 2, out)
        out = self.trans(self.c1, self.proj1, out)
        out = self.basics(self.m2, 2, out)
        out = self.trans(self.c2, self.proj2, out)
        out = self.basics(self.m3, 2, out)
        out = self.trans(self.c3, self.proj3, out)
        out = self.basics(self.m4, 2, out)

        out = self.c4(out)
        out = self.c5(out)

        x = torch.nn.functional.log_softmax(out, dim=1)
        return x