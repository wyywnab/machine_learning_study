import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST

class Net64_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(28*28, 64),
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

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_set = EMNIST("..\..\emnist_data", split="letters", transform=to_tensor, download=True, train=is_train)
    return DataLoader(data_set, batch_size=15, shuffle=True)

def evaluate(test_data, net, device):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net64_2()
    net = net.to(device)

    print("Init accuracy:", evaluate(test_data, net, device))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(20):
        for (x, y) in train_data:
            net.zero_grad()
            x, y = x.to(device), y.to(device)
            output = net.forward(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net, device))

    torch.save(net.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()