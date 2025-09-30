import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST
import matplotlib.pyplot as plt

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


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    data_set = EMNIST("..\..\emnist_data", split="letters", transform=to_tensor, download=True, train=is_train)
    return DataLoader(data_set, batch_size=15, shuffle=True)

def number_to_letter(number):
    if number == 0:
        return "Unknown"
    else:
        return chr(ord('A') + number - 1)

def evaluate_test_set(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = net(x.view(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

def predict(ts, net):
    net.eval()
    with torch.no_grad():
        mean = 0.1307
        std = 0.3081
        ts = (ts - mean) / std

        plt.figure(0)
        plt.imshow(ts.view(28, 28))

        predict_number = torch.argmax(net.forward(ts.view(-1, 28 * 28)))
        predict_letter = number_to_letter(int(predict_number))
        print(str(int(predict_number)), predict_letter)

        plt.title("prediction: " + str(int(predict_number)) + " " + predict_letter)
        plt.show()

def main():
    test_data = get_data_loader(is_train=False)
    #net = Net64()
    net = Net64_2()

    state_dict = torch.load('model.pth', weights_only=True)
    net.load_state_dict(state_dict)

    evaluate_test_set(net, test_data)

    for (n, (x, _)) in enumerate(test_data):
        predict(x[0], net)
        time.sleep(0.5)

    x1 = [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
      [0, 0, 0.3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.3],
      [0, 0, 0.3, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    ts = torch.Tensor(x1)
    predict(ts, net)

if __name__ == "__main__":
    main()