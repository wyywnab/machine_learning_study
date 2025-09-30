import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST
import matplotlib.pyplot as plt
import model


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
            #outputs = net(x.view(-1, 28 * 28))
            outputs = net(x)
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

        # 添加批次维度和通道维度
        if ts.dim() == 2:  # 如果是2D [H, W]
            ts = ts.unsqueeze(0).unsqueeze(0)  # 变为 [1, 1, H, W]
        elif ts.dim() == 3:  # 如果是3D [C, H, W]
            ts = ts.unsqueeze(0)  # 变为 [1, C, H, W]

        #predict_number = torch.argmax(net.forward(ts.view(-1, 28 * 28)))
        predict_number = torch.argmax(net.forward(ts))
        predict_letter = number_to_letter(int(predict_number))
        print(str(int(predict_number)), predict_letter)

        plt.title("prediction: " + str(int(predict_number)) + " " + predict_letter)
        plt.show()

def main():
    test_data = get_data_loader(is_train=False)
    #net = Net64()
    net = model.NetConv()

    state_dict = torch.load('model.pth', weights_only=True)
    net.load_state_dict(state_dict)

    evaluate_test_set(net, test_data)

    """for (n, (x, _)) in enumerate(test_data):
        predict(x[0], net)
        time.sleep(0.5)"""

    x1 = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 1, 0.3, 0.3, 0, 0, 0],
  [0, 0, 0, 0, 0.3, 0.3, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1, 1, 1, 1, 1, 0.3, 0, 0, 0],
  [0, 0, 0, 0, 0.3, 1, 0.3, 1, 1, 1, 0.3, 1, 1, 0.3, 1, 1, 0.3, 1, 1, 0.3, 0.3, 0.3, 0.3, 1, 0.3, 0, 0, 0],
  [0, 0, 0, 0, 0.3, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0, 0, 0.3, 1, 0.3, 0, 0, 0],
  [0, 0, 0, 0.3, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0.3, 0.3, 1, 0.3, 0, 0, 0],
  [0, 0, 0, 0.3, 1, 0.3, 0.3, 0, 0, 0, 0, 0, 0.3, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0.3, 0, 0, 0],
  [0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0, 0, 0, 0.3, 1, 1, 0.3, 0, 0, 0, 0, 0, 0.3, 1, 0.3, 0, 0, 0, 0],
  [0, 0, 0, 0.3, 1, 0.3, 0.3, 0, 0, 0, 0, 0.3, 0.3, 1, 1, 0.3, 0, 0, 0, 0, 0.3, 0.3, 1, 0.3, 0, 0, 0, 0],
  [0, 0, 0, 0.3, 0.3, 1, 0.3, 0, 0, 0, 0.3, 0.3, 1, 1, 1, 0.3, 0.3, 0, 0, 0, 0.3, 1, 0.3, 0.3, 0, 0, 0, 0],
  [0, 0, 0, 0, 0.3, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 1, 0.3, 0.3, 0.3, 1, 0.3, 0, 0, 0.3, 0.3, 1, 0.3, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0.3, 1, 1, 1, 1, 1, 1, 0.3, 0.3, 0, 0.3, 1, 0.3, 0.3, 0.3, 0.3, 1, 0.3, 0.3, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0, 0, 0.3, 0.3, 1, 1, 1, 1, 0.3, 0.3, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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