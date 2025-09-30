import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST
from torch.utils.tensorboard import SummaryWriter
import model

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
            #outputs = net.forward(x.view(-1, 28 * 28))
            outputs = net.forward(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    writer = SummaryWriter()

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    #net = Net64()
    net = model.NetConv()
    net = net.to(device)

    init_accuracy = evaluate(test_data, net, device)
    print("Init accuracy:", init_accuracy)
    writer.add_scalar('Accuracy/test', init_accuracy, 0)

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    writer.add_scalar('Learning_rate', 0.001, 0)

    example_data, _ = next(iter(train_data))
    #writer.add_graph(net, example_data.to(device).view(-1, 28 * 28))
    writer.add_graph(net, example_data.to(device))

    global_step = 0
    for epoch in range(16):
        if epoch % 4 == 0:
            torch.save(net.state_dict(), 'model_epoch_{}.pth'.format(epoch))

            # 每迭代4次，更新一次学习率
            for params in optimizer.param_groups:
                # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
                params['lr'] *= 0.8
        running_loss = 0.0
        for i, (x, y) in enumerate(train_data):
            net.zero_grad()
            x, y = x.to(device), y.to(device)
            #output = net.forward(x.view(-1, 28 * 28))
            output = net.forward(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                writer.add_scalar('Loss/train', running_loss / 100, global_step)
                running_loss = 0.0

            global_step += 1

        accuracy = evaluate(test_data, net, device)
        print("epoch", epoch, "accuracy:", accuracy)
        writer.add_scalar('Accuracy/test', accuracy, epoch + 1)

        for name, param in net.named_parameters():
            writer.add_histogram(name, param, epoch)

    writer.close()

    torch.save(net.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()