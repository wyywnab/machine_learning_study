import os
import threading
import time
from datetime import datetime
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST
from torch.utils.tensorboard import SummaryWriter
from models.cnn_small import SmallCNN
from models.resnet18_small_bak import SmallResNet
from Utils import EarlyStopping
from Utils import draw_confusion_matrix
from week2.scripts.models.mlp import MLP
from week2.scripts.models.rescnn import ResCNN

"""def get_data_loader(is_train, batch_size):
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_set = EMNIST("..\..\emnist_data", split="letters", transform=to_tensor, download=True, train=is_train)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)"""

def get_data_loaders(batch_size, seed, num_workers):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 训练集
    train_set = EMNIST("..\..\emnist_data", split="letters", transform=to_tensor, download=True, train=True)

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)  # 确保可重复性
    )

    # 测试集
    test_set = EMNIST("..\..\emnist_data", split="letters", transform=to_tensor, download=True, train=False)

    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def evaluate(data_loader, net, criterion):
    n_correct = 0
    n_total = 0
    total_loss = 0
    net.eval()
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            #outputs = net.forward(x.view(-1, 28 * 28))
            outputs = net.forward(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1

            #loss = torch.nn.functional.nll_loss(outputs, y)
            loss = criterion(outputs, y)
            total_loss += loss.item()

    accuracy = n_correct / n_total
    loss = total_loss / n_total

    return accuracy, loss

def get_confusion_matrix_image(data_loader, net, criterion, checkpoint, config):
    n_correct = 0
    n_total = 0
    total_loss = 0
    label_true = []
    label_pred = []
    net.load_state_dict(checkpoint["net_state_dict"])
    net.eval()
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            #outputs = net.forward(x.view(-1, 28 * 28))
            outputs = net.forward(x)
            for i, output in enumerate(outputs):
                label_true.append(int(y[i]))
                label_pred.append(int(torch.argmax(output)))

                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1

            loss = criterion(outputs, y)
            total_loss += loss.item()

    label_name = [chr(ord('A') + i) for i in range(26)]

    draw_confusion_matrix(label_true, label_pred, label_name, display=False,
                          title=f'Confusion Matrix of {config["model"]} Best Epoch {checkpoint["epoch"]}',
                          save_path=os.path.join(current_folder, 'confusion_matrix_final_epoch_{}.png'.format(checkpoint["epoch"])))

    accuracy = n_correct / n_total
    loss = total_loss / n_total

    return accuracy, loss

def main():
    """config = {
        "seed": 608,
        "batch_size": 15,
        "learning_rate": 0.001,
        "learning_rate_decline_epoch": 4,
        "learning_rate_decline_rate": 0.8,
        "num_workers": 4,
        "max_epoch": 32
    }"""
    """config = {
        "seed": 608,
        "batch_size": 15,
        "learning_rate": 0.001,
        "weight_decay": 3e-4,  # 1e-4 ~ 5e-4 范围内
        "label_smoothing": 0.05,
        "step_size": 10,  # StepLR的步长
        "gamma": 0.1,  # StepLR的衰减系数
        "t_max": 50,  # CosineAnnealingLR的周期
        "eta_min": 1e-6,  # CosineAnnealingLR的最小学习率
        "clip_value": 1.0,
        "num_workers": 4,
        "max_epoch": 1,
        "early_stopping_patience": 7,
        "amp_enabled": True,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "model": "MLP"
    }"""
    config = {}
    if os.path.exists("config.yaml"):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)

    with open(os.path.join(current_folder, "config.yaml"), "w", encoding="utf-8") as file:
        yaml.dump(config, file, allow_unicode=True)

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    writer = SummaryWriter(log_dir=current_folder)

    #writer.add_hparams(hparam_dict=config,metric_dict={})

    #net = Net64()
    net = SmallResNet()
    net = net.to(device)

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    #optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    if config["scheduler"] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["t_max"],
            eta_min=config["eta_min"]
        )
    elif config["scheduler"] == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=config["step_size"],
            gamma=config["gamma"]
        )
    else:
        scheduler = None

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(verbose=True, dir=current_folder, patience=config["early_stopping_patience"])

    total_start_time = time.time()

    train_loader, val_loader, test_loader = get_data_loaders(config["batch_size"], config["seed"], config["num_workers"])

    init_train_acc, init_train_loss = evaluate(train_loader, net, criterion)
    init_val_acc, init_val_loss = evaluate(val_loader, net, criterion)
    init_test_acc, init_test_loss = evaluate(test_loader, net, criterion)
    print(f"Initial Train Accuracy: {init_train_acc:.5f}, Validation Accuracy: {init_val_acc:.5f}, Test Accuracy: {init_test_acc:.5f}")
    initial_metrics = {
        "best_val_accuracy": init_val_acc,
        "best_val_loss": init_val_loss,
        "best_epoch": 0,
        "test_accuracy": init_test_acc,
        "test_loss": init_test_loss,
        "total_training_time": 0,
        "avg_throughput_samples_sec": 0
    }
    #writer.add_hparams(hparam_dict=config, metric_dict=initial_metrics)

    writer.add_scalar('Accuracy/train', init_train_acc, 0)
    writer.add_scalar('Accuracy/val', init_val_acc, 0)
    writer.add_scalar('Accuracy/test', init_test_acc, 0)
    writer.add_scalar('Loss/train', init_train_loss, 0)
    writer.add_scalar('Loss/val', init_val_loss, 0)
    writer.add_scalar('Loss/test', init_test_loss, 0)

    writer.add_scalar('Learning_rate', config["learning_rate"], 0)

    global_step = 0
    throughputs = []
    for epoch in range(1, config["max_epoch"] + 1):
        epoch_start_time = time.time()
        net.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            #net.zero_grad()
            batch_start_time = time.time()
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            """with torch.amp.autocast(device):
                #output = net.forward(x.view(-1, 28 * 28))
                output = net.forward(x)
                #loss = torch.nn.functional.nll_loss(output, y)
                loss = criterion(output, y)
                
            #loss.backward()
            scaler.scale(loss).backward()"""

            if config["amp_enabled"]:
                with torch.amp.autocast(device):
                    output = net.forward(x)
                    loss = criterion(output, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config["clip_value"])

                scaler.step(optimizer)
                scaler.update()

            else:
                output = net.forward(x)
                loss = criterion(output, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config["clip_value"])

                optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                writer.add_scalar('Loss/train', running_loss / 100, global_step)
                running_loss = 0.0

            if (i + 1) % 1000 == 0:
                throughputs.append(config["batch_size"] / (time.time() - batch_start_time))
                print(f'Epoch {epoch}, Batch: {i + 1} - Throughput: {throughputs[-1]:.5f} sample/s, Loss: {loss.item():.5f}')

            global_step += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_duration = time.time() - epoch_start_time

        train_acc, train_loss = evaluate(train_loader, net, criterion)
        val_acc, val_loss = evaluate(val_loader, net, criterion)
        print(f"Epoch {epoch} - Duration: {epoch_duration:.5f}s, Learning Rate: {current_lr},\n Train Accuracy: {train_acc:.5f}, Validation Accuracy: {val_acc:.5f}")

        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        for name, param in net.named_parameters():
            writer.add_histogram(name, param, epoch)

        early_stopping(val_loss, val_acc, epoch, optimizer, net)
        if early_stopping.early_stop:
            print("EarlyStopping: exiting loop.")
            break

    best_checkpoint = early_stopping.get_best_checkpoint()

    final_model_path = os.path.join(current_folder, 'model_final_epoch_{}.pth').format(best_checkpoint["epoch"])
    torch.save(best_checkpoint["net_state_dict"], final_model_path)
    test_acc, test_loss = get_confusion_matrix_image(test_loader, net, criterion, best_checkpoint, config)

    final_metrics = {
        "hparams/best_val_accuracy": best_checkpoint["best_val_acc"],
        "hparams/best_val_loss": best_checkpoint["best_val_loss"],
        "hparams/test_accuracy": test_acc,
        "hparams/test_loss": test_loss,
        "hparams/best_epoch": best_checkpoint["epoch"],
        "hparams/total_training_time": time.time() - total_start_time,
        "hparams/avg_throughput_samples_sec": sum(throughputs) / len(throughputs)
    }
    """for key, value in final_metrics.items():
        writer.add_scalar(key, value)"""
    print("Metrics:", final_metrics)

    writer.add_hparams(hparam_dict=config, metric_dict=final_metrics)
    writer.flush()
    writer.close()

    print("Finished!")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    timeid = datetime.now().strftime("%y%m%d_%H%M%S")
    #current_folder = ".\\experiments\\exp{}\\".format(time)
    current_folder = os.path.join("experiments", "exp{}".format(timeid))
    if not os.path.exists(current_folder):
        os.makedirs(current_folder)
    main()