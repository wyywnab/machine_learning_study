import argparse
import csv
import pathlib
import re
import subprocess
from datetime import datetime
import os

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from week4.scripts.Utils import draw_confusion_matrix
from week4.scripts.trainer import Trainer

def save_confusion_matrix_and_accuracy(data_loader, net, checkpoint, config, current_folder, device):
    label_true = []
    label_pred = []
    map = [[0] * 10] * 10
    net.load_state_dict(checkpoint["net_state_dict"])
    net.eval()
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = net.forward(x)
            for i, output in enumerate(outputs):
                label_true.append(int(y[i]))
                label_pred.append(int(torch.argmax(output)))
                map[int(y[i])][int(torch.argmax(output))] += 1

    label_name = CIFAR10(data_root_folder, download=True, train=True).classes

    draw_confusion_matrix(label_true, label_pred, label_name, display=False,
                          title=f'Confusion Matrix of {config["model"]} Best Epoch {checkpoint["epoch"]}',
                          save_path=os.path.join(current_folder, 'confusion_matrix_final_epoch_{}.png'.format(checkpoint["epoch"])))

    accuracy_single = {}
    correct_sum = 0
    total_sum = 0
    for i in range(len(map)):
        row_sum = 0
        for num in map[i]:
            row_sum += num
        total_sum += row_sum
        accuracy_single[label_name[i]] = map[i][i] / row_sum
        correct_sum += map[i][i]

    return accuracy_single, correct_sum / total_sum

def get_data_loaders(batch_size, seed, num_workers, transform_str):
    """train_set_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),  # 随机旋转 ±10 度
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])"""
    test_compose_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    train_compose_list = []
    for key in transform_str.split(","):
        if key == "color":
            train_compose_list.append(transforms.ColorJitter(
                brightness=0.2,    # 亮度变化
                contrast=0.2,      # 对比度变化
                saturation=0.2,    # 饱和度变化
                hue=0.1            # 色相变化
            ))
        elif key == "affine":
            train_compose_list.append(transforms.RandomAffine(
                degrees=15,        # 旋转范围
                translate=(0.1,0.1), # 平移比例
                scale=(0.8,1.2),   # 缩放范围
                shear=10           # 剪切角度
            ))

    train_compose_list += test_compose_list

    train_set = CIFAR10(data_root_folder, transform=transforms.Compose(train_compose_list), download=True, train=True)

    # 划分训练集和验证集
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    """test_set_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])"""

    test_set = CIFAR10(data_root_folder, transform=transforms.Compose(test_compose_list), download=True, train=False)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_nvidia_driver_version():
    try:
        # 运行nvidia-smi命令并捕获输出
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        output = result.stdout

        # 使用正则表达式匹配驱动版本
        match = re.search(r'Driver Version: (\d+\.\d+)', output)
        if match:
            return match.group(1)
        else:
            return "驱动版本未找到"

    except subprocess.CalledProcessError as e:
        return f"命令执行失败: {e}"
    except FileNotFoundError:
        return "nvidia-smi命令未找到，请检查NVIDIA驱动安装"
    except Exception as e:
        return f"发生错误: {str(e)}"

if __name__ == "__main__":
    data_root_folder = "..\..\cifar10_data"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_id = "exp{}".format(datetime.now().strftime("%y%m%d_%H%M%S"))
    current_folder = os.path.join("experiments", exp_id)
    if not os.path.exists(current_folder):
        os.makedirs(current_folder)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.yaml"))
    args = ap.parse_args()

    config = {}
    if not os.path.exists(args.config):
        exit(1)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(os.path.join(current_folder, "config.yaml"), "w", encoding="utf-8") as file:
        yaml.dump(config, file, allow_unicode=True)

    train_loader, val_loader, test_loader = get_data_loaders(config["batch_size"], config["seed"], config["num_workers"], config["data_enhancement"])
    trainer = Trainer(current_folder, config, device, train_loader, val_loader)
    trainer.train()
    checkpoint = trainer.get_best_checkpoint()
    model = trainer.get_model()

    accuracy_single, accuracy = save_confusion_matrix_and_accuracy(test_loader, model, checkpoint, config, current_folder, device)

    final_metrics = trainer.get_final_metrics()
    inf = {
        "name": config["name"],
        "date": datetime.now().isoformat(),
        "best_epoch": checkpoint["epoch"],
        "total_time" : final_metrics["total_train_time"],
        "test_accuracy": accuracy,
        "test_accuracy_single": accuracy_single,
        "torch_version": torch.__version__,
        "torchvision_version": str(torchvision.__version__),
        "cuda_version": torch.version.cuda,
        "driver_version": get_nvidia_driver_version(),
        "graphic_card": torch.cuda.get_device_name(0),
    }
    with open(os.path.join(current_folder, "information.yaml"), "w", encoding="utf-8") as file:
        yaml.dump(inf, file, allow_unicode=True)

    all_curves_csv_path = os.path.join("experiments","all_curves.csv")
    all_hparams_csv_path = os.path.join("experiments","all_hparams.csv")
    if not os.path.exists(all_curves_csv_path):
        with open(all_curves_csv_path, 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_head = ["exp_id", "exp_name", "epoch", "train_acc", "train_loss", "val_acc", "val_loss", "is_best", "duration"]
            csv_write.writerow(csv_head)
    if not os.path.exists(all_hparams_csv_path):
        with open(all_hparams_csv_path, 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_head = ["exp_id", "exp_name", "seed", "best_epoch", "lr_scheduler", "data_enhancement", "cbam_enabled", "top-1_acc", "duration"]
            csv_write.writerow(csv_head)

    current_curves_path = os.path.join(current_folder, "curves.csv")
    with open(all_curves_csv_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_reader = csv.reader(open(current_curves_path))
        next(csv_reader)
        for row in csv_reader:
            csv_write.writerow([exp_id] + row)

    with open(all_hparams_csv_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        row = [exp_id, config["name"], config["seed"], inf["best_epoch"], config["lr_scheduler"], config["data_enhancement"], config["cbam_enabled"], accuracy, inf['total_time']]
        csv_write.writerow(row)