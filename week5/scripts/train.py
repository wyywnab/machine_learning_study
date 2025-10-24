import argparse
import csv
import hashlib
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

from trainer_ema import Trainer
from evaluater import evaluate_with_detailed_analysis, evaluate_grad_cam
from utils import get_data_root_folder


def get_data_loaders(data_root, batch_size, seed, num_workers, transform_str):
    """train_set_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),  # 随机旋转 ±10 度
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])"""
    test_compose_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    train_compose_list = []
    if (transform_str != None and transform_str != ""):
        for key in transform_str.split(","):
            if key == "color_jitter":
                train_compose_list.append(transforms.ColorJitter(
                    brightness=0.2,    # 亮度变化
                    contrast=0.2,      # 对比度变化
                    saturation=0.2,    # 饱和度变化
                    hue=0.1            # 色相变化
                ))
            elif key == "cutout":
                train_compose_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0))
            elif key == "horizontal_flip":
                train_compose_list.append(transforms.RandomHorizontalFlip(p=0.5))
            elif key == "rand_augment":
                train_compose_list.append(transforms.RandAugment(
                    num_ops=config.get("randaug_num_ops", 2),
                    magnitude=config.get("randaug_magnitude", 9)
                ))
            elif key == "standard":
                train_compose_list.append(transforms.RandomAffine(
                    degrees=15,  # 旋转范围
                    translate=(0.1, 0.1),  # 平移比例
                    scale=(0.8, 1.2),  # 缩放范围
                    shear=10  # 剪切角度
                ))
                train_compose_list.append(transforms.RandomHorizontalFlip(p=0.5))

    train_compose_list += test_compose_list

    train_set = CIFAR10(data_root, transform=transforms.Compose(train_compose_list), download=True, train=True)

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

    test_set = CIFAR10(data_root, transform=transforms.Compose(test_compose_list), download=True, train=False)

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

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

if __name__ == "__main__":
    data_root_folder = get_data_root_folder()

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

    train_loader, val_loader, test_loader = get_data_loaders(data_root_folder, config["batch_size"], config["seed"], config["num_workers"], config["data_augmentation"])
    trainer = Trainer(current_folder, config, device, train_loader, val_loader)
    trainer.train()
    checkpoint = trainer.get_best_checkpoint()
    model = trainer.get_model()

    model.load_state_dict(checkpoint["net_state_dict"])

    report, test_acc = evaluate_with_detailed_analysis(
        model=model,
        test_loader=test_loader,
        class_names=CIFAR10(data_root_folder, download=True, train=True).classes,
        folder=current_folder,
        device=device
    )
    report['detailed_report'].to_csv(os.path.join(current_folder, 'per_class_analysis.csv'), index=False)
    evaluate_grad_cam(
        model=model,
        test_loader=test_loader,
        class_names=CIFAR10(data_root_folder, download=True, train=True).classes,
        folder=current_folder,
        device=device
    )

    #accuracy_single, test_acc = save_confusion_matrix_and_accuracy(test_loader, model, checkpoint["epoch"], config["name"], current_folder, device)

    final_metrics = trainer.get_final_metrics()
    inf = {
        "name": config["name"],
        "date": str(datetime.now().isoformat()),
        "best_epoch": checkpoint["epoch"],
        "total_time" : final_metrics["total_train_time"],
        "test_accuracy": test_acc,
        #"test_accuracy_single": accuracy_single,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cuda_version": str(torch.version.cuda),
        "driver_version": str(get_nvidia_driver_version()),
        "graphic_card": torch.cuda.get_device_name(0),
        "cuda": torch.cuda.is_available(),
        "weights": trainer.get_weights_name(),
        "weights_sha256": calculate_sha256(os.path.join(current_folder, trainer.get_weights_name()))
    }
    with open(os.path.join(current_folder, "information.yaml"), "w", encoding="utf-8") as file:
        yaml.dump(inf, file, allow_unicode=True)

    all_curves_csv_path = os.path.join("experiments", "all_curves.csv")
    experiments_csv_path = os.path.join("experiments", "experiments.csv")
    if not os.path.exists(all_curves_csv_path):
        with open(all_curves_csv_path, 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_head = ["exp_id", "exp_name", "epoch", "train_acc", "train_loss", "val_acc", "val_loss", "is_best", "duration"]
            csv_write.writerow(csv_head)
    if not os.path.exists(experiments_csv_path):
        with open(experiments_csv_path, 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_head = ["exp_id", "exp_name", "seed", "epochs", "learning_rate", "optimizer",
                        "lr_scheduler", "data_augmentation", "mix_up", "cut_mix", "cbam_enabled", "top-1_acc", "duration"]
            csv_write.writerow(csv_head)

    current_curves_path = os.path.join(current_folder, "curves.csv")
    with open(all_curves_csv_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_reader = csv.reader(open(current_curves_path))
        next(csv_reader)
        for row in csv_reader:
            csv_write.writerow([exp_id] + row)

    with open(experiments_csv_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        row = [exp_id, config["name"], config["seed"], inf["best_epoch"], config["learning_rate"], config["optimizer"],
               config["lr_scheduler"], config["data_augmentation"], config["mix_up"], config["cut_mix"], config["cbam_enabled"], test_acc, inf['total_time']]
        csv_write.writerow(row)