import argparse
import os

import torch
import yaml

from train import get_data_loaders, save_confusion_matrix_and_accuracy, data_root_folder
from week4.scripts.trainer import get_net


def get_files_starting_with(directory, prefix):
    matching_files = []

    # 确保目录存在
    if not os.path.exists(directory):
        raise FileNotFoundError(f"目录 '{directory}' 不存在")

    # 确保路径是目录
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"'{directory}' 不是目录")

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 检查是否是文件且以指定前缀开头
        if os.path.isfile(file_path) and filename.startswith(prefix):
            matching_files.append(file_path)

    return matching_files

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", type=str, default="exp251014_200933")
    args = ap.parse_args()

    exp_path = os.path.join("experiments", args.exp_id)

    with open(os.path.join(exp_path, "config.yaml"), 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(os.path.join(exp_path, "information.yaml"), 'r', encoding='utf-8') as f:
        inf = yaml.load(f.read(), Loader=yaml.FullLoader)

    _, _, test_loader = get_data_loaders(data_root_folder, config["batch_size"], config["seed"], config["num_workers"], config["data_enhancement"])
    net = get_net(config, device)

    model_path = get_files_starting_with(exp_path, "model_final_epoch_")[0]

    net.load_state_dict(torch.load(model_path))
    accuracy_single, test_acc = save_confusion_matrix_and_accuracy(test_loader, net, inf["best_epoch"], config["name"], "./", device)

    print(f"Test Accuracy: {test_acc}")
    print(f"Accuracy over Classes: \n{accuracy_single}")