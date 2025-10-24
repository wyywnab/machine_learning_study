import os
import time
import torch
import csv

from ema_pytorch import EMA
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from models.cnn_small import SmallCNN, SmallCNNCBAM
from models.resnet18_small import SmallResNet
from utils import EarlyStopping
from models.resnet18_ext import Resnet18Ext
from models.resnet34 import ResNet34


def get_net(config, device):
    if config["model"] == "SmallCNN":
        if config["cbam_enabled"]:
            net = SmallCNNCBAM()
        else:
            net = SmallCNN()
    elif config["model"] == "ResNet34":
        net = ResNet34(cbam_enabled=config["cbam_enabled"])
    elif config["model"] == "SmallResNet":
        net = SmallResNet(cbam_enabled=config["cbam_enabled"])
    elif config["model"] == "Resnet18Ext":
        net = Resnet18Ext(cbam_enabled=config["cbam_enabled"])
    else:
        print("Illegal Model Name!")
        exit(0)

    net = net.to(device)
    return net


class Trainer:
    def __init__(self, base_folder, config, device, train_loader, val_loader):
        self.folder = base_folder
        self.config = config
        self.best_checkpoint = None
        self.total_train_time = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = None
        self.device = device
        self.final_metrics = None
        self.ema = None
        self.evaluation_net = None  # 新增：用于评估的模型引用

    def evaluate(self, data_loader, net, criterion):
        correct = 0
        total = 0
        total_loss = 0
        net.eval()
        with torch.no_grad():
            for (x, y) in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = net(x)
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == y).sum().item()
                total += y.size(0)

                loss = criterion(outputs, y)
                total_loss += loss.item() * x.size(0)

        accuracy = correct / total
        avg_loss = total_loss / total

        return accuracy, avg_loss

    def train(self):
        config = self.config

        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        writer = SummaryWriter(log_dir=self.folder)

        net = get_net(config, self.device)

        # 初始化EMA，完全取代原模型
        if config["ema_decay"] is not None:
            self.ema = EMA(net, beta=config["ema_decay"], update_after_step=100, update_every=10)
            # 设置EMA模型为主要的评估模型
            self.evaluation_net = self.ema.ema_model
        else:
            self.evaluation_net = net

        if config["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"]
            )
        elif config["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                net.parameters(),
                momentum=config["sgd_momentum"],
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"]
            )
        if config["lr_scheduler"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config["cosLR_t_max"],
                eta_min=config["cosLR_eta_min"]
            )
        elif config["lr_scheduler"] == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=config["stepLR_step_size"],
                gamma=config["stepLR_gamma"]
            )
        else:
            scheduler = None

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        scaler = torch.amp.GradScaler()
        early_stopping = EarlyStopping(verbose=True, dir=self.folder, patience=config["early_stopping_patience"])

        total_start_time = time.time()

        train_loader = self.train_loader
        val_loader = self.val_loader

        # 初始评估使用EMA模型（如果启用）或原始模型
        init_train_acc, init_train_loss = self.evaluate(train_loader, self.evaluation_net, criterion)
        init_val_acc, init_val_loss = self.evaluate(val_loader, self.evaluation_net, criterion)
        print(f"Initial Train Acc: {init_train_acc:.5f}, Validation Acc: {init_val_acc:.5f}\n")

        writer.add_scalar('Accuracy/train', init_train_acc, 0)
        writer.add_scalar('Accuracy/val', init_val_acc, 0)
        writer.add_scalar('Loss/train', init_train_loss, 0)
        writer.add_scalar('Loss/val', init_val_loss, 0)
        writer.add_scalar('Learning_rate', config["learning_rate"], 0)

        csv_path = os.path.join(self.folder, "curves.csv")
        with open(csv_path, 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_head = ["name", "epoch", "train_acc", "train_loss", "val_acc", "val_loss", "is_best", "duration"]
            csv_write.writerow(csv_head)

        cutmix = v2.CutMix(num_classes=10)
        mixup = v2.MixUp(num_classes=10)

        if config["mix_up"] and config["cut_mix"]:
            cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        elif config["mix_up"] and not config["cut_mix"]:
            cutmix_or_mixup = mixup
        elif not config["mix_up"] and config["cut_mix"]:
            cutmix_or_mixup = cutmix

        global_step = 0
        throughputs = []
        warmup_epochs = 5

        for epoch in range(1, config["max_epoch"] + 1):
            epoch_start_time = time.time()

            if epoch <= warmup_epochs:
                warmup_lr = self.config["learning_rate"] * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                current_lr = warmup_lr
                print(f"Warmup Epoch {epoch}/{warmup_epochs}, Learning Rate: {current_lr:.6f}")
            else:
                if scheduler is not None:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]

            net.train()
            running_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                batch_start_time = time.time()
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)

                if config["mix_up"] or config["cut_mix"]:
                    x, y = cutmix_or_mixup(x, y)

                if config["amp_enabled"]:
                    with torch.amp.autocast(self.device):
                        output = net.forward(x)
                        loss = criterion(output, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    if config["grad_clip_value"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            net.parameters(),
                            max_norm=config["grad_clip_value"],
                            norm_type=2
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = net.forward(x)
                    loss = criterion(output, y)
                    loss.backward()

                    if config["grad_clip_value"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            net.parameters(),
                            max_norm=config["grad_clip_value"],
                            norm_type=2
                        )

                    optimizer.step()

                # 更新EMA模型
                if self.ema is not None:
                    self.ema.update()

                running_loss += loss.item()
                if i % 100 == 99:
                    writer.add_scalar('Loss/train', running_loss / 100, global_step)
                    running_loss = 0.0

                if (i + 1) % 500 == 0:
                    throughputs.append(config["batch_size"] / (time.time() - batch_start_time))
                    print(
                        f'Epoch {epoch}, Batch: {i + 1} - Throughput: {throughputs[-1]:.5f} sample/s, Loss: {loss.item():.5f}')

                global_step += 1

            epoch_duration = time.time() - epoch_start_time

            # 使用EMA模型（如果启用）或原始模型进行评估
            train_acc, train_loss = self.evaluate(train_loader, self.evaluation_net, criterion)
            val_acc, val_loss = self.evaluate(val_loader, self.evaluation_net, criterion)
            print(
                f"Epoch {epoch} - Duration: {epoch_duration:.5f}s, Learning Rate: {current_lr:.6f}, \nTrain Acc: {train_acc:.5f}, Val Acc: {val_acc:.5f}\n")

            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)

            for name, param in net.named_parameters():
                writer.add_histogram(name, param, epoch)

            # 保存EMA模型的状态（如果启用）或原始模型的状态
            if self.ema is not None:
                model_to_save = self.ema.ema_model
            else:
                model_to_save = net

            # 传递给early stopping的是用于评估的模型
            early_stopping(val_loss, val_acc, epoch, optimizer, model_to_save)

            with open(csv_path, 'a+', newline='') as f:
                csv_write = csv.writer(f)
                data_row = [self.config["name"], epoch, train_acc, train_loss, val_acc, val_loss,
                            int(early_stopping.get_best_checkpoint()["epoch"]) == epoch, epoch_duration]
                csv_write.writerow(data_row)

            if early_stopping.early_stop:
                print("EarlyStopping: exiting loop.")
                break

        # 关键修复：确保使用early stopping中的最佳检查点
        self.best_checkpoint = early_stopping.get_best_checkpoint()
        self.total_train_time = time.time() - total_start_time
        self.model = self.evaluation_net  # 保存用于评估的模型（EMA或原始）

        self.weights_name = 'model_final_epoch_{}.pth'.format(self.best_checkpoint["epoch"])

        # 关键修复：直接保存最佳检查点中的模型状态
        torch.save(self.best_checkpoint['net_state_dict'], os.path.join(self.folder, self.weights_name))

        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0

        self.final_metrics = {
            "best_val_accuracy": self.best_checkpoint["best_val_acc"],
            "best_val_loss": self.best_checkpoint["best_val_loss"],
            "best_epoch": self.best_checkpoint["epoch"],
            "total_train_time": self.total_train_time,
            "avg_throughput_samples_sec": avg_throughput
        }
        print("Metrics:", self.final_metrics)

        writer.add_hparams(hparam_dict=config, metric_dict=self.final_metrics)
        writer.flush()
        writer.close()

        print("Finish Training!")

    def get_best_checkpoint(self):
        return self.best_checkpoint

    def get_model(self):
        return self.model

    def get_final_metrics(self):
        return self.final_metrics

    def get_weights_name(self):
        return self.weights_name