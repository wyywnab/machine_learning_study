import os
import time
import torch
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from models.cnn_small import SmallCNN, SmallCNNCBAM
from models.resnet18_small import SmallResNet
from Utils import EarlyStopping

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
    
        if config["model"] == "SmallCNN":
            if config["cbam_enabled"]:
                net = SmallCNNCBAM()
            else:
                net = SmallCNN()
        elif config["model"] == "SmallResNet":
            net = SmallResNet(config["cbam_enabled"])
        else:
            print("Illegal Model Name!")
            exit(0)
        net = net.to(self.device)
    
        #optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
        #optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        if config["lr_scheduler"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config["t_max"],
                eta_min=config["eta_min"]
            )
        elif config["lr_scheduler"] == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=config["step_size"],
                gamma=config["gamma"]
            )
        else:
            scheduler = None
    
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        scaler = torch.amp.GradScaler()
        early_stopping = EarlyStopping(verbose=True, dir=self.folder, patience=config["early_stopping_patience"])
    
        total_start_time = time.time()
    
        #train_loader, val_loader, test_loader = self.get_data_loaders(config["batch_size"], config["seed"], config["num_workers"])
        train_loader = self.train_loader
        val_loader = self.val_loader
    
        init_train_acc, init_train_loss = self.evaluate(train_loader, net, criterion)
        init_val_acc, init_val_loss = self.evaluate(val_loader, net, criterion)
        #init_test_acc, init_test_loss = self.evaluate(test_loader, net, criterion, device)
        print(f"Initial Train Accuracy: {init_train_acc:.5f}, Validation Accuracy: {init_val_acc:.5f}")
    
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
    
        global_step = 0
        throughputs = []
        for epoch in range(1, config["max_epoch"] + 1):
            #if (epoch == 5): break
    
            epoch_start_time = time.time()
            net.train()
            running_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                batch_start_time = time.time()
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
    
                if config["amp_enabled"]:
                    with torch.amp.autocast(self.device):
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
    
                if (i + 1) % 500 == 0:
                    throughputs.append(config["batch_size"] / (time.time() - batch_start_time))
                    print(f'Epoch {epoch}, Batch: {i + 1} - Throughput: {throughputs[-1]:.5f} sample/s, Loss: {loss.item():.5f}')
    
                global_step += 1
    
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
    
            epoch_duration = time.time() - epoch_start_time
    
            train_acc, train_loss = self.evaluate(train_loader, net, criterion)
            val_acc, val_loss = self.evaluate(val_loader, net, criterion)
            print(f"Epoch {epoch} - Duration: {epoch_duration:.5f}s, Learning Rate: {current_lr},\n Train Accuracy: {train_acc:.5f}, Validation Accuracy: {val_acc:.5f}")

            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

            for name, param in net.named_parameters():
                writer.add_histogram(name, param, epoch)

            early_stopping(val_loss, val_acc, epoch, optimizer, net)

            with open(csv_path, 'a+', newline='') as f:
                csv_write = csv.writer(f)
                data_row = [self.config["name"], epoch, train_acc, train_loss, val_acc, val_loss,
                            int(early_stopping.get_best_checkpoint()["epoch"]) == epoch, epoch_duration]
                csv_write.writerow(data_row)

            if early_stopping.early_stop:
                print("EarlyStopping: exiting loop.")
                break
    
        self.best_checkpoint = early_stopping.get_best_checkpoint()
        self.total_train_time = time.time() - total_start_time
        self.model = net
    
        final_model_path = os.path.join(self.folder, 'model_final_epoch_{}.pth').format(self.best_checkpoint["epoch"])
        torch.save(self.best_checkpoint["net_state_dict"], final_model_path)

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
    
        print("Finished!")
        
    def get_best_checkpoint(self):
        return self.best_checkpoint

    def get_model(self):
        return self.model

    def get_final_metrics(self):
        return self.final_metrics
