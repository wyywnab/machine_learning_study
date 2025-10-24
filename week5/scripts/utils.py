import copy
import os

import numpy
import pandas
import seaborn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_data_root_folder():
    return os.path.join("..", "..", "cifar10_data")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, dir=''):
        """
        Args:
            patience (int): 等待多少个epoch没有提升后停止训练
            verbose (bool): 是否打印提示信息
            delta (float): 认为有提升的最小变化量
            dir (str): 模型保存目录
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_acc = 0
        self.val_loss_min = 1
        self.delta = delta
        self.dir = dir
        self.best_model = None
        self.best_epoch = -1
        self.best_checkpoint = {}

    def __call__(self, val_loss, val_acc, epoch, optimizer, model):
        # 基于准确率进行早停判断
        score = val_acc

        # 总是更新最佳准确率
        improved_acc = False
        if val_acc > self.best_val_acc + self.delta:
            self.best_val_acc = val_acc
            improved_acc = True

        # 首次运行
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, optimizer, model)
            return

        # 判断是否有提升
        if score > self.best_score + self.delta:
            # 有提升
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, optimizer, model)
            self.counter = 0
            if self.verbose:
                print(f'Validation accuracy improved to {val_acc:.4f}\n')
        elif improved_acc:
            # 准确率有提升但综合评分没有，也保存检查点（可选）
            self.save_checkpoint(val_loss, epoch, optimizer, model)
            self.counter = 0
            if self.verbose:
                print(f'Validation accuracy improved to {val_acc:.4f}\n')
        else:
            # 没有提升
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, epoch, optimizer, model):
        '''保存当前最佳模型'''
        # 关键修复：确保保存的是模型当前的状态字典
        #self.best_model = model.state_dict().copy()  # 使用copy确保保存当前状态
        self.best_model = copy.deepcopy(model.state_dict())
        self.best_epoch = epoch
        self.val_loss_min = val_loss
        checkpoint = {
            'epoch': epoch,
            'net_state_dict': self.best_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.val_loss_min
        }
        self.best_checkpoint = checkpoint
        # 可选：保存最佳检查点文件
        torch.save(checkpoint, os.path.join(self.dir, 'checkpoint_best_epoch_{}.pt'.format(epoch)))

    def get_best_checkpoint(self):
        return self.best_checkpoint

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", save_path=None, dpi=500, display=True):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param save_path: 是否保存，是则为保存路径save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontsize=5)

    if not save_path is None:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    if display:
        plt.show()


class ClassificationAnalyzer:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f'Class_{i}' for i in range(num_classes)]

        # 初始化统计变量
        self.reset()

    def reset(self):
        """重置统计"""
        self.all_predictions = []
        self.all_targets = []
        self.cm = numpy.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, predictions, targets):
        """更新预测和真实标签"""
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())

    def compute_per_class_metrics(self):
        """计算每个类别的指标"""
        predictions = numpy.array(self.all_predictions)
        targets = numpy.array(self.all_targets)

        # 计算混淆矩阵
        self.cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))

        # 为每个类别计算指标
        results = {}
        for i in range(self.num_classes):
            tp = self.cm[i, i]  # 真正例
            fp = numpy.sum(self.cm[:, i]) - tp  # 假正例（预测为i但实际不是）
            fn = numpy.sum(self.cm[i, :]) - tp  # 假负例（实际是i但预测不是）
            tn = numpy.sum(self.cm) - tp - fp - fn  # 真负例

            # 计算指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[self.class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn  # 该类别的样本数
            }

        return results

    def identify_underperforming_classes(self, results, f1_threshold=0.7, support_threshold=10):
        """识别性能较差的类别"""
        underperforming = []

        for class_name, metrics in results.items():
            f1 = metrics['f1']
            support = metrics['support']

            # 如果F1分数低于阈值且样本数足够（避免因样本太少导致的统计不可靠）
            if f1 < f1_threshold and support >= support_threshold:
                underperforming.append({
                    'class': class_name,
                    'f1_score': f1,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'support': support,
                    'reason': self.analyze_failure_reason(class_name)
                })

        # 按F1分数排序（从低到高）
        underperforming.sort(key=lambda x: x['f1_score'])
        return underperforming

    def analyze_failure_reason(self, class_name):
        """分析类别性能差的原因"""
        class_idx = self.class_names.index(class_name)

        # 分析混淆矩阵中该类别的错误模式
        row_sum = numpy.sum(self.cm[class_idx, :])
        if row_sum == 0:
            return "No samples in this class"

        # 最常见的错误预测
        errors = []
        for other_class in range(self.num_classes):
            if other_class != class_idx and self.cm[class_idx, other_class] > 0:
                error_rate = self.cm[class_idx, other_class] / row_sum
                errors.append((self.class_names[other_class], error_rate))

        # 按错误率排序
        errors.sort(key=lambda x: x[1], reverse=True)

        if errors and errors[0][1] > 0.3:  # 如果超过30%的错误都预测为某个特定类别
            return f"Often Mismatched with {errors[0][0]} ({errors[0][1]:.1%})"
        elif len(errors) > 1:
            return f"Scattered"
        else:
            return "Need Further Analysis"

    def generate_report(self, f1_threshold=0.7):
        """生成完整分析报告"""
        results = self.compute_per_class_metrics()
        underperforming = self.identify_underperforming_classes(results, f1_threshold)

        # 创建详细的数据框
        report_data = []
        for class_name, metrics in results.items():
            report_data.append({
                'Class': class_name,
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'Support': metrics['support'],
                'Status': '⚠️ Underperforming' if any(u['class'] == class_name for u in underperforming) else '✅ Normal'
            })

        df_report = pandas.DataFrame(report_data)

        return {
            'detailed_report': df_report,
            'underperforming_classes': underperforming,
            'confusion_matrix': self.cm
        }

    def plot_confusion_matrix(self, save_path, normalize=True, dpi=500):
        """绘制混淆矩阵热力图"""
        cm = self.cm.astype('float') if normalize else self.cm

        if normalize:
            cm = cm / cm.sum(axis=1)[:, numpy.newaxis]
            cm = numpy.nan_to_num(cm)  # 处理除零情况

        plt.figure(figsize=(12, 10))
        seaborn.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cmap='Blues')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        #plt.show()
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)