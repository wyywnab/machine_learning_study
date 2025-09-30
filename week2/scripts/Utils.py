import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
        score = -val_loss
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, optimizer, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, optimizer, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, epoch, optimizer, model):
        '''保存当前最佳模型'''
        self.best_model = model.state_dict()
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