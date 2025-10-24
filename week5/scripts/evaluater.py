import os

import numpy
import torch
from torchvision.datasets import CIFAR10

from utils import draw_confusion_matrix, ClassificationAnalyzer
from grad_cam import GradCAMVisualizer
from utils import get_data_root_folder


def save_confusion_matrix_and_accuracy(data_loader, net, epoch, exp_name, current_folder, device):
    label_true = []
    label_pred = []
    map = [[0] * 10 for _ in range(10)]

    net.eval()

    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = net.forward(x)
            predicted = torch.argmax(outputs, dim=1)

            for i in range(len(y)):
                true_label = int(y[i])
                pred_label = int(predicted[i])
                label_true.append(true_label)
                label_pred.append(pred_label)
                map[true_label][pred_label] += 1

    label_name = CIFAR10(data_root_folder, download=True, train=True).classes

    draw_confusion_matrix(label_true, label_pred, label_name, display=False,
                          title=f'Confusion Matrix of {exp_name} Epoch {epoch}',
                          save_path=os.path.join(current_folder,
                                                 f'confusion_matrix_final_{exp_name}_epoch_{epoch}.png'))

    accuracy_single = {}
    correct_sum = 0
    total_sum = 0
    for i in range(len(map)):
        row_sum = sum(map[i])
        total_sum += row_sum
        accuracy_single[label_name[i]] = map[i][i] / row_sum if row_sum > 0 else 0
        correct_sum += map[i][i]

    overall_accuracy = correct_sum / total_sum if total_sum > 0 else 0

    return accuracy_single, overall_accuracy


def evaluate_with_detailed_analysis(model, test_loader, class_names, folder, device='cuda'):
    """在模型评估时进行详细分析"""
    analyzer = ClassificationAnalyzer(num_classes=len(class_names), class_names=class_names)

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = torch.argmax(output, dim=1)

            analyzer.update(predictions, target)

    # 生成报告
    report = analyzer.generate_report(f1_threshold=0.7)
    overall_acc = float(numpy.mean(numpy.array(analyzer.all_predictions) == numpy.array(analyzer.all_targets)))

    # 打印摘要
    print()
    print("=" * 60)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    """print(f"\nUnderperforming Classes (F1 < 0.7):")
    print("-" * 40)

    for issue in report['underperforming_classes']:
        print(f"🔴 {issue['class']}: F1={issue['f1_score']:.4f}, "
              f"P={issue['precision']:.4f}, R={issue['recall']:.4f}")
        print(f"   Reason: {issue['reason']}")
        print()"""

    # 显示详细表格
    print("Detailed Per-Class Metrics:")
    print("-" * 60)
    print(report['detailed_report'].to_string(index=False))

    # 绘制混淆矩阵
    analyzer.plot_confusion_matrix(os.path.join(folder,f'confusion_matrix.png'), normalize=True)

    return report, overall_acc


def evaluate_grad_cam(model, class_names, test_loader, folder, device):
    model.eval()

    target_layer = model.layer4[-1].left[3]
    visualizer = GradCAMVisualizer(model, target_layer, class_names)

    num_examples = 5
    examples_shown = 0

    print("\nStart Genereting Grad-CAM Visuals...")
    print("Num\t\tReal\t\t\tPredict")

    for i, (images, labels) in enumerate(test_loader):
        if examples_shown >= num_examples:
            break

        # 处理当前batch
        for j in range(len(images)):
            if examples_shown >= num_examples:
                break

            print(f"{examples_shown + 1}/{num_examples}", end="\t\t")
            print(f"{class_names[labels[j].item()]}", end="\t\t\t")

            try:
                # 创建单个图像的batch，并启用梯度计算
                single_image = images[j:j + 1].clone().to(device)
                single_image.requires_grad_(True)  # 启用梯度计算

                # 确保模型处于训练模式
                model.eval()

                heatmap, predicted_class = visualizer.visualize(
                    input_tensor=single_image,
                    original_image=images[j].cpu(),  # 使用CPU上的图像用于显示
                    target_class=labels[j].item(),
                    save_path=os.path.join(folder, f'grad_cam_example_{examples_shown + 1}.png')
                )

                print(f"{class_names[predicted_class]}")
                examples_shown += 1

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()  # 打印完整的错误堆栈
                continue

    # 恢复模型到评估模式
    model.eval()
    print("Grad-CAM Visualization Finished.")

data_root_folder = get_data_root_folder()