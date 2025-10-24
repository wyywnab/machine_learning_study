import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) 类
    用于生成深度学习模型的注意力热力图
    """

    def __init__(self, model, target_layer):
        """
        初始化Grad-CAM

        参数:
            model: PyTorch模型
            target_layer: 目标层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子来捕获梯度和激活值
        self._register_hooks()

    def _register_hooks(self):
        """注册前向和反向钩子来捕获目标层的激活值和梯度"""

        def forward_hook(module, input, output):
            """前向钩子：保存激活值"""
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """反向钩子：保存梯度"""
            self.gradients = grad_output[0].detach()

        # 使用新的 full_backward_hook 替代旧的 backward_hook
        self.target_layer.register_forward_hook(forward_hook)

        # 检查 PyTorch 版本并选择相应的钩子注册方法
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.target_layer.register_full_backward_hook(backward_hook)
        else:
            # 旧版本兼容
            self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        """
        生成类别激活映射（CAM）
        """
        # 确保模型处于训练模式以启用梯度计算
        self.model.train()

        # 确保输入张量需要梯度
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        # 前向传播
        output = self.model(input_tensor)

        # 1. 确定实际的预测类别
        predicted_class_idx = torch.argmax(output, dim=1).item()

        # 2. 确定用于生成CAM的目标类别
        if target_class is None:
            cam_target_class = predicted_class_idx # 如果未指定，则使用预测类别
        else:
            cam_target_class = target_class       # 否则，使用指定的类别 (例如真实标签)

        # 清零梯度
        self.model.zero_grad()

        # 创建one-hot向量用于反向传播
        one_hot = torch.zeros_like(output)
        one_hot[0, cam_target_class] = 1 # <--- 使用 cam_target_class

        try:
            # 反向传播计算梯度
            output.backward(gradient=one_hot, retain_graph=True)
        except Exception as e:
            print(f"反向传播失败: {e}")
            # 尝试替代方法
            self.model.zero_grad()
            loss = torch.sum(output * one_hot)
            loss.backward(retain_graph=True)

        # 检查是否成功获取梯度和激活值
        if self.gradients is None or self.activations is None:
            raise RuntimeError("未能捕获梯度或激活值，请检查目标层设置")

        # 计算权重：对梯度进行全局平均池化
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 加权组合特征图
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # 应用ReLU，只保留正贡献
        cam = F.relu(cam)

        # 归一化到[0, 1]范围
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # 上采样到输入图像尺寸
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 3. 返回CAM和 *实际的预测类别*
        return cam.squeeze().detach().cpu().numpy(), predicted_class_idx

    def superimpose_heatmap(self, original_image, heatmap, alpha=0.5):
        """
        将热力图叠加到原始图像上

        参数:
            original_image: 原始图像 (height, width, channels)
            heatmap: 热力图
            alpha: 热力图透明度

        返回:
            superimposed_img: 叠加后的图像
        """
        # 确保热力图是2D的
        if len(heatmap.shape) == 3:
            heatmap = heatmap.squeeze()

        # 调整热力图尺寸匹配原图（如果需要）
        if heatmap.shape != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        # 将热力图转换为彩色图
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 如果原图是RGB，需要转换颜色顺序
        if original_image.shape[-1] == 3:
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 叠加热力图和原图
        superimposed_img = heatmap_colored * alpha + original_image * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return superimposed_img


class GradCAMVisualizer:
    """Grad-CAM可视化工具类"""

    def __init__(self, model, target_layer, class_names):
        self.grad_cam = GradCAM(model, target_layer)
        self.class_names = class_names

    def visualize(self, input_tensor, original_image, target_class=None,
                  save_path=None, figsize=(15, 5)):
        """
        完整的可视化流程

        参数:
            input_tensor: 模型输入张量
            original_image: 原始图像（用于显示）
            target_class: 目标类别
            save_path: 保存路径
            figsize: 图像尺寸
        """
        # 生成热力图
        heatmap, predicted_class = self.grad_cam.generate_cam(input_tensor, target_class)

        # 转换原始图像为numpy数组用于显示
        if isinstance(original_image, torch.Tensor):
            img = self.tensor_to_numpy(original_image)
        else:
            img = original_image

        # 确保图像值在[0, 255]范围内
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        # 叠加热力图
        superimposed_img = self.grad_cam.superimpose_heatmap(img, heatmap)

        # 创建可视化图像
        plt.figure(figsize=figsize)

        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f'Original Image\nTrue: {self.class_names[target_class] if target_class is not None else "Unknown"}')
        plt.axis('off')

        # 热力图
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        plt.colorbar()

        # 叠加图像
        plt.subplot(1, 3, 3)
        plt.imshow(superimposed_img)
        plt.title(f'Superimposed\nPredicted: {self.class_names[predicted_class]}')
        plt.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        #plt.show()

        return heatmap, predicted_class

    def tensor_to_numpy(self, tensor):
        """将张量转换为numpy数组用于显示"""
        img = tensor.cpu().numpy().transpose(1, 2, 0)

        # 反归一化（假设使用ImageNet的均值和标准差）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        return (img * 255).astype(np.uint8)