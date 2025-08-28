import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft2, fftshift
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import argparse
from torch.nn import functional as F

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ------------------------------
# 1. 特征捕获钩子类
# ------------------------------
class FeatureHook:
    """用于捕获模型中间层特征的钩子类"""

    def __init__(self):
        self.features = {
            'fsa_input': [],  # FSA模块输入特征
            'fsa_output': [],  # FSA模块输出特征
            'gca_input': [],  # GCA模块输入特征
            'gca_output': [],  # GCA模块输出特征
            'st_input': [],  # 空间Transformer输入特征
            'st_output': []  # 空间Transformer输出特征
        }

    def register_hooks(self, model):
        """为模型中的关键模块注册钩子"""
        # 遍历模型所有模块，为特定类型的模块注册钩子
        for name, module in model.named_modules():
            # 为频率语义注意力模块注册钩子
            if isinstance(module, FrequencySemanticAttention2D):
                module.register_forward_pre_hook(self._hook_fsa_input)
                module.register_forward_hook(self._hook_fsa_output)
            # 为门控卷积注意力模块注册钩子
            if isinstance(module, GatedCNNBlock):
                module.register_forward_pre_hook(self._hook_gca_input)
                module.register_forward_hook(self._hook_gca_output)
            # 为空间Transformer模块注册钩子
            if isinstance(module, SpatialTransformer):
                module.register_forward_pre_hook(self._hook_st_input)
                module.register_forward_hook(self._hook_st_output)

    def _hook_fsa_input(self, module, input):
        """捕获FSA模块的输入特征"""
        # 移出GPU以节省显存
        self.features['fsa_input'].append(input[0].detach().cpu())

    def _hook_fsa_output(self, module, input, output):
        """捕获FSA模块的输出特征"""
        self.features['fsa_output'].append(output.detach().cpu())

    def _hook_gca_input(self, module, input):
        """捕获GCA模块的输入特征"""
        self.features['gca_input'].append(input[0].detach().cpu())

    def _hook_gca_output(self, module, input, output):
        """捕获GCA模块的输出特征"""
        self.features['gca_output'].append(output.detach().cpu())

    def _hook_st_input(self, module, input):
        """捕获空间Transformer模块的输入特征"""
        self.features['st_input'].append(input[0].detach().cpu())

    def _hook_st_output(self, module, input, output):
        """捕获空间Transformer模块的输出特征"""
        self.features['st_output'].append(output.detach().cpu())

    def clear(self):
        """清空已捕获的特征"""
        for key in self.features:
            self.features[key] = []

    def get_features(self, module_name, batch_idx=0):
        """获取指定模块的特征"""
        if module_name not in self.features:
            raise ValueError(f"模块名 {module_name} 不存在")

        if len(self.features[module_name]) == 0:
            return None

        if len(self.features[module_name]) <= batch_idx:
            print(f"警告：批次索引{batch_idx}超出范围")
            return None

        return self.features[module_name][batch_idx]


# ------------------------------
# 2. 特征评估器（计算PSNR和噪声抑制比例）
# ------------------------------
class FeatureEvaluator:
    """特征评估器，用于计算PSNR和高频噪声抑制比例"""

    def __init__(self, cutoff_ratio=0.5):
        """
        初始化特征评估器
        :param cutoff_ratio: 高频截止比例，高于此比例的频率视为高频
        """
        self.cutoff_ratio = cutoff_ratio
        self.epsilon = 1e-8  # 防止除零错误

    def calculate_psnr(self, clean_feat, noisy_feat, data_range=1.0):
        """
        计算两个特征图之间的PSNR
        :param clean_feat: 干净特征 (B, C, H, W)
        :param noisy_feat: 带噪声特征/处理后特征 (B, C, H, W)
        :param data_range: 特征值范围（默认归一化到[0,1]）
        :return: PSNR值 (dB)
        """
        # 确保输入是张量
        if not isinstance(clean_feat, torch.Tensor):
            clean_feat = torch.tensor(clean_feat)
        if not isinstance(noisy_feat, torch.Tensor):
            noisy_feat = torch.tensor(noisy_feat)

        # 计算MSE
        mse = torch.mean((clean_feat - noisy_feat) ** 2)

        # 避免除零错误
        if mse < self.epsilon:
            return float('inf')

        # 计算PSNR
        psnr = 10 * torch.log10((data_range ** 2) / mse)
        return psnr.item()

    def get_high_freq_energy(self, feat):
        """
        计算特征图高频成分的能量占比
        :param feat: 输入特征 (B, C, H, W)
        :return: 高频能量占比的平均值
        """
        # 确保输入是numpy数组
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().detach().numpy()

        B, C, H, W = feat.shape
        high_energy = []

        for b in range(B):
            for c in range(C):
                # 获取单张特征图
                img = feat[b, c]

                # 傅里叶变换并移频（将低频移至中心）
                fft = fft2(img)
                fft_shift = fftshift(fft)

                # 计算幅度谱（能量相关）
                magnitude = np.abs(fft_shift)

                # 定义高频区域（中心区域为低频，边缘为高频）
                h_center, w_center = H // 2, W // 2
                h_cutoff = int(h_center * self.cutoff_ratio)
                w_cutoff = int(w_center * self.cutoff_ratio)

                # 高频区域掩码（中心以外的区域）
                mask = np.ones_like(magnitude)
                mask[h_center - h_cutoff:h_center + h_cutoff,
                w_center - w_cutoff:w_center + w_cutoff] = 0

                # 计算高频能量和总能量
                total_energy = np.sum(magnitude ** 2) + self.epsilon
                high_energy_val = np.sum((magnitude ** 2) * mask)

                high_energy_ratio = high_energy_val / total_energy
                high_energy.append(high_energy_ratio)

        # 返回平均高频能量占比
        return np.mean(high_energy)

    def evaluate_feature_improvement(self, clean_feat, noisy_feat, processed_feat):
        """
        评估特征改进效果：计算PSNR提升和高频噪声抑制比例
        :param clean_feat: 干净特征 (B, C, H, W)
        :param noisy_feat: 带噪声特征 (B, C, H, W)
        :param processed_feat: 处理后特征 (B, C, H, W)
        :return: 包含评估指标的字典
        """
        # 计算PSNR
        psnr_before = self.calculate_psnr(clean_feat, noisy_feat)
        psnr_after = self.calculate_psnr(clean_feat, processed_feat)
        psnr_gain = psnr_after - psnr_before

        # 计算高频能量
        high_energy_before = self.get_high_freq_energy(noisy_feat)
        high_energy_after = self.get_high_freq_energy(processed_feat)

        # 计算高频噪声抑制比例
        if high_energy_before < self.epsilon:
            noise_suppression = 0.0
        else:
            noise_suppression = (high_energy_before - high_energy_after) / high_energy_before * 100

        return {
            'psnr_before': psnr_before,
            'psnr_after': psnr_after,
            'psnr_gain': psnr_gain,
            'high_energy_before': high_energy_before,
            'high_energy_after': high_energy_after,
            'high_noise_suppression(%)': noise_suppression
        }


# ------------------------------
# 3. 特征可视化工具
# ------------------------------
class FeatureVisualizer:
    """特征可视化工具，用于生成特征图和频域谱的可视化结果"""

    def __init__(self, save_dir='visualization_results'):
        """
        初始化特征可视化工具
        :param save_dir: 可视化结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.pca = PCA(n_components=3)  # 用于多通道特征降维可视化

    def normalize_feat(self, feat):
        """将特征归一化到[0,1]范围以便可视化"""
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().detach().numpy()

        min_val = np.min(feat)
        max_val = np.max(feat)

        if max_val - min_val == 0:
            return np.zeros_like(feat)

        return (feat - min_val) / (max_val - min_val)

    def reduce_channels(self, feat):
        """将多通道特征通过PCA降维到3通道（模拟RGB）以便可视化"""
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().detach().numpy()

        B, C, H, W = feat.shape

        # 如果通道数小于等于3，直接返回
        if C <= 3:
            return self.normalize_feat(feat)

        # 重塑特征以便PCA
        feat_reshaped = feat.reshape(B, C, -1).transpose(0, 2, 1).reshape(-1, C)

        # 应用PCA
        feat_pca = self.pca.fit_transform(feat_reshaped)

        # 重塑回原始形状
        feat_pca = feat_pca.reshape(B, H, W, 3).transpose(0, 3, 1, 2)

        return self.normalize_feat(feat_pca)

    def plot_feature_comparison(self, clean_feat, noisy_feat, processed_feat,
                                sample_idx=0, channel=None, title="特征对比",
                                save_prefix="feature_comparison"):
        """
        绘制干净特征、带噪声特征和处理后特征的对比图
        :param clean_feat: 干净特征 (B, C, H, W)
        :param noisy_feat: 带噪声特征 (B, C, H, W)
        :param processed_feat: 处理后特征 (B, C, H, W)
        :param sample_idx: 样本索引
        :param channel: 要可视化的通道，None则使用PCA降维到3通道
        :param title: 图像标题
        :param save_prefix: 保存文件名前缀
        """
        # 提取单个样本
        clean = clean_feat[sample_idx]
        noisy = noisy_feat[sample_idx]
        processed = processed_feat[sample_idx]

        # 处理通道（单通道或PCA降维）
        if channel is not None:
            # 单通道可视化
            clean_vis = self.normalize_feat(clean[channel])
            noisy_vis = self.normalize_feat(noisy[channel])
            processed_vis = self.normalize_feat(processed[channel])
            cmap = 'gray'
        else:
            # PCA降维到3通道（RGB）
            clean_vis = self.reduce_channels(clean[np.newaxis, ...])[0].transpose(1, 2, 0)
            noisy_vis = self.reduce_channels(noisy[np.newaxis, ...])[0].transpose(1, 2, 0)
            processed_vis = self.reduce_channels(processed[np.newaxis, ...])[0].transpose(1, 2, 0)
            cmap = None

        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 干净特征
        im0 = axes[0].imshow(clean_vis, cmap=cmap)
        axes[0].set_title("干净特征", fontsize=14)
        axes[0].axis('off')
        if cmap is not None:  # 单通道添加颜色条
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # 带噪声特征
        im1 = axes[1].imshow(noisy_vis, cmap=cmap)
        axes[1].set_title("带噪声特征", fontsize=14)
        axes[1].axis('off')
        if cmap is not None:
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 处理后特征
        im2 = axes[2].imshow(processed_vis, cmap=cmap)
        axes[2].set_title("处理后特征", fontsize=14)
        axes[2].axis('off')
        if cmap is not None:
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=16)

        # 保存图像
        if channel is not None:
            save_path = os.path.join(self.save_dir, f"{save_prefix}_sample{sample_idx}_channel{channel}.png")
        else:
            save_path = os.path.join(self.save_dir, f"{save_prefix}_sample{sample_idx}_pca.png")

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"特征对比图已保存至: {save_path}")

        return save_path

    def plot_frequency_spectrum(self, noisy_feat, processed_feat,
                                sample_idx=0, channel=0, cutoff_ratio=0.5,
                                title="频域谱对比", save_prefix="frequency_spectrum"):
        """
        绘制带噪声特征和处理后特征的频域谱对比
        :param noisy_feat: 带噪声特征 (B, C, H, W)
        :param processed_feat: 处理后特征 (B, C, H, W)
        :param sample_idx: 样本索引
        :param channel: 要可视化的通道
        :param cutoff_ratio: 高频截止比例
        :param title: 图像标题
        :param save_prefix: 保存文件名前缀
        """
        # 提取单个样本和通道的特征
        if isinstance(noisy_feat, torch.Tensor):
            noisy = noisy_feat[sample_idx, channel].cpu().detach().numpy()
            processed = processed_feat[sample_idx, channel].cpu().detach().numpy()
        else:
            noisy = noisy_feat[sample_idx, channel]
            processed = processed_feat[sample_idx, channel]

        H, W = noisy.shape

        # 计算频域谱（幅度谱，取对数增强可视化）
        def get_spectrum(img):
            fft = fft2(img)
            fft_shift = fftshift(fft)  # 低频移至中心
            spectrum = np.log(np.abs(fft_shift) + 1e-8)  # 取对数
            return self.normalize_feat(spectrum)

        noisy_spec = get_spectrum(noisy)
        processed_spec = get_spectrum(processed)

        # 确定高频区域边界
        h_center, w_center = H // 2, W // 2
        h_cutoff = int(h_center * cutoff_ratio)
        w_cutoff = int(w_center * cutoff_ratio)

        # 创建图像
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(1, 2, figure=fig)

        # 带噪声特征的频域谱
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(noisy_spec, cmap='viridis')
        ax1.add_patch(plt.Rectangle(
            (w_center - w_cutoff, h_center - h_cutoff),
            2 * w_cutoff, 2 * h_cutoff,
            fill=False, edgecolor='red', linewidth=2, label='低频区域'
        ))
        ax1.set_title("带噪声特征的频域谱", fontsize=14)
        ax1.axis('off')
        ax1.legend(loc='upper right')

        # 处理后特征的频域谱
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(processed_spec, cmap='viridis')
        ax2.add_patch(plt.Rectangle(
            (w_center - w_cutoff, h_center - h_cutoff),
            2 * w_cutoff, 2 * h_cutoff,
            fill=False, edgecolor='red', linewidth=2, label='低频区域'
        ))
        ax2.set_title("处理后特征的频域谱", fontsize=14)
        ax2.axis('off')
        ax2.legend(loc='upper right')

        # 添加共享颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label('归一化幅度谱（对数）', rotation=270, labelpad=20)

        plt.suptitle(title, fontsize=16)

        # 保存图像
        save_path = os.path.join(self.save_dir, f"{save_prefix}_sample{sample_idx}_channel{channel}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"频域谱对比图已保存至: {save_path}")

        return save_path

    def plot_metrics_comparison(self, metrics_list, labels=None, title="特征改进指标对比",
                                save_name="metrics_comparison.png"):
        """
        绘制多个样本的PSNR提升和高频噪声抑制比例对比
        :param metrics_list: 指标列表，每个元素为一个评估指标字典
        :param labels: 每个样本的标签
        :param title: 图像标题
        :param save_name: 保存文件名
        """
        if not metrics_list:
            print("没有指标数据可可视化")
            return

        if labels is None:
            labels = [f"样本 {i + 1}" for i in range(len(metrics_list))]

        # 提取数据
        psnr_gains = [m['psnr_gain'] for m in metrics_list]
        noise_suppressions = [m['high_noise_suppression(%)'] for m in metrics_list]

        # 创建图像
        x = np.arange(len(labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 左侧轴：PSNR提升
        ax1.bar(x - width / 2, psnr_gains, width, label='PSNR提升 (dB)', color='skyblue')
        ax1.set_xlabel('样本', fontsize=12)
        ax1.set_ylabel('PSNR提升 (dB)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45)
        ax1.legend(loc='upper left')

        # 右侧轴：噪声抑制比例
        ax2 = ax1.twinx()
        ax2.bar(x + width / 2, noise_suppressions, width, label='高频噪声抑制 (%)', color='salmon', alpha=0.7)
        ax2.set_ylabel('高频噪声抑制 (%)', fontsize=12)
        ax2.legend(loc='upper right')

        plt.title(title, fontsize=16)
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"指标对比图已保存至: {save_path}")

        return save_path


# ------------------------------
# 4. 模型定义（FSC-Net完整实现）
# ------------------------------
class FrequencySemanticAttention2D(nn.Module):
    """频率语义注意力模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # 卷积层用于特征降维
        self.conv_reduce = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

        # MLP用于生成注意力权重
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

        # 池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # 傅里叶变换
        fft = torch.fft.rfft2(x, norm='ortho')
        real, imag = fft.real, fft.imag

        # 对实部特征进行池化和拼接
        real_avg = self.avg_pool(real)
        real_max = self.max_pool(real)
        real_cat = torch.cat([real_avg, real_max], dim=1)

        # 对虚部特征进行池化和拼接
        imag_avg = self.avg_pool(imag)
        imag_max = self.max_pool(imag)
        imag_cat = torch.cat([imag_avg, imag_max], dim=1)

        # 特征降维和权重生成
        real_reduced = self.conv_reduce(real_cat).squeeze(-1).squeeze(-1)
        imag_reduced = self.conv_reduce(imag_cat).squeeze(-1).squeeze(-1)

        real_weights = self.mlp(real_reduced)[:, :, None, None]
        imag_weights = self.mlp(imag_reduced)[:, :, None, None]

        # 应用注意力权重
        real_attended = real * real_weights
        imag_attended = imag * imag_weights

        # 逆傅里叶变换
        fft_attended = torch.complex(real_attended, imag_attended)
        output = torch.fft.irfft2(fft_attended, s=x.shape[-2:], norm='ortho')

        return output


class GatedCNNBlock(nn.Module):
    """门控卷积注意力模块"""

    def __init__(self, dim, expansion_ratio=8 / 3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, drop_path=0.01):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)

        # 全连接层
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()

        # 卷积通道数
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)

        # 深度卷积
        self.conv = nn.Conv2d(conv_channels, conv_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              groups=conv_channels)

        # 输出投影
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        # 通道维度转换
        self.to_channels_last = lambda x: x.permute(0, 2, 3, 1)
        self.to_channels_first = lambda x: x.permute(0, 3, 1, 2)

    def forward(self, x):
        # 转换为通道最后格式
        x = self.to_channels_last(x)
        shortcut = x

        # 归一化和线性变换
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)

        # 卷积处理
        c = self.to_channels_first(c)
        c = self.conv(c)
        c = self.to_channels_last(c)

        # 门控机制和输出
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        out = x + shortcut

        # 转换回通道优先格式并应用dropout
        out = self.to_channels_first(out)
        out = self.drop_path(out)

        return out


class SpatialTransformer(nn.Module):
    """空间Transformer模块"""

    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, 3, padding=1, groups=channel)
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, x):
        return x + torch.tanh(self.norm(self.conv(x)))


class DEE_module(nn.Module):
    """双通道增强模块"""

    def __init__(self, channel, ablation='all'):
        super(DEE_module, self).__init__()
        self.ablation = ablation

        # 多尺度卷积
        self.FC11 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC12 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC13 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC1 = nn.Conv2d(channel // 4, channel, kernel_size=1)

        self.FC21 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC22 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC23 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC2 = nn.Conv2d(channel // 4, channel, kernel_size=1)

        #  dropout层
        self.dropout = nn.Dropout(p=0.01)

        # 注意力模块
        self.gate1 = GatedCNNBlock(dim=channel // 4, kernel_size=7, conv_ratio=0.5)
        self.gate2 = GatedCNNBlock(dim=channel // 4, kernel_size=7, conv_ratio=0.5)
        self.spitial1 = SpatialTransformer(channel=channel // 4)
        self.spitial2 = SpatialTransformer(channel=channel // 4)
        self.freq_attn1 = FrequencySemanticAttention2D(channel // 4)
        self.freq_attn2 = FrequencySemanticAttention2D(channel // 4)

    def forward(self, x):
        # 第一个分支
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x)) / 3

        # 根据消融实验配置应用不同模块
        if self.ablation == 'all':
            x1 = self.spitial1(self.gate1(self.freq_attn1(x1)))
        elif self.ablation == 'fsa':
            x1 = self.freq_attn1(x1)
        elif self.ablation == 'gca':
            x1 = self.gate1(x1)
        elif self.ablation == 'st':
            x1 = self.spitial1(x1)
        elif self.ablation == 'fsa_gca':
            x1 = self.gate1(self.freq_attn1(x1))
        elif self.ablation == 'fsa_st':
            x1 = self.spitial1(self.freq_attn1(x1))
        elif self.ablation == 'gca_st':
            x1 = self.spitial1(self.gate1(x1))

        x1 = self.FC1(F.relu(x1))

        # 第二个分支
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x)) / 3

        # 根据消融实验配置应用不同模块
        if self.ablation == 'all':
            x2 = self.spitial2(self.gate2(self.freq_attn2(x2)))
        elif self.ablation == 'fsa':
            x2 = self.freq_attn2(x2)
        elif self.ablation == 'gca':
            x2 = self.gate2(x2)
        elif self.ablation == 'st':
            x2 = self.spitial2(x2)
        elif self.ablation == 'fsa_gca':
            x2 = self.gate2(self.freq_attn2(x2))
        elif self.ablation == 'fsa_st':
            x2 = self.spitial2(self.freq_attn2(x2))
        elif self.ablation == 'gca_st':
            x2 = self.spitial2(self.gate2(x2))

        x2 = self.FC2(F.relu(x2))

        # 拼接输出（沿通道维度）
        out = torch.cat((x, x1, x2), dim=1)
        out = self.dropout(out)

        return out


# 完整的ResNet50基础网络
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """完整的ResNet50基础网络"""

    def __init__(self, pretrained=True, last_conv_stride=1, last_conv_dilation=1):
        self.inplanes = 64
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=last_conv_stride,
                                       dilation=last_conv_dilation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class visible_module(nn.Module):
    """可见光特征提取模块"""

    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()
        self.visible = ResNet50(pretrained=True, last_conv_stride=1)

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    """热红外特征提取模块"""

    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()
        self.thermal = ResNet50(pretrained=True, last_conv_stride=1)

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    """基础ResNet网络"""

    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()
        self.base = ResNet50(pretrained=True, last_conv_stride=1)
        self.base.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class CNL(nn.Module):
    """跨网络连接模块"""

    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(high_dim), )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                   nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)
        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        return W_y + x_h


class PNL(nn.Module):
    """渐进式网络连接模块"""

    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio
        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)
        theta_x = self.theta(x_h).view(B, self.low_dim, -1).permute(0, 2, 1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1)
        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous().view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        return W_y + x_h


class MFA_block(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()
        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z


class Normalize(nn.Module):
    """特征归一化层"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        return x.div(norm)


def weights_init_kaiming(m):
    """Kaiming初始化"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.zeros_(m.bias.data)


def weights_init_classifier(m):
    """分类器初始化"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            nn.init.zeros_(m.bias.data)


class embed_net(nn.Module):
    """嵌入网络"""

    def __init__(self, class_num, dataset, arch='resnet50', ablation='all'):
        super(embed_net, self).__init__()
        self.ablation = ablation
        self.dataset = dataset

        # 模态特征提取
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        # 根据数据集设置参数
        if self.dataset == 'regdb':
            self.pool_dim = 1024
            self.DEE = DEE_module(512, ablation=self.ablation)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)
        else:
            self.pool_dim = 2048
            self.DEE = DEE_module(1024, ablation=self.ablation)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)
            self.MFA3 = MFA_block(1024, 512, 1)

        # 瓶颈层和分类器
        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal=0):
        # 根据模态选择输入
        if modal == 0:  # 双模态
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:  # 可见光模态
            x = self.visible_module(x1)
        elif modal == 2:  # 热红外模态
            x = self.thermal_module(x2)

        x_ = x
        x = self.base_resnet.base.layer1(x_)
        x_ = self.MFA1(x, x_)
        x = self.base_resnet.base.layer2(x_)
        x_ = self.MFA2(x, x_)

        if self.dataset == 'regdb':
            x_ = self.DEE(x_)
            x = self.base_resnet.base.layer3(x_)
        else:
            x = self.base_resnet.base.layer3(x_)
            x_ = self.MFA3(x, x_)
            x_ = self.DEE(x_)
            x = self.base_resnet.base.layer4(x_)

        xp = self.avgpool(x)
        x_pool = xp.view(xp.size(0), xp.size(1))
        feat = self.bottleneck(x_pool)

        if self.training:
            xps = xp.view(xp.size(0), xp.size(1), xp.size(2)).permute(0, 2, 1)
            xp1, xp2, xp3 = torch.chunk(xps, 3, 0)
            xpss = torch.cat((xp2, xp3), 1)
            loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal=1).sum() / (xp.size(0))
            return x_pool, self.classifier(feat), loss_ort
        else:
            return self.l2norm(x_pool), self.l2norm(feat)


# ------------------------------
# 5. 数据加载和测试代码
# ------------------------------
def fliplr(img):
    """水平翻转图像"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# 数据管理器和加载器（根据实际数据集实现）
class TestData(data.Dataset):
    """测试数据集"""

    def __init__(self, img_path, label, transform=None, img_size=(144, 384)):
        self.img_path = img_path
        self.label = label
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index):
        img = plt.imread(self.img_path[index])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        label = self.label[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.label)


def process_query_sysu(data_path, mode='all'):
    """处理SYSU-MM01数据集的查询集"""
    ir_cameras = ['cam3', 'cam6']
    rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as f:
        ids = f.read().splitlines()
        ids = [int(y) for y in ids[0:]]

    for id in ids:
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, str(id))
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.extend(new_files)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, str(id))
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)

    query_img = files_ir
    query_label = [int(img_path.split('/')[-2]) for img_path in query_img]
    query_cam = [rgb_cameras.index(cam) if cam in rgb_cameras else len(rgb_cameras) + ir_cameras.index(cam)
                 for img_path in query_img for cam in [img_path.split('/')[-3]]]

    return query_img, query_label, query_cam


def process_gallery_sysu(data_path, mode='all', trial=0):
    """处理SYSU-MM01数据集的画廊集"""
    rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    with open(file_path, 'r') as f:
        ids = f.read().splitlines()
        ids = [int(y) for y in ids[0:]]

    # 加载测试图像
    gall_img = []
    gall_label = []
    gall_cam = []

    for id in ids:
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, str(id))
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                gall_img.extend(new_files)
                gall_label.extend([id] * len(new_files))
                gall_cam.extend([rgb_cameras.index(cam)] * len(new_files))

    return gall_img, gall_label, gall_cam


def extract_features_with_analysis(model, dataloader, clean_loader, feature_hook,
                                   evaluator, visualizer, num_vis_samples=5,
                                   modal=1):
    """
    提取特征并进行分析和可视化
    :param model: 模型
    :param dataloader: 数据加载器
    :param clean_loader: 干净样本加载器
    :param feature_hook: 特征钩子
    :param evaluator: 特征评估器
    :param visualizer: 特征可视化工具
    :param num_vis_samples: 要可视化的样本数量
    :param modal: 模态
    :return: 提取的特征和评估指标
    """
    model.eval()
    device = next(model.parameters()).device
    metrics_list = []

    # 获取干净样本作为基准
    clean_inputs, clean_labels = next(iter(clean_loader))
    clean_inputs = clean_inputs.to(device)

    # 获取干净样本的特征作为基准
    feature_hook.clear()
    with torch.no_grad():
        _, _ = model(clean_inputs, clean_inputs, modal=modal)
        clean_fsa_feats = feature_hook.get_features('fsa_input')

    # 处理测试数据
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= 1:  # 只处理第一个批次用于可视化
                break

            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            num_vis = min(num_vis_samples, batch_size)

            # 清除之前的特征
            feature_hook.clear()

            # 前向传播，捕获特征
            _, _ = model(inputs, inputs, modal=modal)

            # 获取FSA模块的输入和输出特征
            fsa_input_feats = feature_hook.get_features('fsa_input')
            fsa_output_feats = feature_hook.get_features('fsa_output')

            if fsa_input_feats is None or fsa_output_feats is None:
                print("无法获取FSA模块的特征，跳过可视化")
                continue

            # 对每个样本进行分析和可视化
            for i in range(num_vis):
                # 确保有对应的干净特征
                if i >= len(clean_fsa_feats):
                    continue

                # 提取单个样本的特征
                clean_feat = clean_fsa_feats[i:i + 1]
                noisy_feat = fsa_input_feats[i:i + 1]
                processed_feat = fsa_output_feats[i:i + 1]

                # 评估特征改进
                metrics = evaluator.evaluate_feature_improvement(
                    clean_feat, noisy_feat, processed_feat
                )
                metrics_list.append(metrics)

                print(f"样本 {i} 评估结果:")
                print(f"  PSNR提升: {metrics['psnr_gain']:.2f} dB")
                print(f"  高频噪声抑制比例: {metrics['high_noise_suppression(%)']:.2f}%")

                # 可视化特征图（单通道和PCA）
                visualizer.plot_feature_comparison(
                    clean_feat, noisy_feat, processed_feat,
                    sample_idx=0, channel=0,  # 可视化第0个通道
                    title=f"样本 {i} 的FSA模块特征对比 (PSNR提升: {metrics['psnr_gain']:.2f}dB)",
                    save_prefix="fsa_feature"
                )

                visualizer.plot_feature_comparison(
                    clean_feat, noisy_feat, processed_feat,
                    sample_idx=0, channel=None,  # 使用PCA降维
                    title=f"样本 {i} 的FSA模块特征对比 (PCA降维)",
                    save_prefix="fsa_feature"
                )

                # 可视化频域谱
                visualizer.plot_frequency_spectrum(
                    noisy_feat, processed_feat,
                    sample_idx=0, channel=0,
                    cutoff_ratio=evaluator.cutoff_ratio,
                    title=f"样本 {i} 的频域谱对比 (噪声抑制: {metrics['high_noise_suppression(%)']:.2f}%)",
                    save_prefix="fsa_spectrum"
                )

            # 清理显存
            torch.cuda.empty_cache()

        # 绘制所有样本的指标对比图
        if metrics_list:
            visualizer.plot_metrics_comparison(
                metrics_list,
                title="所有样本的特征改进指标对比"
            )

    return metrics_list


# ------------------------------
# 6. 主函数
# ------------------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='特征分析与可视化工具')
    parser.add_argument('--dataset', default='sysu', help='数据集名称: sysu, regdb 或 llcm')
    parser.add_argument('--data_path', default='/root/autodl-tmp/SYSU-MM01/', help='数据集路径')
    parser.add_argument('--model_path', default='', help='模型权重路径')
    parser.add_argument('--save_dir', default='./feature_visualization', help='可视化结果保存目录')
    parser.add_argument('--num_vis_samples', type=int, default=5, help='要可视化的样本数量')
    parser.add_argument('--cutoff_ratio', type=float, default=0.5, help='高频截止比例')
    parser.add_argument('--modal', type=int, default=2, help='模态选择: 0-双模态, 1-可见光, 2-热红外')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号，-1表示使用CPU')
    args = parser.parse_args()

    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"使用GPU: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        print("使用CPU")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化特征分析组件
    feature_hook = FeatureHook()
    evaluator = FeatureEvaluator(cutoff_ratio=args.cutoff_ratio)
    visualizer = FeatureVisualizer(save_dir=args.save_dir)

    # 加载模型
    if args.dataset == 'regdb':
        class_num = 206
    elif args.dataset == 'sysu':
        class_num = 395
    else:  # llcm
        class_num = 467

    model = embed_net(class_num, args.dataset)
    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"已加载模型权重: {args.model_path}")
    model = model.to(device)

    # 注册特征钩子
    feature_hook.register_hooks(model)

    # 准备数据
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 144)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 根据数据集加载数据
    if args.dataset == 'sysu':
        query_img, query_label, query_cam = process_query_sysu(args.data_path)
        gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_path)
        test_img = query_img + gall_img
        test_label = query_label + gall_label
    else:
        # 这里可以添加其他数据集的加载逻辑
        raise NotImplementedError(f"数据集 {args.dataset} 的加载逻辑尚未实现")

    # 创建数据集和数据加载器
    test_dataset = TestData(test_img, test_label, transform=transform_test)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.num_vis_samples,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # 获取干净样本（这里简单使用测试集的前N个样本作为干净样本）
    clean_loader = data.DataLoader(
        data.Subset(test_dataset, range(min(args.num_vis_samples, len(test_dataset)))),
        batch_size=args.num_vis_samples,
        shuffle=False,
        num_workers=4
    )

    # 提取特征并进行分析
    print("开始特征提取和分析...")
    metrics_list = extract_features_with_analysis(
        model,
        test_loader,
        clean_loader,
        feature_hook,
        evaluator,
        visualizer,
        num_vis_samples=args.num_vis_samples,
        modal=args.modal
    )

    # 打印总体统计结果
    if metrics_list:
        avg_psnr_gain = np.mean([m['psnr_gain'] for m in metrics_list])
        avg_suppression = np.mean([m['high_noise_suppression(%)'] for m in metrics_list])
        print("\n特征分析总体结果:")
        print(f"平均PSNR提升: {avg_psnr_gain:.2f} dB")
        print(f"平均高频噪声抑制比例: {avg_suppression:.2f}%")
        print(f"所有可视化结果已保存至: {args.save_dir}")


if __name__ == '__main__':
    main()