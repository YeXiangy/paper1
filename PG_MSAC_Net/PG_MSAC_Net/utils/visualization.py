"""
可视化工具模块
提供训练过程可视化、特征可视化、注意力可视化等功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# 设置默认样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_curves(train_history, val_history=None, metrics=['loss', 'accuracy'],
                         save_path=None, show_best=True):
    """
    绘制训练曲线

    Args:
        train_history: 训练历史字典 {'loss': [], 'accuracy': [], ...}
        val_history: 验证历史字典
        metrics: 要绘制的指标列表
        save_path: 保存路径
        show_best: 是否标记最佳点

    Returns:
        fig: matplotlib图形对象
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 绘制训练曲线
        if metric in train_history:
            epochs = range(1, len(train_history[metric]) + 1)
            ax.plot(epochs, train_history[metric], 'b-', linewidth=2, label=f'Train {metric}')

            # 标记最佳点
            if show_best:
                if metric == 'loss':
                    best_idx = np.argmin(train_history[metric])
                    best_value = train_history[metric][best_idx]
                else:
                    best_idx = np.argmax(train_history[metric])
                    best_value = train_history[metric][best_idx]

                ax.scatter(best_idx + 1, best_value, color='blue', s=100, zorder=5)
                ax.annotate(f'Best: {best_value:.4f}',
                            xy=(best_idx + 1, best_value),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

        # 绘制验证曲线
        if val_history and metric in val_history:
            epochs = range(1, len(val_history[metric]) + 1)
            ax.plot(epochs, val_history[metric], 'r--', linewidth=2, label=f'Val {metric}')

            # 标记验证最佳点
            if show_best:
                if metric == 'loss':
                    best_idx = np.argmin(val_history[metric])
                    best_value = val_history[metric][best_idx]
                else:
                    best_idx = np.argmax(val_history[metric])
                    best_value = val_history[metric][best_idx]

                ax.scatter(best_idx + 1, best_value, color='red', s=100, zorder=5)
                ax.annotate(f'Val Best: {best_value:.4f}',
                            xy=(best_idx + 1, best_value),
                            xytext=(10, -15), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} vs Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    return fig


def plot_confusion_matrix_advanced(y_true, y_pred, class_names, normalize=False,
                                   save_path=None, figsize=(10, 8)):
    """
    绘制增强版混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        normalize: 是否归一化
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        fig: matplotlib图形对象
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 主混淆矩阵
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(title)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    # 每类别的准确率条形图
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    bars = ax2.bar(range(len(class_names)), class_accuracy,
                   color=plt.cm.Blues(np.linspace(0.4, 0.8, len(class_names))))

    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    ax2.set_title('Per-Class Accuracy')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    return fig


def plot_feature_tsne(features, labels, class_names=None, perplexity=30,
                      save_path=None, figsize=(12, 8)):
    """
    绘制特征的t-SNE可视化

    Args:
        features: 特征矩阵 [n_samples, n_features]
        labels: 标签 [n_samples]
        class_names: 类别名称列表
        perplexity: t-SNE参数
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        fig: matplotlib图形对象
    """
    # 确保features是numpy数组
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    # 如果特征维度太高，先用PCA降维
    if features.shape[1] > 50:
        print(f"High dimensional features ({features.shape[1]}D), applying PCA first...")
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)

    print("Computing t-SNE embedding...")
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_jobs=-1, verbose=1)
    features_2d = tsne.fit_transform(features)

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 设置类别名称
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(labels)))]

    # 颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

    # 子图1: 按类别着色的散点图
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        if np.any(mask):
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=[color], label=class_name, alpha=0.7, s=50)

    ax1.set_title('t-SNE Visualization of Features')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 密度图
    ax2.hexbin(features_2d[:, 0], features_2d[:, 1], gridsize=30, cmap='Blues')
    ax2.set_title('Feature Distribution Density')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to {save_path}")

    return fig


def plot_attention_weights(attention_weights, signal_length=1024,
                           save_path=None, figsize=(15, 10)):
    """
    绘制注意力权重可视化

    Args:
        attention_weights: 注意力权重字典
        signal_length: 信号长度
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        fig: matplotlib图形对象
    """
    # 确定子图数量
    n_plots = len(attention_weights)
    fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=figsize)

    if n_plots == 1:
        axes = [axes]
    elif n_plots <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    plot_idx = 0

    for name, weights in attention_weights.items():
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]

        # 处理不同类型的注意力权重
        if 'modulation' in name.lower():
            # 物理调制权重 - 条形图
            bars = ax.bar(range(len(weights[0])), weights[0], alpha=0.7)
            ax.set_title(f'{name} (Sample 0)')
            ax.set_xlabel('Feature Dimension')
            ax.set_ylabel('Modulation Weight')

            # 添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        elif 'channel' in name.lower():
            # 通道注意力 - 条形图
            bars = ax.bar(range(len(weights[0])), weights[0], alpha=0.7)
            ax.set_title(f'{name} (Sample 0)')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Attention Weight')

        elif 'spatial' in name.lower():
            # 空间注意力 - 线图
            ax.plot(weights[0], linewidth=2)
            ax.set_title(f'{name} (Sample 0)')
            ax.set_xlabel('Spatial Position')
            ax.set_ylabel('Attention Weight')
            ax.grid(True, alpha=0.3)

        else:
            # 默认处理 - 热力图
            if len(weights.shape) > 1:
                im = ax.imshow(weights[:5], cmap='viridis', aspect='auto')
                ax.set_title(f'{name} (First 5 samples)')
                ax.set_xlabel('Feature Dimension')
                ax.set_ylabel('Sample Index')
                plt.colorbar(im, ax=ax)
            else:
                ax.plot(weights, linewidth=2)
                ax.set_title(f'{name}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Weight')
                ax.grid(True, alpha=0.3)

        plot_idx += 1

    # 隐藏多余的子图
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention weights visualization saved to {save_path}")

    return fig


def plot_domain_adaptation_analysis(domain_history, save_path=None, figsize=(15, 10)):
    """
    绘制域适应分析图

    Args:
        domain_history: 域适应历史记录
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        fig: matplotlib图形对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 提取数据
    epochs = range(1, len(domain_history) + 1)
    mmd_losses = [record['mmd_loss'] for record in domain_history]
    adv_losses = [record['adversarial_loss'] for record in domain_history]
    adaptive_weights = [record['adaptive_weight'] for record in domain_history]
    domain_losses = [record['domain_loss'] for record in domain_history]

    # 子图1: MMD损失
    axes[0, 0].plot(epochs, mmd_losses, 'b-', linewidth=2, label='MMD Loss')
    axes[0, 0].set_title('MMD Loss over Training')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MMD Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 子图2: 对抗损失
    axes[0, 1].plot(epochs, adv_losses, 'r-', linewidth=2, label='Adversarial Loss')
    axes[0, 1].set_title('Adversarial Loss over Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Adversarial Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 子图3: 自适应权重
    axes[1, 0].plot(epochs, adaptive_weights, 'g-', linewidth=2, label='Adaptive Weight')
    axes[1, 0].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Equal Weight')
    axes[1, 0].set_title('Adaptive Weight over Training')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Weight (0=MMD, 1=Adversarial)')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 子图4: 总域损失
    axes[1, 1].plot(epochs, domain_losses, 'purple', linewidth=2, label='Total Domain Loss')
    axes[1, 1].set_title('Total Domain Loss over Training')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Domain Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Domain adaptation analysis saved to {save_path}")

    return fig


def save_model_architecture_diagram(model, input_shape=(1, 1, 1024), save_path=None):
    """
    保存模型架构图

    Args:
        model: PyTorch模型
        input_shape: 输入形状
        save_path: 保存路径
    """
    try:
        from torchviz import make_dot
        import torch

        # 创建输入张量
        x = torch.randn(input_shape).unsqueeze(0)
        if hasattr(model, 'device'):
            x = x.to(model.device)

        # 前向传播
        model.eval()
        with torch.no_grad():
            y = model(x)

        # 生成计算图
        dot = make_dot(y, params=dict(model.named_parameters()))

        if save_path:
            # 保存为PDF和PNG
            base_path = save_path.rsplit('.', 1)[0]
            dot.render(base_path, format='png', cleanup=True)
            print(f"Model architecture diagram saved to {base_path}.png")

        return dot

    except ImportError:
        print("torchviz not installed. Install with: pip install torchviz")
        return None
    except Exception as e:
        print(f"Failed to generate model architecture diagram: {e}")
        return None


def create_training_dashboard(train_history, val_history=None, save_path=None):
    """
    创建交互式训练仪表板

    Args:
        train_history: 训练历史
        val_history: 验证历史
        save_path: 保存路径

    Returns:
        fig: plotly图形对象
    """
    try:
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Accuracy', 'Learning Rate', 'Domain Loss'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        epochs = list(range(1, len(train_history['loss']) + 1))

        # Loss曲线
        fig.add_trace(
            go.Scatter(x=epochs, y=train_history['loss'],
                       mode='lines', name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )

        if val_history and 'loss' in val_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=val_history['loss'],
                           mode='lines', name='Val Loss', line=dict(color='red')),
                row=1, col=1
            )

        # Accuracy曲线
        if 'accuracy' in train_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=train_history['accuracy'],
                           mode='lines', name='Train Acc', line=dict(color='green')),
                row=1, col=2
            )

        if val_history and 'accuracy' in val_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=val_history['accuracy'],
                           mode='lines', name='Val Acc', line=dict(color='orange')),
                row=1, col=2
            )

        # Learning Rate
        if 'lr' in train_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=train_history['lr'],
                           mode='lines', name='Learning Rate', line=dict(color='purple')),
                row=2, col=1
            )

        # Domain Loss
        if 'domain_loss' in train_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=train_history['domain_loss'],
                           mode='lines', name='Domain Loss', line=dict(color='brown')),
                row=2, col=2
            )

        # 更新布局
        fig.update_layout(
            title="Training Dashboard",
            showlegend=True,
            height=800,
            template="plotly_white"
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive training dashboard saved to {save_path}")

        return fig

    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return None


def plot_physical_feature_importance(importance_dict, save_path=None, figsize=(12, 6)):
    """
    绘制物理特征重要性

    Args:
        importance_dict: 重要性字典
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        fig: matplotlib图形对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 域重要性饼图
    domain_names = ['Time Domain', 'Frequency Domain', 'Time-Frequency Domain']
    domain_values = [
        importance_dict['time_domain'],
        importance_dict['frequency_domain'],
        importance_dict['time_frequency_domain']
    ]

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    wedges, texts, autotexts = ax1.pie(domain_values, labels=domain_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax1.set_title('Physical Feature Domain Importance')

    # 原始数值条形图
    bars = ax2.bar(domain_names, [importance_dict['raw_values'][key]
                                  for key in ['time', 'frequency', 'time_frequency']],
                   color=colors, alpha=0.7)

    ax2.set_title('Raw Feature Importance Values')
    ax2.set_ylabel('Importance Score')
    ax2.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Physical feature importance plot saved to {save_path}")

    return fig


# 测试代码
if __name__ == '__main__':
    # 测试可视化功能
    import numpy as np

    # 创建模拟数据
    np.random.seed(42)

    # 测试训练曲线
    print("Testing training curves...")
    epochs = 50
    train_history = {
        'loss': [1.0 - 0.8 * (1 - np.exp(-i / 10)) + 0.1 * np.random.randn() for i in range(epochs)],
        'accuracy': [0.25 + 0.7 * (1 - np.exp(-i / 15)) + 0.05 * np.random.randn() for i in range(epochs)]
    }

    val_history = {
        'loss': [1.0 - 0.7 * (1 - np.exp(-i / 12)) + 0.15 * np.random.randn() for i in range(epochs)],
        'accuracy': [0.25 + 0.65 * (1 - np.exp(-i / 18)) + 0.08 * np.random.randn() for i in range(epochs)]
    }

    fig = plot_training_curves(train_history, val_history)
    plt.show()
    plt.close()

    # 测试混淆矩阵
    print("Testing confusion matrix...")
    y_true = np.random.randint(0, 4, 200)
    y_pred = np.random.randint(0, 4, 200)
    class_names = ['Normal', 'Inner_Race', 'Outer_Race', 'Rolling_Element']

    fig = plot_confusion_matrix_advanced(y_true, y_pred, class_names, normalize=True)
    plt.show()
    plt.close()

    # 测试t-SNE可视化
    print("Testing t-SNE visualization...")
    features = np.random.randn(200, 50)
    labels = np.random.randint(0, 4, 200)

    fig = plot_feature_tsne(features, labels, class_names, perplexity=15)
    plt.show()
    plt.close()

    # 测试物理特征重要性
    print("Testing physical feature importance...")
    importance_dict = {
        'time_domain': 0.4,
        'frequency_domain': 0.35,
        'time_frequency_domain': 0.25,
        'raw_values': {
            'time': 2.1,
            'frequency': 1.8,
            'time_frequency': 1.3
        }
    }

    fig = plot_physical_feature_importance(importance_dict)
    plt.show()
    plt.close()

    print("All visualization tests completed!")