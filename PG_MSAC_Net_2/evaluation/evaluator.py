"""
评估器模块
包含单域评估、跨域评估等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    模型评估器

    功能：
    1. 单域模型评估
    2. 多类别分类指标计算
    3. 混淆矩阵和ROC曲线绘制
    """

    def __init__(self, class_names: List[str] = None, device: torch.device = None):
        """
        初始化评估器

        Args:
            class_names: 类别名称列表
            device: 计算设备
        """
        self.class_names = class_names or ['Normal', 'Inner_Race', 'Outer_Race', 'Rolling_Element']
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.class_names)

        # 存储评估结果
        self.evaluation_history = []

    def evaluate_model(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                       criterion: nn.Module = None, return_predictions: bool = False) -> Dict:
        """
        评估模型性能

        Args:
            model: 待评估模型
            dataloader: 测试数据加载器
            criterion: 损失函数
            return_predictions: 是否返回预测结果

        Returns:
            evaluation_results: 评估结果字典
        """
        model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        num_batches = 0

        logger.info("Starting model evaluation...")

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
                data, labels = data.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = model(data)

                # 计算损失
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                # 获取预测结果
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                num_batches += 1

        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # 计算评估指标
        metrics = self.compute_metrics(all_labels, all_predictions, all_probabilities)

        # 计算平均损失
        if criterion is not None:
            metrics['avg_loss'] = total_loss / num_batches

        # 构建评估结果
        evaluation_results = {
            'metrics': metrics,
            'num_samples': len(all_labels),
            'num_batches': num_batches
        }

        # 如果需要返回预测结果
        if return_predictions:
            evaluation_results.update({
                'predictions': all_predictions,
                'labels': all_labels,
                'probabilities': all_probabilities
            })

        # 保存到历史记录
        self.evaluation_history.append(evaluation_results)

        logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")

        return evaluation_results

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: np.ndarray = None) -> Dict:
        """
        计算评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率

        Returns:
            metrics: 指标字典
        """
        metrics = {}

        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 加权平均指标
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 每个类别的指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # ROC-AUC（多分类）
        if y_prob is not None and self.num_classes > 2:
            try:
                # 对于多分类问题，使用one-vs-rest策略
                y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))
                auc_scores = []

                for i in range(self.num_classes):
                    if len(np.unique(y_true_binarized[:, i])) > 1:  # 确保类别存在
                        auc = roc_auc_score(y_true_binarized[:, i], y_prob[:, i])
                        auc_scores.append(auc)
                        metrics[f'auc_{self.class_names[i]}'] = auc

                if auc_scores:
                    metrics['auc_macro'] = np.mean(auc_scores)

            except Exception as e:
                logger.warning(f"Failed to compute AUC: {str(e)}")

        # 分类报告
        try:
            report = classification_report(y_true, y_pred, target_names=self.class_names,
                                           output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        except Exception as e:
            logger.warning(f"Failed to generate classification report: {str(e)}")

        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              normalize: bool = False, save_path: str = None) -> plt.Figure:
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            normalize: 是否归一化
            save_path: 保存路径

        Returns:
            fig: matplotlib图形对象
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'

        # 创建图形
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        return plt.gcf()

    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray,
                        save_path: str = None) -> plt.Figure:
        """
        绘制ROC曲线

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            save_path: 保存路径

        Returns:
            fig: matplotlib图形对象
        """
        # 将标签二值化
        y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(12, 8))

        # 为每个类别绘制ROC曲线
        for i in range(self.num_classes):
            if len(np.unique(y_true_binarized[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
                auc = roc_auc_score(y_true_binarized[:, i], y_prob[:, i])

                plt.plot(fpr, tpr, linewidth=2,
                         label=f'{self.class_names[i]} (AUC = {auc:.3f})')

        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Multi-class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")

        return plt.gcf()

    def generate_evaluation_report(self, evaluation_results: Dict,
                                   save_path: str = None) -> str:
        """
        生成评估报告

        Args:
            evaluation_results: 评估结果
            save_path: 保存路径

        Returns:
            report: 报告字符串
        """
        metrics = evaluation_results['metrics']

        report_lines = [
            "=" * 80,
            "MODEL EVALUATION REPORT",
            "=" * 80,
            f"Number of samples: {evaluation_results['num_samples']}",
            f"Number of classes: {self.num_classes}",
            "",
            "OVERALL PERFORMANCE:",
            f"  Accuracy: {metrics['accuracy']:.4f}",
            f"  Macro F1-Score: {metrics['f1_macro']:.4f}",
            f"  Weighted F1-Score: {metrics['f1_weighted']:.4f}",
            ""
        ]

        # 添加每个类别的性能
        report_lines.append("PER-CLASS PERFORMANCE:")
        for class_name in self.class_names:
            precision = metrics.get(f'precision_{class_name}', 0)
            recall = metrics.get(f'recall_{class_name}', 0)
            f1 = metrics.get(f'f1_{class_name}', 0)

            report_lines.append(f"  {class_name}:")
            report_lines.append(f"    Precision: {precision:.4f}")
            report_lines.append(f"    Recall: {recall:.4f}")
            report_lines.append(f"    F1-Score: {f1:.4f}")

        # 添加AUC信息
        if 'auc_macro' in metrics:
            report_lines.append("")
            report_lines.append("AUC SCORES:")
            report_lines.append(f"  Macro AUC: {metrics['auc_macro']:.4f}")
            for class_name in self.class_names:
                auc_key = f'auc_{class_name}'
                if auc_key in metrics:
                    report_lines.append(f"  {class_name} AUC: {metrics[auc_key]:.4f}")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")

        return report


class CrossDomainEvaluator:
    """
    跨域评估器

    功能：
    1. 跨域性能评估
    2. 域适应效果分析
    3. 域差异可视化
    """

    def __init__(self, class_names: List[str] = None, device: torch.device = None):
        """
        初始化跨域评估器

        Args:
            class_names: 类别名称列表
            device: 计算设备
        """
        self.class_names = class_names or ['Normal', 'Inner_Race', 'Outer_Race', 'Rolling_Element']
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_evaluator = ModelEvaluator(class_names, device)

        # 存储跨域评估结果
        self.cross_domain_history = []

    def evaluate_cross_domain(self, model: nn.Module,
                              source_loader: torch.utils.data.DataLoader,
                              target_loader: torch.utils.data.DataLoader,
                              domain_names: Tuple[str, str] = ('Source', 'Target')) -> Dict:
        """
        评估跨域性能

        Args:
            model: 模型
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器
            domain_names: 域名称元组

        Returns:
            cross_domain_results: 跨域评估结果
        """
        logger.info(f"Evaluating cross-domain performance: {domain_names[0]} -> {domain_names[1]}")

        # 评估源域性能
        source_results = self.base_evaluator.evaluate_model(
            model, source_loader, return_predictions=True
        )

        # 评估目标域性能
        target_results = self.base_evaluator.evaluate_model(
            model, target_loader, return_predictions=True
        )

        # 计算域适应指标
        domain_adaptation_metrics = self._compute_domain_adaptation_metrics(
            source_results, target_results
        )

        # 构建跨域评估结果
        cross_domain_results = {
            'domain_names': domain_names,
            'source_domain': {
                'name': domain_names[0],
                'metrics': source_results['metrics'],
                'num_samples': source_results['num_samples']
            },
            'target_domain': {
                'name': domain_names[1],
                'metrics': target_results['metrics'],
                'num_samples': target_results['num_samples']
            },
            'domain_adaptation': domain_adaptation_metrics,
            'predictions': {
                'source': {
                    'labels': source_results['labels'],
                    'predictions': source_results['predictions'],
                    'probabilities': source_results['probabilities']
                },
                'target': {
                    'labels': target_results['labels'],
                    'predictions': target_results['predictions'],
                    'probabilities': target_results['probabilities']
                }
            }
        }

        # 保存到历史记录
        self.cross_domain_history.append(cross_domain_results)

        logger.info(f"Cross-domain evaluation completed.")
        logger.info(f"Source accuracy: {source_results['metrics']['accuracy']:.4f}")
        logger.info(f"Target accuracy: {target_results['metrics']['accuracy']:.4f}")
        logger.info(f"Domain gap: {domain_adaptation_metrics['accuracy_gap']:.4f}")

        return cross_domain_results

    def _compute_domain_adaptation_metrics(self, source_results: Dict,
                                           target_results: Dict) -> Dict:
        """
        计算域适应指标

        Args:
            source_results: 源域结果
            target_results: 目标域结果

        Returns:
            domain_metrics: 域适应指标
        """
        source_metrics = source_results['metrics']
        target_metrics = target_results['metrics']

        domain_metrics = {}

        # 计算性能差距
        domain_metrics['accuracy_gap'] = source_metrics['accuracy'] - target_metrics['accuracy']
        domain_metrics['f1_gap'] = source_metrics['f1_macro'] - target_metrics['f1_macro']

        # 计算迁移率
        domain_metrics['transfer_ratio'] = target_metrics['accuracy'] / source_metrics['accuracy']

        # 计算相对性能保持度
        domain_metrics['performance_retention'] = 1 - abs(domain_metrics['accuracy_gap']) / source_metrics['accuracy']

        # 每个类别的域适应效果
        for class_name in self.class_names:
            source_f1 = source_metrics.get(f'f1_{class_name}', 0)
            target_f1 = target_metrics.get(f'f1_{class_name}', 0)
            domain_metrics[f'f1_gap_{class_name}'] = source_f1 - target_f1

            if source_f1 > 0:
                domain_metrics[f'transfer_ratio_{class_name}'] = target_f1 / source_f1
            else:
                domain_metrics[f'transfer_ratio_{class_name}'] = 0

        return domain_metrics

    def plot_domain_comparison(self, cross_domain_results: Dict,
                               save_path: str = None) -> plt.Figure:
        """
        绘制域间性能对比图

        Args:
            cross_domain_results: 跨域评估结果
            save_path: 保存路径

        Returns:
            fig: matplotlib图形对象
        """
        source_metrics = cross_domain_results['source_domain']['metrics']
        target_metrics = cross_domain_results['target_domain']['metrics']
        domain_names = cross_domain_results['domain_names']

        # 准备数据
        metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        source_values = [source_metrics[metric] for metric in metrics_to_plot]
        target_values = [target_metrics[metric] for metric in metrics_to_plot]

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 子图1: 整体性能对比
        x = np.arange(len(metrics_to_plot))
        width = 0.35

        ax1.bar(x - width / 2, source_values, width, label=domain_names[0], alpha=0.8)
        ax1.bar(x + width / 2, target_values, width, label=domain_names[1], alpha=0.8)

        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Cross-Domain Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 每个类别的F1分数对比
        class_f1_source = [source_metrics.get(f'f1_{cls}', 0) for cls in self.class_names]
        class_f1_target = [target_metrics.get(f'f1_{cls}', 0) for cls in self.class_names]

        x_classes = np.arange(len(self.class_names))
        ax2.bar(x_classes - width / 2, class_f1_source, width, label=domain_names[0], alpha=0.8)
        ax2.bar(x_classes + width / 2, class_f1_target, width, label=domain_names[1], alpha=0.8)

        ax2.set_xlabel('Classes')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('Per-Class F1-Score Comparison')
        ax2.set_xticks(x_classes)
        ax2.set_xticklabels(self.class_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Domain comparison plot saved to {save_path}")

        return fig

    def generate_cross_domain_report(self, cross_domain_results: Dict,
                                     save_path: str = None) -> str:
        """
        生成跨域评估报告

        Args:
            cross_domain_results: 跨域评估结果
            save_path: 保存路径

        Returns:
            report: 报告字符串
        """
        domain_names = cross_domain_results['domain_names']
        source_metrics = cross_domain_results['source_domain']['metrics']
        target_metrics = cross_domain_results['target_domain']['metrics']
        adaptation_metrics = cross_domain_results['domain_adaptation']

        report_lines = [
            "=" * 80,
            "CROSS-DOMAIN EVALUATION REPORT",
            "=" * 80,
            f"Source Domain: {domain_names[0]}",
            f"Target Domain: {domain_names[1]}",
            "",
            "DOMAIN ADAPTATION SUMMARY:",
            f"  Source Accuracy: {source_metrics['accuracy']:.4f}",
            f"  Target Accuracy: {target_metrics['accuracy']:.4f}",
            f"  Accuracy Gap: {adaptation_metrics['accuracy_gap']:.4f}",
            f"  Transfer Ratio: {adaptation_metrics['transfer_ratio']:.4f}",
            f"  Performance Retention: {adaptation_metrics['performance_retention']:.4f}",
            ""
        ]

        # 添加详细的域间对比
        report_lines.append("DETAILED PERFORMANCE COMPARISON:")
        metrics_comparison = [
            ('Accuracy', 'accuracy'),
            ('Precision (Macro)', 'precision_macro'),
            ('Recall (Macro)', 'recall_macro'),
            ('F1-Score (Macro)', 'f1_macro')
        ]

        for metric_name, metric_key in metrics_comparison:
            source_val = source_metrics[metric_key]
            target_val = target_metrics[metric_key]
            gap = source_val - target_val

            report_lines.append(f"  {metric_name}:")
            report_lines.append(f"    {domain_names[0]}: {source_val:.4f}")
            report_lines.append(f"    {domain_names[1]}: {target_val:.4f}")
            report_lines.append(f"    Gap: {gap:.4f}")
            report_lines.append("")

        # 添加每个类别的适应效果
        report_lines.append("PER-CLASS ADAPTATION ANALYSIS:")
        for class_name in self.class_names:
            source_f1 = source_metrics.get(f'f1_{class_name}', 0)
            target_f1 = target_metrics.get(f'f1_{class_name}', 0)
            transfer_ratio = adaptation_metrics.get(f'transfer_ratio_{class_name}', 0)

            report_lines.append(f"  {class_name}:")
            report_lines.append(f"    Source F1: {source_f1:.4f}")
            report_lines.append(f"    Target F1: {target_f1:.4f}")
            report_lines.append(f"    Transfer Ratio: {transfer_ratio:.4f}")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Cross-domain evaluation report saved to {save_path}")

        return report


# 便捷函数
def evaluate_single_domain(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                           class_names: List[str] = None, device: torch.device = None) -> Dict:
    """
    评估单域性能的便捷函数

    Args:
        model: 模型
        dataloader: 数据加载器
        class_names: 类别名称
        device: 设备

    Returns:
        evaluation_results: 评估结果
    """
    evaluator = ModelEvaluator(class_names, device)
    return evaluator.evaluate_model(model, dataloader, return_predictions=True)


def evaluate_cross_domain(model: nn.Module,
                          source_loader: torch.utils.data.DataLoader,
                          target_loader: torch.utils.data.DataLoader,
                          domain_names: Tuple[str, str] = ('Source', 'Target'),
                          class_names: List[str] = None,
                          device: torch.device = None) -> Dict:
    """
    评估跨域性能的便捷函数

    Args:
        model: 模型
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
        domain_names: 域名称
        class_names: 类别名称
        device: 设备

    Returns:
        cross_domain_results: 跨域评估结果
    """
    evaluator = CrossDomainEvaluator(class_names, device)
    return evaluator.evaluate_cross_domain(model, source_loader, target_loader, domain_names)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None, class_names: List[str] = None) -> Dict:
    """
    计算评估指标的便捷函数

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率
        class_names: 类别名称

    Returns:
        metrics: 指标字典
    """
    evaluator = ModelEvaluator(class_names)
    return evaluator.compute_metrics(y_true, y_pred, y_prob)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str] = None, normalize: bool = False,
                          save_path: str = None) -> plt.Figure:
    """
    绘制混淆矩阵的便捷函数

    Args:
        y_true: 真实标签
        y_pred: 预测
    y_pred: 预测标签
            class_names: 类别名称
            normalize: 是否归一化
            save_path: 保存路径

        Returns:
            fig: matplotlib图形对象
        """
    evaluator = ModelEvaluator(class_names)
    return evaluator.plot_confusion_matrix(y_true, y_pred, normalize, save_path)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                    class_names: List[str] = None, save_path: str = None) -> plt.Figure:
    """
    绘制ROC曲线的便捷函数

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        class_names: 类别名称
        save_path: 保存路径

    Returns:
        fig: matplotlib图形对象
    """
    evaluator = ModelEvaluator(class_names)
    return evaluator.plot_roc_curves(y_true, y_prob, save_path)


class MetricsTracker:
    """
    指标跟踪器 - 用于训练过程中的指标监控
    """

    def __init__(self, metrics_to_track: List[str] = None):
        """
        初始化指标跟踪器

        Args:
            metrics_to_track: 要跟踪的指标列表
        """
        self.metrics_to_track = metrics_to_track or ['accuracy', 'loss', 'f1_macro']
        self.history = defaultdict(list)
        self.best_scores = {}
        self.best_epochs = {}

    def update(self, epoch: int, metrics: Dict):
        """
        更新指标

        Args:
            epoch: 当前轮次
            metrics: 指标字典
        """
        for metric_name in self.metrics_to_track:
            if metric_name in metrics:
                value = metrics[metric_name]
                self.history[metric_name].append(value)

                # 更新最佳分数
                if metric_name not in self.best_scores or value > self.best_scores[metric_name]:
                    self.best_scores[metric_name] = value
                    self.best_epochs[metric_name] = epoch

    def get_best_metrics(self) -> Dict:
        """获取最佳指标"""
        return {
            'best_scores': self.best_scores.copy(),
            'best_epochs': self.best_epochs.copy()
        }

    def plot_metrics(self, save_path: str = None) -> plt.Figure:
        """
        绘制指标变化曲线

        Args:
            save_path: 保存路径

        Returns:
            fig: matplotlib图形对象
        """
        n_metrics = len(self.metrics_to_track)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

        if n_metrics == 1:
            axes = [axes]

        for i, metric_name in enumerate(self.metrics_to_track):
            if metric_name in self.history:
                epochs = range(1, len(self.history[metric_name]) + 1)
                axes[i].plot(epochs, self.history[metric_name], 'b-', linewidth=2)
                axes[i].set_title(f'{metric_name.title()} over Epochs')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric_name.title())
                axes[i].grid(True, alpha=0.3)

                # 标记最佳点
                if metric_name in self.best_epochs:
                    best_epoch = self.best_epochs[metric_name]
                    best_score = self.best_scores[metric_name]
                    axes[i].scatter(best_epoch + 1, best_score, color='red', s=100, zorder=5)
                    axes[i].annotate(f'Best: {best_score:.4f}',
                                     xy=(best_epoch + 1, best_score),
                                     xytext=(10, 10), textcoords='offset points',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics plot saved to {save_path}")

        return fig


class EvaluationPipeline:
    """
    评估流水线 - 统一管理各种评估任务
    """

    def __init__(self, class_names: List[str] = None, device: torch.device = None,
                 output_dir: str = './results'):
        """
        初始化评估流水线

        Args:
            class_names: 类别名称
            device: 设备
            output_dir: 输出目录
        """
        self.class_names = class_names or ['Normal', 'Inner_Race', 'Outer_Race', 'Rolling_Element']
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir

        # 创建输出目录
        import os
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)

        # 初始化评估器
        self.single_evaluator = ModelEvaluator(class_names, device)
        self.cross_evaluator = CrossDomainEvaluator(class_names, device)

        logger.info(f"Evaluation pipeline initialized. Output directory: {output_dir}")

    def run_comprehensive_evaluation(self, model: nn.Module,
                                     test_dataloaders: Dict[str, torch.utils.data.DataLoader],
                                     cross_domain_pairs: List[Tuple[str, str]] = None) -> Dict:
        """
        运行综合评估

        Args:
            model: 模型
            test_dataloaders: 测试数据加载器字典 {'domain_name': dataloader}
            cross_domain_pairs: 跨域对列表 [('source_domain', 'target_domain')]

        Returns:
            comprehensive_results: 综合评估结果
        """
        logger.info("Starting comprehensive evaluation...")

        results = {
            'single_domain': {},
            'cross_domain': {},
            'summary': {}
        }

        # 1. 单域评估
        logger.info("Performing single-domain evaluations...")
        for domain_name, dataloader in test_dataloaders.items():
            logger.info(f"Evaluating on {domain_name} domain...")

            domain_results = self.single_evaluator.evaluate_model(
                model, dataloader, return_predictions=True
            )

            results['single_domain'][domain_name] = domain_results

            # 保存混淆矩阵
            cm_path = os.path.join(self.output_dir, 'figures', f'confusion_matrix_{domain_name}.png')
            self.single_evaluator.plot_confusion_matrix(
                domain_results['labels'], domain_results['predictions'],
                normalize=True, save_path=cm_path
            )

            # 保存ROC曲线
            roc_path = os.path.join(self.output_dir, 'figures', f'roc_curves_{domain_name}.png')
            self.single_evaluator.plot_roc_curves(
                domain_results['labels'], domain_results['probabilities'],
                save_path=roc_path
            )

            # 保存评估报告
            report_path = os.path.join(self.output_dir, 'reports', f'evaluation_report_{domain_name}.txt')
            self.single_evaluator.generate_evaluation_report(domain_results, report_path)

        # 2. 跨域评估
        if cross_domain_pairs:
            logger.info("Performing cross-domain evaluations...")
            for source_domain, target_domain in cross_domain_pairs:
                if source_domain in test_dataloaders and target_domain in test_dataloaders:
                    logger.info(f"Cross-domain evaluation: {source_domain} -> {target_domain}")

                    cross_results = self.cross_evaluator.evaluate_cross_domain(
                        model,
                        test_dataloaders[source_domain],
                        test_dataloaders[target_domain],
                        domain_names=(source_domain, target_domain)
                    )

                    pair_name = f"{source_domain}_to_{target_domain}"
                    results['cross_domain'][pair_name] = cross_results

                    # 保存域间对比图
                    comparison_path = os.path.join(self.output_dir, 'figures', f'domain_comparison_{pair_name}.png')
                    self.cross_evaluator.plot_domain_comparison(cross_results, comparison_path)

                    # 保存跨域评估报告
                    cross_report_path = os.path.join(self.output_dir, 'reports', f'cross_domain_report_{pair_name}.txt')
                    self.cross_evaluator.generate_cross_domain_report(cross_results, cross_report_path)

        # 3. 生成综合摘要
        results['summary'] = self._generate_summary(results)

        # 保存综合报告
        summary_path = os.path.join(self.output_dir, 'reports', 'comprehensive_summary.txt')
        self._save_comprehensive_report(results, summary_path)

        logger.info("Comprehensive evaluation completed!")

        return results

    def _generate_summary(self, results: Dict) -> Dict:
        """
        生成评估摘要

        Args:
            results: 评估结果

        Returns:
            summary: 摘要字典
        """
        summary = {
            'single_domain_performance': {},
            'cross_domain_performance': {},
            'overall_statistics': {}
        }

        # 单域性能摘要
        single_accuracies = []
        for domain_name, domain_results in results['single_domain'].items():
            accuracy = domain_results['metrics']['accuracy']
            f1_macro = domain_results['metrics']['f1_macro']

            summary['single_domain_performance'][domain_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro
            }
            single_accuracies.append(accuracy)

        # 跨域性能摘要
        transfer_ratios = []
        domain_gaps = []
        for pair_name, cross_results in results['cross_domain'].items():
            transfer_ratio = cross_results['domain_adaptation']['transfer_ratio']
            domain_gap = cross_results['domain_adaptation']['accuracy_gap']

            summary['cross_domain_performance'][pair_name] = {
                'transfer_ratio': transfer_ratio,
                'domain_gap': domain_gap
            }
            transfer_ratios.append(transfer_ratio)
            domain_gaps.append(abs(domain_gap))

        # 整体统计
        if single_accuracies:
            summary['overall_statistics']['mean_single_domain_accuracy'] = np.mean(single_accuracies)
            summary['overall_statistics']['std_single_domain_accuracy'] = np.std(single_accuracies)

        if transfer_ratios:
            summary['overall_statistics']['mean_transfer_ratio'] = np.mean(transfer_ratios)
            summary['overall_statistics']['mean_domain_gap'] = np.mean(domain_gaps)

        return summary

    def _save_comprehensive_report(self, results: Dict, save_path: str):
        """
        保存综合报告

        Args:
            results: 评估结果
            save_path: 保存路径
        """
        summary = results['summary']

        report_lines = [
            "=" * 100,
            "COMPREHENSIVE EVALUATION REPORT",
            "=" * 100,
            f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of Classes: {len(self.class_names)}",
            f"Class Names: {', '.join(self.class_names)}",
            "",
            "SINGLE-DOMAIN PERFORMANCE SUMMARY:",
            "-" * 50
        ]

        # 单域性能摘要
        for domain_name, performance in summary['single_domain_performance'].items():
            report_lines.extend([
                f"  {domain_name}:",
                f"    Accuracy: {performance['accuracy']:.4f}",
                f"    F1-Score (Macro): {performance['f1_macro']:.4f}",
                ""
            ])

        # 跨域性能摘要
        if summary['cross_domain_performance']:
            report_lines.extend([
                "CROSS-DOMAIN PERFORMANCE SUMMARY:",
                "-" * 50
            ])

            for pair_name, performance in summary['cross_domain_performance'].items():
                report_lines.extend([
                    f"  {pair_name.replace('_to_', ' -> ')}:",
                    f"    Transfer Ratio: {performance['transfer_ratio']:.4f}",
                    f"    Domain Gap: {performance['domain_gap']:.4f}",
                    ""
                ])

        # 整体统计
        if summary['overall_statistics']:
            report_lines.extend([
                "OVERALL STATISTICS:",
                "-" * 50
            ])

            stats = summary['overall_statistics']
            if 'mean_single_domain_accuracy' in stats:
                report_lines.extend([
                    f"  Mean Single-Domain Accuracy: {stats['mean_single_domain_accuracy']:.4f} ± {stats['std_single_domain_accuracy']:.4f}",
                ])

            if 'mean_transfer_ratio' in stats:
                report_lines.extend([
                    f"  Mean Transfer Ratio: {stats['mean_transfer_ratio']:.4f}",
                    f"  Mean Domain Gap: {stats['mean_domain_gap']:.4f}",
                ])

        report_lines.append("=" * 100)

        report = "\n".join(report_lines)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Comprehensive report saved to {save_path}")


# 测试代码
if __name__ == '__main__':
    # 测试评估器
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # 创建测试数据
    torch.manual_seed(42)
    device = torch.device('cpu')

    # 模拟数据
    batch_size = 32
    num_samples = 200
    num_classes = 4

    # 创建模拟预测结果
    y_true = torch.randint(0, num_classes, (num_samples,)).numpy()
    y_pred = torch.randint(0, num_classes, (num_samples,)).numpy()
    y_prob = torch.softmax(torch.randn(num_samples, num_classes), dim=1).numpy()

    # 测试单域评估
    print("Testing single-domain evaluation...")
    evaluator = ModelEvaluator()

    # 测试指标计算
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")

    # 测试混淆矩阵绘制
    try:
        fig = evaluator.plot_confusion_matrix(y_true, y_pred, normalize=True)
        plt.show()
        plt.close()
        print("Confusion matrix plot test passed!")
    except Exception as e:
        print(f"Confusion matrix plot test failed: {e}")

    # 测试ROC曲线绘制
    try:
        fig = evaluator.plot_roc_curves(y_true, y_prob)
        plt.show()
        plt.close()
        print("ROC curves plot test passed!")
    except Exception as e:
        print(f"ROC curves plot test failed: {e}")

    # 测试跨域评估
    print("\nTesting cross-domain evaluation...")
    cross_evaluator = CrossDomainEvaluator()

    # 模拟源域和目标域结果
    source_results = {
        'metrics': metrics,
        'num_samples': num_samples // 2,
        'labels': y_true[:num_samples // 2],
        'predictions': y_pred[:num_samples // 2],
        'probabilities': y_prob[:num_samples // 2]
    }

    target_results = {
        'metrics': metrics,
        'num_samples': num_samples // 2,
        'labels': y_true[num_samples // 2:],
        'predictions': y_pred[num_samples // 2:],
        'probabilities': y_prob[num_samples // 2:]
    }

    # 测试域适应指标计算
    domain_metrics = cross_evaluator._compute_domain_adaptation_metrics(source_results, target_results)
    print(f"Transfer ratio: {domain_metrics['transfer_ratio']:.4f}")
    print(f"Domain gap: {domain_metrics['accuracy_gap']:.4f}")

    # 测试指标跟踪器
    print("\nTesting metrics tracker...")
    tracker = MetricsTracker(['accuracy', 'f1_macro'])

    # 模拟训练过程
    for epoch in range(10):
        fake_metrics = {
            'accuracy': 0.7 + 0.02 * epoch + 0.01 * np.random.randn(),
            'f1_macro': 0.65 + 0.025 * epoch + 0.01 * np.random.randn()
        }
        tracker.update(epoch, fake_metrics)

    best_metrics = tracker.get_best_metrics()
    print(
        f"Best accuracy: {best_metrics['best_scores']['accuracy']:.4f} at epoch {best_metrics['best_epochs']['accuracy']}")

    # 测试指标绘制
    try:
        fig = tracker.plot_metrics()
        plt.show()
        plt.close()
        print("Metrics tracking plot test passed!")
    except Exception as e:
        print(f"Metrics tracking plot test failed: {e}")

    print("\nAll evaluation tests completed!")