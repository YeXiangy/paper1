"""
evaluation/metrics.py - 评估指标计算模块
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Union, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray] = None,
                      class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）
        class_names: 类别名称（可选）

    Returns:
        metrics: 指标字典
    """
    metrics = {}

    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 加权指标
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 每个类别的指标（如果提供了类别名称）
    if class_names is not None:
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]

    # AUC（如果有概率）
    if y_prob is not None:
        try:
            num_classes = len(np.unique(y_true))
            if num_classes == 2:
                # 二分类
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # 多分类
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")
            metrics['auc'] = 0.0

    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics


def compute_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    计算每个类别的详细指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    Returns:
        class_metrics: 每个类别的指标字典
    """
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    class_metrics = {}
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            class_metrics[class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1_score': f1_per_class[i]
            }

    return class_metrics


def compute_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    """
    计算混淆矩阵相关指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        metrics: 混淆矩阵指标字典
    """
    cm = confusion_matrix(y_true, y_pred)

    # 总体准确率
    accuracy = np.trace(cm) / np.sum(cm)

    # 每类准确率
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)

    # 每类精确率和召回率
    precision_per_class = np.diag(cm) / np.sum(cm, axis=0)
    recall_per_class = np.diag(cm) / np.sum(cm, axis=1)

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class
    }


def compute_roc_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                        class_names: Optional[List[str]] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    计算ROC相关指标

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        class_names: 类别名称（可选）

    Returns:
        roc_metrics: ROC指标字典
    """
    num_classes = len(np.unique(y_true))
    roc_metrics = {}

    if num_classes == 2:
        # 二分类ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])

        roc_metrics['fpr'] = fpr
        roc_metrics['tpr'] = tpr
        roc_metrics['thresholds'] = thresholds
        roc_metrics['auc'] = auc

    else:
        # 多分类ROC (one-vs-rest)
        y_true_binarized = label_binarize(y_true, classes=range(num_classes))

        fpr_dict = {}
        tpr_dict = {}
        auc_dict = {}

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true_binarized[:, i], y_prob[:, i])

            class_name = class_names[i] if class_names and i < len(class_names) else f'Class_{i}'
            fpr_dict[class_name] = fpr
            tpr_dict[class_name] = tpr
            auc_dict[class_name] = auc

        roc_metrics['fpr_dict'] = fpr_dict
        roc_metrics['tpr_dict'] = tpr_dict
        roc_metrics['auc_dict'] = auc_dict
        roc_metrics['auc_macro'] = np.mean(list(auc_dict.values()))

    return roc_metrics


def compute_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: Optional[List[str]] = None) -> Dict:
    """
    生成分类报告

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称（可选）

    Returns:
        report: 分类报告字典
    """
    try:
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        return report
    except Exception as e:
        print(f"Warning: Could not generate classification report: {e}")
        return {}


def compute_domain_adaptation_metrics(source_metrics: Dict[str, float],
                                      target_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    计算域适应相关指标

    Args:
        source_metrics: 源域指标
        target_metrics: 目标域指标

    Returns:
        domain_metrics: 域适应指标
    """
    domain_metrics = {}

    # 性能差距
    domain_metrics['accuracy_gap'] = source_metrics['accuracy'] - target_metrics['accuracy']
    domain_metrics['f1_gap'] = source_metrics['f1_score'] - target_metrics['f1_score']

    # 迁移率
    domain_metrics['accuracy_transfer_ratio'] = target_metrics['accuracy'] / source_metrics['accuracy']
    domain_metrics['f1_transfer_ratio'] = target_metrics['f1_score'] / source_metrics['f1_score']

    # 性能保持度
    domain_metrics['accuracy_retention'] = 1 - abs(domain_metrics['accuracy_gap']) / source_metrics['accuracy']
    domain_metrics['f1_retention'] = 1 - abs(domain_metrics['f1_gap']) / source_metrics['f1_score']

    return domain_metrics


def evaluate_model_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None,
                               class_names: Optional[List[str]] = None) -> Dict:
    """
    综合评估模型预测结果

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）
        class_names: 类别名称（可选）

    Returns:
        evaluation_results: 综合评估结果
    """
    results = {}

    # 基础指标
    results['basic_metrics'] = calculate_metrics(y_true, y_pred, y_prob, class_names)

    # 每类别详细指标
    if class_names:
        results['class_metrics'] = compute_class_metrics(y_true, y_pred, class_names)

    # 混淆矩阵指标
    results['confusion_metrics'] = compute_confusion_matrix_metrics(y_true, y_pred)

    # ROC指标（如果有概率）
    if y_prob is not None:
        results['roc_metrics'] = compute_roc_metrics(y_true, y_prob, class_names)

    # 分类报告
    results['classification_report'] = compute_classification_report(y_true, y_pred, class_names)

    return results


def compare_model_performance(results1: Dict, results2: Dict,
                              model1_name: str = "Model 1",
                              model2_name: str = "Model 2") -> Dict:
    """
    比较两个模型的性能

    Args:
        results1: 模型1的评估结果
        results2: 模型2的评估结果
        model1_name: 模型1名称
        model2_name: 模型2名称

    Returns:
        comparison: 比较结果
    """
    comparison = {
        'models': [model1_name, model2_name],
        'metrics_comparison': {}
    }

    # 比较基础指标
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if metric in results1['basic_metrics'] and metric in results2['basic_metrics']:
            val1 = results1['basic_metrics'][metric]
            val2 = results2['basic_metrics'][metric]

            comparison['metrics_comparison'][metric] = {
                model1_name: val1,
                model2_name: val2,
                'difference': val2 - val1,
                'improvement': ((val2 - val1) / val1 * 100) if val1 > 0 else 0
            }

    return comparison


def compute_confidence_intervals(predictions: List[np.ndarray],
                                 targets: List[np.ndarray],
                                 confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    计算多次实验结果的置信区间

    Args:
        predictions: 多次实验的预测结果列表
        targets: 多次实验的真实标签列表
        confidence_level: 置信水平

    Returns:
        confidence_intervals: 各指标的置信区间
    """
    from scipy import stats

    metrics_list = []

    # 计算每次实验的指标
    for pred, target in zip(predictions, targets):
        metrics = calculate_metrics(target, pred)
        metrics_list.append(metrics)

    confidence_intervals = {}
    alpha = 1 - confidence_level

    # 计算每个指标的置信区间
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        values = [m[metric] for m in metrics_list if metric in m]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)

            # t分布置信区间
            t_value = stats.t.ppf(1 - alpha / 2, n - 1)
            margin_error = t_value * std_val / np.sqrt(n)

            confidence_intervals[metric] = (
                mean_val - margin_error,
                mean_val + margin_error
            )

    return confidence_intervals


def compute_statistical_significance(results1: List[float],
                                     results2: List[float],
                                     test_type: str = 'ttest') -> Dict[str, float]:
    """
    计算两组结果的统计显著性

    Args:
        results1: 第一组结果
        results2: 第二组结果
        test_type: 统计检验类型 ('ttest', 'wilcoxon')

    Returns:
        significance: 统计显著性结果
    """
    from scipy import stats

    significance = {}

    if test_type == 'ttest':
        # 配对t检验
        statistic, p_value = stats.ttest_rel(results1, results2)
        significance['test'] = 'Paired t-test'
    elif test_type == 'wilcoxon':
        # Wilcoxon符号秩检验
        statistic, p_value = stats.wilcoxon(results1, results2)
        significance['test'] = 'Wilcoxon signed-rank test'
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    significance['statistic'] = statistic
    significance['p_value'] = p_value
    significance['significant'] = p_value < 0.05

    return significance


def format_metrics_table(metrics: Dict[str, float],
                         class_names: Optional[List[str]] = None) -> str:
    """
    格式化指标为表格字符串

    Args:
        metrics: 指标字典
        class_names: 类别名称（可选）

    Returns:
        table: 格式化的表格字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION METRICS")
    lines.append("=" * 60)

    # 总体指标
    lines.append("Overall Metrics:")
    lines.append("-" * 30)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if metric in metrics:
            lines.append(f"  {metric.capitalize():12}: {metrics[metric]:.4f}")

    # 加权指标
    lines.append("\nWeighted Metrics:")
    lines.append("-" * 30)
    for metric in ['precision_weighted', 'recall_weighted', 'f1_weighted']:
        if metric in metrics:
            name = metric.replace('_weighted', ' (weighted)')
            lines.append(f"  {name.capitalize():18}: {metrics[metric]:.4f}")

    # 每类别指标
    if class_names:
        lines.append("\nPer-Class Metrics:")
        lines.append("-" * 30)
        for class_name in class_names:
            lines.append(f"  {class_name}:")
            for metric in ['precision', 'recall', 'f1']:
                key = f"{metric}_{class_name}"
                if key in metrics:
                    lines.append(f"    {metric.capitalize():9}: {metrics[key]:.4f}")

    lines.append("=" * 60)

    return "\n".join(lines)


# 便捷函数
def quick_evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   print_results: bool = True) -> Dict[str, float]:
    """
    快速评估函数

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称（可选）
        print_results: 是否打印结果

    Returns:
        metrics: 评估指标
    """
    metrics = calculate_metrics(y_true, y_pred, class_names=class_names)

    if print_results:
        print(format_metrics_table(metrics, class_names))

    return metrics


# 测试函数
if __name__ == '__main__':
    # 测试评估指标计算
    print("Testing evaluation metrics...")

    # 生成测试数据
    np.random.seed(42)
    n_samples = 200
    n_classes = 4

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # 归一化为概率

    class_names = ['Normal', 'Inner', 'Ball', 'Outer']

    # 测试基础指标计算
    metrics = calculate_metrics(y_true, y_pred, y_prob, class_names)
    print("✅ Basic metrics calculation successful")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    # 测试综合评估
    results = evaluate_model_predictions(y_true, y_pred, y_prob, class_names)
    print("✅ Comprehensive evaluation successful")

    # 测试快速评估
    quick_metrics = quick_evaluate(y_true, y_pred, class_names, print_results=True)
    print("✅ Quick evaluation successful")

    print("\n🎉 All evaluation tests completed!")