"""
模型工具函数
提供模型创建、加载、保存等实用功能
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np
from .PG_MSAC_Net import PG_MSAC_Net


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    计算模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        参数统计字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size(model: nn.Module) -> float:
    """
    计算模型大小（MB）

    Args:
        model: PyTorch模型

    Returns:
        模型大小（MB）
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                          epoch: int, loss: float, accuracy: float,
                          save_path: str, **kwargs) -> None:
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        accuracy: 当前准确率
        save_path: 保存路径
        **kwargs: 其他信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'model_config': getattr(model, 'config', None),
        **kwargs
    }

    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")


def load_model_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                          checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """
    加载模型检查点

    Args:
        model: 模型
        optimizer: 优化器（可选）
        checkpoint_path: 检查点路径
        device: 设备

    Returns:
        检查点信息字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Model checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"Accuracy: {checkpoint.get('accuracy', 'N/A')}")

    return checkpoint


def print_model_summary(model: nn.Module, input_shape: tuple = (1, 1, 1024)) -> None:
    """
    打印模型摘要

    Args:
        model: 模型
        input_shape: 输入形状
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    # 模型基本信息
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape}")

    # 参数统计
    param_stats = count_parameters(model)
    print(f"Total parameters: {param_stats['total_parameters']:,}")
    print(f"Trainable parameters: {param_stats['trainable_parameters']:,}")
    print(f"Non-trainable parameters: {param_stats['non_trainable_parameters']:,}")

    # 模型大小
    model_size = get_model_size(model)
    print(f"Model size: {model_size:.2f} MB")

    # 如果是PG_MSAC_Net，显示详细信息
    if isinstance(model, PG_MSAC_Net):
        complexity = model.get_model_complexity()
        print("\nModule-wise parameter distribution:")
        for module, percentage in complexity['parameter_distribution'].items():
            print(f"  {module}: {percentage:.1f}%")

    print("=" * 80)


def initialize_weights(model: nn.Module, init_type: str = 'xavier') -> None:
    """
    初始化模型权重

    Args:
        model: 模型
        init_type: 初始化类型 ('xavier', 'kaiming', 'normal')
    """

    def init_func(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_func)
    print(f"Model weights initialized with {init_type} initialization")


def freeze_layers(model: nn.Module, layer_names: list) -> None:
    """
    冻结指定层的参数

    Args:
        model: 模型
        layer_names: 要冻结的层名称列表
    """
    frozen_params = 0

    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                frozen_params += param.numel()
                break

    print(f"Frozen {frozen_params:,} parameters in layers: {layer_names}")


def unfreeze_layers(model: nn.Module, layer_names: list) -> None:
    """
    解冻指定层的参数

    Args:
        model: 模型
        layer_names: 要解冻的层名称列表
    """
    unfrozen_params = 0

    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                unfrozen_params += param.numel()
                break

    print(f"Unfrozen {unfrozen_params:,} parameters in layers: {layer_names}")


def get_learning_rate_groups(model: PG_MSAC_Net, base_lr: float = 1e-3) -> list:
    """
    为PG_MSAC_Net创建分层学习率组

    Args:
        model: PG_MSAC_Net模型
        base_lr: 基础学习率

    Returns:
        参数组列表
    """
    param_groups = [
        {
            'params': model.physical_encoder.parameters(),
            'lr': base_lr * 2,  # 物理编码器使用更高学习率
            'name': 'physical_encoder'
        },
        {
            'params': model.adaptive_cnn.parameters(),
            'lr': base_lr,
            'name': 'adaptive_cnn'
        },
        {
            'params': model.domain_adapter.parameters(),
            'lr': base_lr * 0.5,  # 域适应器使用较低学习率
            'name': 'domain_adapter'
        },
        {
            'params': model.classifier.parameters(),
            'lr': base_lr,
            'name': 'classifier'
        }
    ]

    return param_groups


def validate_model_config(config) -> bool:
    """
    验证模型配置的合理性

    Args:
        config: 配置对象

    Returns:
        是否有效
    """
    try:
        # 检查基本配置
        assert hasattr(config, 'data'), "Missing data config"
        assert hasattr(config, 'model'), "Missing model config"
        assert hasattr(config, 'training'), "Missing training config"

        # 检查数据配置
        assert config.data.num_classes > 0, "num_classes must be positive"
        assert config.data.sample_len > 0, "sample_len must be positive"

        # 检查模型配置
        mpie_config = config.model.MPIEConfig()
        amscnn_config = config.model.AMSCNNConfig()
        msgda_config = config.model.MSGDAConfig()

        assert mpie_config.output_dim > 0, "MPIE output_dim must be positive"
        assert len(amscnn_config.kernel_sizes) > 0, "AMSCNN kernel_sizes cannot be empty"
        assert msgda_config.num_statistical_features > 0, "MSGDA num_statistical_features must be positive"

        # 检查训练配置
        assert config.training.batch_size > 0, "batch_size must be positive"
        assert config.training.max_epochs > 0, "max_epochs must be positive"
        assert config.training.base_lr > 0, "base_lr must be positive"

        print("Model configuration validation passed")
        return True

    except Exception as e:
        print(f"Model configuration validation failed: {str(e)}")
        return False


def model_inference(model: nn.Module, signal: torch.Tensor,
                    class_names: list = None) -> Dict[str, Any]:
    """
    模型推理函数

    Args:
        model: 训练好的模型
        signal: 输入信号 [batch_size, 1, sample_len]
        class_names: 类别名称列表

    Returns:
        推理结果字典
    """
    model.eval()

    with torch.no_grad():
        # 前向传播
        logits = model(signal)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        # 置信度
        confidence, _ = torch.max(probabilities, dim=1)

        # 构建结果
        results = {
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'confidence': confidence.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }

        # 如果提供了类别名称，添加预测标签
        if class_names is not None:
            predicted_labels = [class_names[pred] for pred in results['predictions']]
            results['predicted_labels'] = predicted_labels

        return results


# 导出所有工具函数
__all__ = [
    'count_parameters',
    'get_model_size',
    'save_model_checkpoint',
    'load_model_checkpoint',
    'print_model_summary',
    'initialize_weights',
    'freeze_layers',
    'unfreeze_layers',
    'get_learning_rate_groups',
    'validate_model_config',
    'model_inference'
]