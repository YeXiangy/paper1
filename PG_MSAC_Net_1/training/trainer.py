"""
PG-MSAC-Net训练器
实现完整的训练、验证和测试流程
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# 修复导入路径问题
try:
    # 尝试相对导入
    from .loss_functions import CrossDomainLoss, get_loss_function
    from ..models.PG_MSAC_Net import PG_MSAC_Net
    from ..evaluation.metrics import calculate_metrics
    from ..utils.checkpoint import save_checkpoint, load_checkpoint
except ImportError:
    # 如果相对导入失败，使用绝对导入
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from training.loss_functions import CrossDomainLoss, get_loss_function
    from models.PG_MSAC_Net import PG_MSAC_Net
    from evaluation.metrics import calculate_metrics
    from utils.checkpoint import save_checkpoint, load_checkpoint


class PGMSACTrainer:
    """PG-MSAC-Net训练器"""

    def __init__(self,
                 model,  # 移除类型注解避免导入问题
                 config: Dict,
                 device: torch.device,
                 logger: Optional[logging.Logger] = None):
        """
        初始化训练器

        Args:
            model: PG-MSAC-Net模型
            config: 训练配置
            device: 设备
            logger: 日志器
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger or self._setup_logger()

        # 训练状态
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}

        # 设置损失函数
        self.criterion = self._setup_loss_function()

        # 设置优化器
        self.optimizer = self._setup_optimizer()

        # 设置学习率调度器
        self.scheduler = self._setup_scheduler()

        # 训练配置
        self.max_epochs = config.training.max_epochs
        self.save_dir = config.training.save_dir
        self.log_interval = getattr(config.training, 'log_interval', 50)
        self.val_interval = getattr(config.training, 'val_interval', 1)
        self.save_interval = getattr(config.training, 'save_interval', 10)

        # 早停配置
        self.early_stopping = getattr(config.training, 'early_stopping', False)
        self.patience = getattr(config.training, 'patience', 20)
        self.patience_counter = 0

        # 域适应配置
        self.domain_adaptation = getattr(config.training, 'domain_adaptation', True)
        self.alpha_schedule = getattr(config.training, 'alpha_schedule', 'constant')

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger.info("Trainer initialized successfully")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger('PGMSACTrainer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_loss_function(self) -> nn.Module:
        """设置损失函数"""
        loss_config = {
            'type': 'cross_domain',
            'num_classes': self.config.data.num_classes,
            'weights': {
                'classification': self.config.training.lambda_classification,
                'physical': self.config.training.lambda_physical,
                'domain': self.config.training.lambda_domain
            },
            'label_smoothing': 0.1
        }
        return get_loss_function(loss_config)

    def _setup_optimizer(self) -> optim.Optimizer:
        """设置优化器"""
        optimizer_type = self.config.training.optimizer

        if optimizer_type == 'adam':
            # 使用分层学习率
            param_groups = [
                {
                    'params': self.model.physical_encoder.parameters(),
                    'lr': self.config.training.physical_encoder_lr,
                    'name': 'physical_encoder'
                },
                {
                    'params': self.model.adaptive_cnn.parameters(),
                    'lr': self.config.training.base_lr,
                    'name': 'adaptive_cnn'
                },
                {
                    'params': self.model.domain_adapter.parameters(),
                    'lr': self.config.training.domain_adapter_lr,
                    'name': 'domain_adapter'
                },
                {
                    'params': self.model.classifier.parameters(),
                    'lr': self.config.training.classifier_lr,
                    'name': 'classifier'
                }
            ]

            optimizer = optim.Adam(
                param_groups,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.base_lr,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.base_lr,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        return optimizer

    def _setup_scheduler(self) -> Optional[object]:
        """设置学习率调度器"""
        scheduler_type = self.config.training.lr_scheduler

        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=self.config.training.lr_step_size,
                gamma=self.config.training.lr_gamma
            )
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None

        return scheduler

    def _get_alpha(self, epoch: int) -> float:
        """获取梯度反转参数alpha"""
        if self.alpha_schedule == 'constant':
            return 1.0
        elif self.alpha_schedule == 'linear':
            return 2.0 / (1.0 + np.exp(-10 * epoch / self.max_epochs)) - 1.0
        else:
            return 1.0

    def train_epoch(self,
                    source_loader: DataLoader,
                    target_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器（可选）

        Returns:
            epoch_metrics: epoch指标字典
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        num_samples = 0

        # 如果有目标域数据，创建迭代器
        if target_loader is not None and self.domain_adaptation:
            target_iter = iter(target_loader)
            use_domain_adaptation = True
        else:
            use_domain_adaptation = False

        # 获取alpha参数
        alpha = self._get_alpha(self.current_epoch)

        # 进度条
        pbar = tqdm(source_loader, desc=f'Epoch {self.current_epoch + 1}/{self.max_epochs}')

        for batch_idx, (source_data, source_labels) in enumerate(pbar):
            # 移动到设备
            source_data = source_data.to(self.device)
            source_labels = source_labels.to(self.device)

            # 获取目标域数据（如果使用域适应）
            if use_domain_adaptation:
                try:
                    target_data, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)

                target_data = target_data.to(self.device)

                # 确保batch size一致
                min_batch_size = min(source_data.size(0), target_data.size(0))
                source_data = source_data[:min_batch_size]
                source_labels = source_labels[:min_batch_size]
                target_data = target_data[:min_batch_size]

            # 前向传播
            self.optimizer.zero_grad()

            if use_domain_adaptation:
                # 获取目标域特征
                with torch.no_grad():
                    target_features = self.model.extract_features(target_data)['deep_features']

                # 源域前向传播（带域适应）
                source_logits, source_features, domain_loss, loss_info = self.model(
                    source_data, target_features, domain_adapt=True, alpha=alpha
                )

                # 计算总损失
                outputs = {'logits': source_logits}
                physical_code = self.model.physical_encoder(source_data)

                total_loss, loss_components = self.criterion(
                    outputs, source_labels, domain_loss,
                    physical_code, source_features
                )
            else:
                # 普通前向传播
                source_logits = self.model(source_data)
                outputs = {'logits': source_logits}

                total_loss, loss_components = self.criterion(outputs, source_labels)

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            if hasattr(self.config.training, 'grad_clip') and self.config.training.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

            self.optimizer.step()

            # 计算准确率
            _, predicted = torch.max(source_logits.data, 1)
            correct = (predicted == source_labels).sum().item()
            accuracy = correct / source_labels.size(0)

            # 累积指标
            epoch_loss += total_loss.item()
            epoch_acc += accuracy
            num_batches += 1
            num_samples += source_labels.size(0)

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{accuracy:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # 日志记录
            if batch_idx % self.log_interval == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch + 1}, Batch {batch_idx}, '
                    f'Loss: {total_loss.item():.4f}, Acc: {accuracy:.4f}'
                )

        # 计算epoch平均指标
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches

        # 更新学习率调度器
        if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step()

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'num_samples': num_samples
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            val_metrics: 验证指标字典
        """
        self.model.eval()

        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc='Validation'):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                outputs = self.model(data)

                # 计算损失
                loss, _ = self.criterion({'logits': outputs}, targets)
                val_loss += loss.item()

                # 收集预测和真实标签
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 计算详细指标
        metrics = calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = val_loss / len(val_loader)

        # 更新学习率调度器（如果是ReduceLROnPlateau）
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics['accuracy'])

        return metrics

    def fit(self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            target_loader: Optional[DataLoader] = None):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            target_loader: 目标域数据加载器（用于域适应）
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Max epochs: {self.max_epochs}")
        self.logger.info(f"Domain adaptation: {self.domain_adaptation}")

        start_time = time.time()

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader, target_loader)
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])

            self.logger.info(
                f'Epoch {epoch + 1}/{self.max_epochs} - '
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Train Acc: {train_metrics["accuracy"]:.4f}'
            )

            # 验证
            if val_loader is not None and (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate(val_loader)
                self.val_history['loss'].append(val_metrics['loss'])
                self.val_history['accuracy'].append(val_metrics['accuracy'])

                self.logger.info(
                    f'Validation - Loss: {val_metrics["loss"]:.4f}, '
                    f'Acc: {val_metrics["accuracy"]:.4f}, '
                    f'F1: {val_metrics.get("f1_score", 0):.4f}'
                )

                # 保存最佳模型
                if val_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = val_metrics['accuracy']
                    self.best_epoch = epoch
                    self.patience_counter = 0

                    # 保存最佳模型
                    best_model_path = os.path.join(self.save_dir, 'best_model.pth')
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=val_metrics['loss'],
                        accuracy=val_metrics['accuracy'],
                        save_path=best_model_path
                    )
                    self.logger.info(f"New best model saved with accuracy: {self.best_accuracy:.4f}")
                else:
                    self.patience_counter += 1

            # 定期保存检查点
            if (epoch + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=train_metrics['loss'],
                    accuracy=train_metrics['accuracy'],
                    save_path=checkpoint_path
                )

            # 早停检查
            if self.early_stopping and self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # 动态调整损失权重
            if hasattr(self.criterion, 'update_weights'):
                self.criterion.update_weights(epoch, self.max_epochs)

        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best accuracy: {self.best_accuracy:.4f} at epoch {self.best_epoch + 1}")

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        测试模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            test_metrics: 测试指标字典
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc='Testing'):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)

                # 收集结果
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 计算详细指标
        test_metrics = calculate_metrics(
            all_targets,
            all_predictions,
            all_probabilities
        )

        self.logger.info("Test Results:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {metric}: {value:.4f}")

        return test_metrics

    def cross_domain_evaluate(self,
                              source_loader: DataLoader,
                              target_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        跨域评估

        Args:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器

        Returns:
            results: 跨域评估结果
        """
        self.logger.info("Starting cross-domain evaluation...")

        # 源域评估
        source_metrics = self.test(source_loader)
        self.logger.info("Source domain results:")
        for metric, value in source_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {metric}: {value:.4f}")

        # 目标域评估
        target_metrics = self.test(target_loader)
        self.logger.info("Target domain results:")
        for metric, value in target_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {metric}: {value:.4f}")

        # 计算域差距
        domain_gap = source_metrics['accuracy'] - target_metrics['accuracy']
        transfer_ratio = target_metrics['accuracy'] / source_metrics['accuracy']

        self.logger.info(f"Domain gap: {domain_gap:.4f}")
        self.logger.info(f"Transfer ratio: {transfer_ratio:.4f}")

        return {
            'source': source_metrics,
            'target': target_metrics,
            'domain_gap': domain_gap,
            'transfer_ratio': transfer_ratio
        }

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint_info = load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path,
            device=self.device
        )

        self.current_epoch = checkpoint_info.get('epoch', 0)
        self.best_accuracy = checkpoint_info.get('accuracy', 0.0)

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resumed from epoch {self.current_epoch}")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if hasattr(self.model, 'get_model_complexity'):
            complexity = self.model.get_model_complexity()
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            complexity = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            }

        return {
            'model_name': self.model.__class__.__name__,
            'device': str(self.device),
            'current_epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            **complexity
        }


# 工厂函数
def create_trainer(model, config: Dict, device: torch.device):
    """
    创建训练器的工厂函数

    Args:
        model: PG-MSAC-Net模型
        config: 配置
        device: 设备

    Returns:
        trainer: 训练器实例
    """
    return PGMSACTrainer(model, config, device)


# 测试函数
if __name__ == '__main__':
    print("PGMSACTrainer module loaded successfully!")