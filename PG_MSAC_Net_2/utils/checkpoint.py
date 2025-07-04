"""
模型检查点管理模块
提供模型保存、加载、版本管理等功能
"""

import os
import shutil
import json
import pickle
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import glob

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    检查点管理器

    功能：
    1. 自动保存和加载模型检查点
    2. 管理模型版本和实验记录
    3. 保留最佳模型和最近模型
    4. 提供实验文件夹管理
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5,
                 save_best: bool = True, monitor_metric: str = 'accuracy'):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保存检查点数量
            save_best: 是否保存最佳模型
            monitor_metric: 监控的指标名称
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor_metric = monitor_metric

        # 创建目录
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 最佳指标记录
        self.best_metric = float('-inf') if monitor_metric in ['accuracy', 'f1_score'] else float('inf')
        self.best_epoch = 0

        # 检查点历史
        self.checkpoint_history = []

        # 加载已有的最佳指标记录
        self._load_best_metric()

    def _load_best_metric(self):
        """加载已保存的最佳指标"""
        best_info_path = os.path.join(self.checkpoint_dir, 'best_model_info.json')
        if os.path.exists(best_info_path):
            try:
                with open(best_info_path, 'r') as f:
                    info = json.load(f)
                    self.best_metric = info.get('best_metric', self.best_metric)
                    self.best_epoch = info.get('best_epoch', 0)
                    logger.info(f"Loaded best metric: {self.best_metric} at epoch {self.best_epoch}")
            except Exception as e:
                logger.warning(f"Failed to load best metric info: {e}")

    def _save_best_metric(self):
        """保存最佳指标信息"""
        best_info = {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'monitor_metric': self.monitor_metric,
            'timestamp': datetime.now().isoformat()
        }

        best_info_path = os.path.join(self.checkpoint_dir, 'best_model_info.json')
        with open(best_info_path, 'w') as f:
            json.dump(best_info, f, indent=2)

    def _is_better_metric(self, current_metric: float) -> bool:
        """判断当前指标是否更好"""
        if self.monitor_metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc']:
            return current_metric > self.best_metric
        else:  # loss metrics
            return current_metric < self.best_metric

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, metrics: Dict[str, float],
                        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                        extra_info: Optional[Dict[str, Any]] = None) -> str:
        """
        保存检查点

        Args:
            model: PyTorch模型
            optimizer: 优化器
            epoch: 当前轮次
            metrics: 指标字典
            scheduler: 学习率调度器
            extra_info: 额外信息

        Returns:
            checkpoint_path: 保存的检查点路径
        """
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_class': model.__class__.__name__
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if extra_info is not None:
            checkpoint['extra_info'] = extra_info

        # 保存常规检查点
        checkpoint_filename = f'checkpoint_epoch_{epoch:03d}.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # 更新检查点历史
        self.checkpoint_history.append({
            'epoch': epoch,
            'path': checkpoint_path,
            'metrics': metrics.copy(),
            'timestamp': checkpoint['timestamp']
        })

        # 检查是否为最佳模型
        if self.save_best and self.monitor_metric in metrics:
            current_metric = metrics[self.monitor_metric]
            if self._is_better_metric(current_metric):
                self.best_metric = current_metric
                self.best_epoch = epoch

                # 保存最佳模型
                best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_model_path)
                logger.info(f"New best model saved: {best_model_path} "
                            f"({self.monitor_metric}: {current_metric:.4f})")

                # 保存最佳指标信息
                self._save_best_metric()

        # 保存最新模型
        latest_model_path = os.path.join(self.checkpoint_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_model_path)

        # 清理旧的检查点
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件"""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # 按epoch排序
            sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['epoch'])

            # 删除最旧的检查点
            to_delete = sorted_checkpoints[:-self.max_checkpoints]
            for checkpoint_info in to_delete:
                try:
                    if os.path.exists(checkpoint_info['path']):
                        os.remove(checkpoint_info['path'])
                        logger.info(f"Removed old checkpoint: {checkpoint_info['path']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_info['path']}: {e}")

            # 更新历史记录
            self.checkpoint_history = sorted_checkpoints[-self.max_checkpoints:]

    def load_checkpoint(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                        checkpoint_path: Optional[str] = None,
                        load_best: bool = False) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            model: PyTorch模型
            optimizer: 优化器
            scheduler: 学习率调度器
            checkpoint_path: 指定的检查点路径
            load_best: 是否加载最佳模型

        Returns:
            checkpoint_info: 检查点信息
        """
        # 确定加载路径
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_model.pth')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"Metrics: {checkpoint.get('metrics', {})}")

        return checkpoint

    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """获取所有检查点列表"""
        return self.checkpoint_history.copy()

    def get_best_model_info(self) -> Dict[str, Any]:
        """获取最佳模型信息"""
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'monitor_metric': self.monitor_metric,
            'best_model_path': os.path.join(self.checkpoint_dir, 'best_model.pth')
        }

    def export_training_log(self, save_path: Optional[str] = None) -> str:
        """
        导出训练日志

        Args:
            save_path: 保存路径

        Returns:
            log_path: 日志文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.checkpoint_dir, 'training_log.json')

        log_data = {
            'checkpoint_history': self.checkpoint_history,
            'best_model_info': self.get_best_model_info(),
            'config': {
                'max_checkpoints': self.max_checkpoints,
                'save_best': self.save_best,
                'monitor_metric': self.monitor_metric
            },
            'export_timestamp': datetime.now().isoformat()
        }

        with open(save_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Training log exported to: {save_path}")
        return save_path


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float, accuracy: float,
                    save_path: str, **kwargs) -> None:
    """
    简单的检查点保存函数

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 轮次
        loss: 损失值
        accuracy: 准确率
        save_path: 保存路径
        **kwargs: 其他信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """
    简单的检查点加载函数

    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        device: 设备

    Returns:
        checkpoint: 检查点信息
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"Loss: {checkpoint.get('loss', 'N/A')}")
    logger.info(f"Accuracy: {checkpoint.get('accuracy', 'N/A')}")

    return checkpoint


def save_best_model(model: nn.Module, save_path: str,
                    metrics: Dict[str, float], epoch: int) -> None:
    """
    保存最佳模型

    Args:
        model: 模型
        save_path: 保存路径
        metrics: 指标字典
        epoch: 轮次
    """
    best_model_info = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'model_class': model.__class__.__name__
    }

    torch.save(best_model_info, save_path)
    logger.info(f"Best model saved to {save_path}")


def load_best_model(model: nn.Module, model_path: str,
                    device: torch.device) -> Dict[str, Any]:
    """
    加载最佳模型

    Args:
        model: 模型
        model_path: 模型路径
        device: 设备

    Returns:
        model_info: 模型信息
    """
    model_info = torch.load(model_path, map_location=device)

    model.load_state_dict(model_info['model_state_dict'])

    logger.info(f"Best model loaded from {model_path}")
    logger.info(f"Metrics: {model_info.get('metrics', {})}")
    logger.info(f"Epoch: {model_info.get('epoch', 'N/A')}")

    return model_info


def create_experiment_folder(base_dir: str, experiment_name: str = None) -> str:
    """
    创建实验文件夹

    Args:
        base_dir: 基础目录
        experiment_name: 实验名称

    Returns:
        experiment_dir: 实验目录路径
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"

    experiment_dir = os.path.join(base_dir, experiment_name)

    # 创建子目录
    subdirs = ['checkpoints', 'logs', 'figures', 'configs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    # 创建实验信息文件
    exp_info = {
        'experiment_name': experiment_name,
        'created_time': datetime.now().isoformat(),
        'experiment_dir': experiment_dir
    }

    with open(os.path.join(experiment_dir, 'experiment_info.json'), 'w') as f:
        json.dump(exp_info, f, indent=2)

    logger.info(f"Experiment folder created: {experiment_dir}")
    return experiment_dir


def backup_code_files(source_dir: str, backup_dir: str,
                      extensions: List[str] = ['.py', '.yaml', '.yml']) -> None:
    """
    备份代码文件

    Args:
        source_dir: 源代码目录
        backup_dir: 备份目录
        extensions: 要备份的文件扩展名
    """
    os.makedirs(backup_dir, exist_ok=True)

    backed_up_files = []

    for root, dirs, files in os.walk(source_dir):
        # 跳过隐藏目录和__pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, source_dir)
                backup_path = os.path.join(backup_dir, relative_path)

                # 创建目标目录
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)

                # 复制文件
                shutil.copy2(source_path, backup_path)
                backed_up_files.append(relative_path)

    # 创建备份清单
    backup_manifest = {
        'backup_time': datetime.now().isoformat(),
        'source_dir': source_dir,
        'backed_up_files': backed_up_files,
        'total_files': len(backed_up_files)
    }

    with open(os.path.join(backup_dir, 'backup_manifest.json'), 'w') as f:
        json.dump(backup_manifest, f, indent=2)

    logger.info(f"Backed up {len(backed_up_files)} files to {backup_dir}")


# 测试代码
if __name__ == '__main__':
    import tempfile
    import torch.nn as nn

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Testing CheckpointManager...")

        # 创建简单模型
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # 创建检查点管理器
        manager = CheckpointManager(
            checkpoint_dir=os.path.join(temp_dir, 'checkpoints'),
            max_checkpoints=3,
            monitor_metric='accuracy'
        )

        # 模拟训练过程
        for epoch in range(5):
            metrics = {
                'loss': 1.0 / (epoch + 1),
                'accuracy': 0.5 + 0.1 * epoch
            }

            checkpoint_path = manager.save_checkpoint(
                model, optimizer, epoch, metrics
            )
            print(f"Epoch {epoch}: {metrics}")

        # 测试加载最佳模型
        best_info = manager.get_best_model_info()
        print(f"Best model info: {best_info}")

        # 加载最佳模型
        checkpoint = manager.load_checkpoint(model, optimizer, load_best=True)
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

        # 导出训练日志
        log_path = manager.export_training_log()
        print(f"Training log exported to: {log_path}")

        print("CheckpointManager test completed successfully!")