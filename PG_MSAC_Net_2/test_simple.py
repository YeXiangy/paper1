#!/usr/bin/env python3
"""
PG-MSAC-Net 简单测试脚本
验证所有模块的基本功能，无需真实数据集
"""

import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import sys
import os
import traceback
from typing import Dict, Any, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    """打印测试标题"""
    print("\n" + "=" * 60)
    print(f"🔍 {title}")
    print("=" * 60)


def print_success(message: str):
    """打印成功消息"""
    print(f"✅ {message}")


def print_error(message: str):
    """打印错误消息"""
    print(f"❌ {message}")


def print_info(message: str):
    """打印信息消息"""
    print(f"ℹ️  {message}")


class TestResults:
    """测试结果统计"""

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = []

    def add_test_result(self, test_name: str, passed: bool, error_msg: str = None):
        """添加测试结果"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print_success(f"{test_name} passed")
        else:
            self.failed_tests.append((test_name, error_msg))
            print_error(f"{test_name} failed: {error_msg}")

    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {len(self.failed_tests)}")

        if self.failed_tests:
            print("\n❌ Failed tests:")
            for test_name, error in self.failed_tests:
                print(f"  • {test_name}: {error}")

        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")

        if len(self.failed_tests) == 0:
            print("\n🎉 All tests passed! PG-MSAC-Net is ready to use.")
            return True
        else:
            print("\n⚠️  Some tests failed. Please check the issues above.")
            return False


def test_imports(results: TestResults):
    """测试所有关键模块的导入"""
    print_header("Testing Module Imports")

    # 测试配置模块
    try:
        from config import config
        results.add_test_result("Config import", True)
        print_info(f"Project: {config.project_name}")
        print_info(f"Device: {config.device}")
    except Exception as e:
        results.add_test_result("Config import", False, str(e))

    # 测试模型模块
    try:
        from models.PG_MSAC_Net import PG_MSAC_Net
        from models.MPIE import MultiLevelPhysicalEncoder
        from models.AMSCNN import AdaptiveMultiScaleCNN
        from models.MSGDA import MultiStatGuidedDomainAdapter
        results.add_test_result("Model modules import", True)
    except Exception as e:
        results.add_test_result("Model modules import", False, str(e))

    # 测试数据集模块
    try:
        from datasets.JNU_bearing_dataset import JNUBearingDataset
        from datasets.data_utils import DataProcessor, BaseDataset
        results.add_test_result("Dataset modules import", True)
    except Exception as e:
        results.add_test_result("Dataset modules import", False, str(e))

    # 测试训练模块
    try:
        from training.trainer import PGMSACTrainer
        from training.loss_functions import CrossDomainLoss
        results.add_test_result("Training modules import", True)
    except Exception as e:
        results.add_test_result("Training modules import", False, str(e))

    # 测试评估模块
    try:
        from evaluation.evaluator import ModelEvaluator
        from evaluation.metrics import calculate_metrics
        results.add_test_result("Evaluation modules import", True)
    except Exception as e:
        results.add_test_result("Evaluation modules import", False, str(e))

    # 测试工具模块
    try:
        from utils.checkpoint import CheckpointManager
        from utils.visualization import plot_training_curves
        results.add_test_result("Utility modules import", True)
    except Exception as e:
        results.add_test_result("Utility modules import", False, str(e))


def create_test_configs():
    """创建测试配置"""
    mpie_config = SimpleNamespace(
        time_features_dim=8, freq_features_dim=7, tf_features_dim=5,
        time_hidden_dim=48, freq_hidden_dim=40, tf_hidden_dim=40,
        output_dim=128
    )

    amscnn_config = SimpleNamespace(
        kernel_sizes=[3, 7, 15], scale_channels=64,
        conv_channels=[64, 128, 256, 512],
        physical_modulation_dim=192, dropout_rate=0.2
    )

    msgda_config = SimpleNamespace(
        num_statistical_features=6, weight_net_hidden=[32, 16],
        discriminator_hidden=[256, 128], discriminator_dropout=0.2,
        mmd_sigma=1.0
    )

    return mpie_config, amscnn_config, msgda_config


def test_individual_models(results: TestResults):
    """测试各个模型组件"""
    print_header("Testing Individual Model Components")

    device = torch.device('cpu')  # 使用CPU避免GPU依赖
    mpie_config, amscnn_config, msgda_config = create_test_configs()

    # 测试MPIE
    try:
        from models.MPIE import MultiLevelPhysicalEncoder

        mpie = MultiLevelPhysicalEncoder(mpie_config).to(device)
        test_input = torch.randn(4, 1, 1024)

        physical_code = mpie(test_input)
        expected_shape = (4, mpie_config.output_dim)

        assert physical_code.shape == expected_shape, f"Expected {expected_shape}, got {physical_code.shape}"
        results.add_test_result("MPIE forward pass", True)
        print_info(f"MPIE output shape: {physical_code.shape}")

        # 测试特征重要性
        importance = mpie.get_feature_importance(test_input)
        assert 'time_domain' in importance, "Missing time domain importance"
        results.add_test_result("MPIE feature importance", True)

    except Exception as e:
        results.add_test_result("MPIE forward pass", False, str(e))

    # 测试AMSCNN
    try:
        from models.AMSCNN import AdaptiveMultiScaleCNN

        amscnn = AdaptiveMultiScaleCNN(amscnn_config, physical_dim=128).to(device)
        test_signal = torch.randn(4, 1, 1024)
        test_physical = torch.randn(4, 128)

        deep_features = amscnn(test_signal, test_physical)
        expected_shape = (4, amscnn_config.conv_channels[-1])

        assert deep_features.shape == expected_shape, f"Expected {expected_shape}, got {deep_features.shape}"
        results.add_test_result("AMSCNN forward pass", True)
        print_info(f"AMSCNN output shape: {deep_features.shape}")

        # 测试注意力权重
        attention = amscnn.get_attention_weights(test_signal[:2], test_physical[:2])
        assert 'modulation_weights' in attention, "Missing modulation weights"
        results.add_test_result("AMSCNN attention weights", True)

    except Exception as e:
        results.add_test_result("AMSCNN forward pass", False, str(e))

    # 测试MSGDA
    try:
        from models.MSGDA import MultiStatGuidedDomainAdapter

        msgda = MultiStatGuidedDomainAdapter(msgda_config, feature_dim=512).to(device)
        source_features = torch.randn(4, 512)
        target_features = torch.randn(4, 512)

        domain_loss, adaptive_weight, loss_components = msgda(source_features, target_features)

        assert isinstance(domain_loss, torch.Tensor), "Domain loss should be a tensor"
        assert isinstance(adaptive_weight, torch.Tensor), "Adaptive weight should be a tensor"
        assert isinstance(loss_components, dict), "Loss components should be a dict"

        results.add_test_result("MSGDA forward pass", True)
        print_info(f"Domain loss: {domain_loss.item():.4f}")
        print_info(f"Adaptive weight: {adaptive_weight.item():.4f}")

    except Exception as e:
        results.add_test_result("MSGDA forward pass", False, str(e))


def test_complete_model(results: TestResults):
    """测试完整的PG-MSAC-Net模型"""
    print_header("Testing Complete PG-MSAC-Net Model")

    device = torch.device('cpu')
    mpie_config, amscnn_config, msgda_config = create_test_configs()

    try:
        from models.PG_MSAC_Net import PG_MSAC_Net

        # 创建模型
        model = PG_MSAC_Net(
            num_classes=4, sample_len=1024,
            mpie_config=mpie_config, amscnn_config=amscnn_config,
            msgda_config=msgda_config
        ).to(device)

        results.add_test_result("Model creation", True)

        # 测试普通前向传播
        test_input = torch.randn(4, 1, 1024)
        output = model(test_input)
        expected_shape = (4, 4)

        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        results.add_test_result("Model forward pass", True)
        print_info(f"Model output shape: {output.shape}")

        # 测试特征提取
        features = model.extract_features(test_input)
        expected_features = ['physical_code', 'deep_features', 'classifier_features', 'raw_signal']

        for feature_name in expected_features:
            assert feature_name in features, f"Missing feature: {feature_name}"

        results.add_test_result("Feature extraction", True)
        print_info(f"Extracted features: {list(features.keys())}")

        # 测试模型复杂度
        complexity = model.get_model_complexity()
        assert 'total_parameters' in complexity, "Missing parameter count"

        results.add_test_result("Model complexity analysis", True)
        print_info(f"Total parameters: {complexity['total_parameters']:,}")
        print_info(f"Model size: {complexity['model_size_mb']:.2f} MB")

        # 测试域适应模式
        target_features = model.extract_features(test_input)['deep_features']
        domain_output = model(test_input, target_features, domain_adapt=True)

        assert len(domain_output) == 4, "Domain adapt mode should return 4 outputs"
        results.add_test_result("Domain adaptation mode", True)

    except Exception as e:
        results.add_test_result("Complete model test", False, str(e))


def test_dataset_creation(results: TestResults):
    """测试数据集创建（使用虚拟数据）"""
    print_header("Testing Dataset Creation")

    try:
        from datasets.data_utils import BaseDataset
        from torch.utils.data import DataLoader

        # 创建虚拟数据集
        num_samples = 50
        sample_len = 1024
        num_classes = 4

        dummy_data = np.random.randn(num_samples, 1, sample_len).astype(np.float32)
        dummy_labels = np.random.randint(0, num_classes, num_samples)

        dataset = BaseDataset(dummy_data, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        results.add_test_result("Dataset creation", True)
        print_info(f"Dataset size: {len(dataset)}")

        # 测试数据加载
        for batch_data, batch_labels in dataloader:
            assert batch_data.shape[1:] == (1, sample_len), f"Wrong data shape: {batch_data.shape}"
            assert batch_labels.shape[0] == batch_data.shape[0], "Batch size mismatch"
            break

        results.add_test_result("Data loading", True)
        print_info(f"Batch shape: {batch_data.shape}")

    except Exception as e:
        results.add_test_result("Dataset creation", False, str(e))


def test_training_components(results: TestResults):
    """测试训练组件"""
    print_header("Testing Training Components")

    try:
        from training.loss_functions import CrossDomainLoss
        from training.trainer import PGMSACTrainer
        from config import config

        # 测试损失函数
        loss_fn = CrossDomainLoss(num_classes=4)

        # 创建虚拟数据
        logits = torch.randn(4, 4)
        targets = torch.randint(0, 4, (4,))
        outputs = {'logits': logits}

        total_loss, loss_components = loss_fn(outputs, targets)

        assert isinstance(total_loss, torch.Tensor), "Loss should be a tensor"
        assert isinstance(loss_components, dict), "Loss components should be a dict"

        results.add_test_result("Loss function", True)
        print_info(f"Total loss: {total_loss.item():.4f}")

        # 测试训练器创建（不进行实际训练）
        mpie_config, amscnn_config, msgda_config = create_test_configs()
        from models.PG_MSAC_Net import PG_MSAC_Net

        model = PG_MSAC_Net(
            num_classes=4, sample_len=1024,
            mpie_config=mpie_config, amscnn_config=amscnn_config,
            msgda_config=msgda_config
        )

        device = torch.device('cpu')
        trainer = PGMSACTrainer(model, config, device)

        results.add_test_result("Trainer creation", True)

        # 测试模型信息
        model_info = trainer.get_model_info()
        assert 'total_parameters' in model_info, "Missing model info"

        results.add_test_result("Model info extraction", True)
        print_info(f"Model: {model_info['model_name']}")
        print_info(f"Parameters: {model_info['total_parameters']:,}")

    except Exception as e:
        results.add_test_result("Training components", False, str(e))


def test_evaluation_components(results: TestResults):
    """测试评估组件"""
    print_header("Testing Evaluation Components")

    try:
        from evaluation.metrics import calculate_metrics

        # 创建虚拟预测结果
        y_true = np.random.randint(0, 4, 100)
        y_pred = np.random.randint(0, 4, 100)
        y_prob = np.random.rand(100, 4)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # 归一化

        class_names = ['Normal', 'Inner', 'Ball', 'Outer']

        # 计算指标
        metrics = calculate_metrics(y_true, y_pred, y_prob, class_names)

        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        results.add_test_result("Metrics calculation", True)
        print_info(f"Accuracy: {metrics['accuracy']:.4f}")
        print_info(f"F1-Score: {metrics['f1_score']:.4f}")

    except Exception as e:
        results.add_test_result("Evaluation components", False, str(e))


def test_utility_components(results: TestResults):
    """测试工具组件"""
    print_header("Testing Utility Components")

    try:
        from utils.checkpoint import CheckpointManager
        import tempfile

        # 测试检查点管理器
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                checkpoint_dir=temp_dir,
                max_checkpoints=3,
                monitor_metric='accuracy'
            )

            results.add_test_result("CheckpointManager creation", True)

            # 测试最佳模型信息
            best_info = manager.get_best_model_info()
            assert 'best_metric' in best_info, "Missing best metric info"

            results.add_test_result("Checkpoint info", True)
            print_info(f"Checkpoint dir: {temp_dir}")

    except Exception as e:
        results.add_test_result("Utility components", False, str(e))


def test_configuration(results: TestResults):
    """测试配置系统"""
    print_header("Testing Configuration System")

    try:
        from config import config

        # 测试配置访问
        assert hasattr(config, 'data'), "Missing data config"
        assert hasattr(config, 'training'), "Missing training config"
        assert hasattr(config, 'mpie'), "Missing MPIE config"
        assert hasattr(config, 'amscnn'), "Missing AMSCNN config"
        assert hasattr(config, 'msgda'), "Missing MSGDA config"

        results.add_test_result("Config structure", True)

        # 测试配置更新
        original_batch_size = config.training.batch_size
        config.update_training_params(batch_size=32)
        assert config.training.batch_size == 32, "Config update failed"

        # 恢复原值
        config.training.batch_size = original_batch_size

        results.add_test_result("Config updates", True)

        # 测试实验名称生成
        exp_name = config.get_experiment_name()
        assert isinstance(exp_name, str) and len(exp_name) > 0, "Invalid experiment name"

        results.add_test_result("Experiment naming", True)
        print_info(f"Experiment name: {exp_name}")

    except Exception as e:
        results.add_test_result("Configuration system", False, str(e))


def test_memory_usage(results: TestResults):
    """测试内存使用情况"""
    print_header("Testing Memory Usage")

    try:
        import psutil
        import gc

        # 记录初始内存
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建模型并进行前向传播
        mpie_config, amscnn_config, msgda_config = create_test_configs()
        from models.PG_MSAC_Net import PG_MSAC_Net

        model = PG_MSAC_Net(
            num_classes=4, sample_len=1024,
            mpie_config=mpie_config, amscnn_config=amscnn_config,
            msgda_config=msgda_config
        )

        # 多次前向传播测试内存泄漏
        for _ in range(10):
            test_input = torch.randn(8, 1, 1024)
            with torch.no_grad():
                output = model(test_input)
            del test_input, output

        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        results.add_test_result("Memory usage test", True)
        print_info(f"Initial memory: {initial_memory:.1f} MB")
        print_info(f"Final memory: {final_memory:.1f} MB")
        print_info(f"Memory increase: {memory_increase:.1f} MB")

        if memory_increase > 500:  # 警告如果内存增长超过500MB
            print("⚠️  High memory usage detected")

    except ImportError:
        results.add_test_result("Memory usage test", False, "psutil not available")
    except Exception as e:
        results.add_test_result("Memory usage test", False, str(e))


def main():
    """主测试函数"""
    print("🚀 PG-MSAC-Net Comprehensive Test Suite")
    print("Testing all components without requiring real data...")

    results = TestResults()

    # 运行所有测试
    test_functions = [
        test_imports,
        test_configuration,
        test_individual_models,
        test_complete_model,
        test_dataset_creation,
        test_training_components,
        test_evaluation_components,
        test_utility_components,
        test_memory_usage
    ]

    for test_func in test_functions:
        try:
            test_func(results)
        except Exception as e:
            results.add_test_result(f"{test_func.__name__}", False, f"Test crashed: {str(e)}")
            print_error(f"Test {test_func.__name__} crashed:")
            traceback.print_exc()

    # 打印总结
    success = results.print_summary()

    if success:
        print("\n🎯 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Prepare JNU dataset in: ./data/JNU/")
        print("3. Run training: python main.py --experiment_type cross_domain")
        print("\n📚 For help:")
        print("• Quick test: python main.py --experiment_type quick_test")
        print("• Single run: python main.py --experiment_type single")
        return 0
    else:
        print("\n🔧 Troubleshooting:")
        print("• Check Python version (>=3.8 recommended)")
        print("• Install missing dependencies")
        print("• Check file permissions")
        return 1


if __name__ == '__main__':
    exit(main())