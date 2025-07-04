"""
PG-MSAC-Net主程序
物理引导的多尺度自适应跨域轴承故障诊断网络
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
from typing import Dict, Any, Optional

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from config import config, update_config_from_args
from models.PG_MSAC_Net import PG_MSAC_Net
from datasets.JNU_bearing_dataset import create_jnu_dataloaders, create_single_domain_dataloader
from training.trainer import PGMSACTrainer
from evaluation.evaluator import ModelEvaluator
from utils.visualization import plot_training_curves, plot_confusion_matrix_advanced
from utils.checkpoint import CheckpointManager


class PGMSACNetError(Exception):
    """PG-MSAC-Net自定义异常"""
    pass


class DataLoadError(PGMSACNetError):
    """数据加载异常"""
    pass


class ModelError(PGMSACNetError):
    """模型相关异常"""
    pass


def set_seed(seed: int = 42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def setup_device() -> torch.device:
    """设置计算设备"""
    device = config.device
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    return device


def check_data_path() -> bool:
    """检查数据路径是否存在"""
    data_path = config.data.data_root
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        print("Please ensure your data is placed in the correct directory or update the path in config.py")
        print(f"Expected structure: {data_path}/JNU_*.mat files")
        return False

    # 检查是否有.mat文件
    mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
    if not mat_files:
        print(f"❌ No .mat files found in {data_path}")
        print("Please place your JNU bearing dataset files in the data directory")
        return False

    print(f"✅ Data path verified: {data_path}")
    print(f"Found {len(mat_files)} .mat files")
    return True


def load_datasets() -> Dict[str, DataLoader]:
    """加载数据集"""
    print("Loading datasets...")

    try:
        # 创建跨域数据加载器
        dataloaders = create_jnu_dataloaders(
            data_path=config.data.data_root,
            source_speed=config.data.source_speed,
            target_speed=config.data.target_speed,
            source_shot=config.data.source_shot,
            target_shot=config.data.target_shot,
            sample_len=config.data.sample_len,
            batch_size=config.training.batch_size,
            normalize_type=config.data.normalize_type,
            num_workers=config.num_workers,
            fault_types=config.data.class_names
        )

        print(f"✅ Source train samples: {len(dataloaders['source_train'].dataset)}")
        print(f"✅ Source test samples: {len(dataloaders['source_test'].dataset)}")
        print(f"✅ Target samples: {len(dataloaders['target'].dataset)}")

        return dataloaders

    except Exception as e:
        raise DataLoadError(f"Failed to load datasets: {str(e)}")


def create_model() -> PG_MSAC_Net:
    """创建模型"""
    print("Creating PG-MSAC-Net model...")

    try:
        model = PG_MSAC_Net(
            num_classes=config.data.num_classes,
            sample_len=config.data.sample_len,
            mpie_config=config.mpie,
            amscnn_config=config.amscnn,
            msgda_config=config.msgda
        )

        model = model.to(config.device)

        # 打印模型信息
        complexity = model.get_model_complexity()
        total_params = complexity['total_parameters']
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✅ Model created successfully")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {complexity['model_size_mb']:.2f} MB")

        return model

    except Exception as e:
        raise ModelError(f"Failed to create model: {str(e)}")


def create_trainer(model: PG_MSAC_Net, dataloaders: Dict[str, DataLoader]) -> PGMSACTrainer:
    """创建训练器"""
    print("Creating trainer...")

    try:
        trainer = PGMSACTrainer(
            model=model,
            config=config,
            device=config.device
        )

        print("✅ Trainer created successfully")
        return trainer

    except Exception as e:
        raise ModelError(f"Failed to create trainer: {str(e)}")


def run_quick_test() -> bool:
    """运行快速测试以验证系统工作"""
    print("=" * 60)
    print("Running Quick Test")
    print("=" * 60)

    try:
        # 1. 检查数据路径（但不要求数据存在）
        print("1. Checking system setup...")
        print(f"   Project root: {os.getcwd()}")
        print(f"   Data path: {config.data.data_root}")
        print(f"   Results path: {config.results_dir}")

        # 2. 创建虚拟数据集进行测试
        print("2. Creating dummy datasets for testing...")
        from datasets.data_utils import BaseDataset

        # 生成虚拟数据
        dummy_data = np.random.randn(20, 1, config.data.sample_len)
        dummy_labels = np.random.randint(0, config.data.num_classes, 20)

        dummy_dataset = BaseDataset(dummy_data, dummy_labels)
        dummy_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=False)

        dataloaders = {
            'source_train': dummy_loader,
            'source_test': dummy_loader,
            'target': dummy_loader
        }
        print("✅ Dummy datasets created")

        # 3. 测试模型创建
        print("3. Testing model creation...")
        model = create_model()
        print("✅ Model creation successful")

        # 4. 测试前向传播
        print("4. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            for data, labels in dummy_loader:
                data = data.to(config.device)
                output = model(data)
                expected_shape = (data.shape[0], config.data.num_classes)
                assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
                print(f"✅ Forward pass successful. Input: {data.shape}, Output: {output.shape}")
                break

        # 5. 测试特征提取
        print("5. Testing feature extraction...")
        features = model.extract_features(data)
        print(f"✅ Feature extraction successful. Features: {list(features.keys())}")

        # 6. 测试训练器创建
        print("6. Testing trainer creation...")
        trainer = create_trainer(model, dataloaders)
        model_info = trainer.get_model_info()
        print(f"✅ Trainer creation successful. Model: {model_info['model_name']}")

        print("\n" + "=" * 60)
        print("🎉 All quick tests passed! System is ready for use.")
        print("\nTo run with real data:")
        print("1. Place your JNU bearing dataset in:", config.data.data_root)
        print("2. Run: python main.py --experiment_type cross_domain")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Quick test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_single_experiment() -> Optional[Dict[str, Any]]:
    """运行单次实验"""
    print("=" * 60)
    print("Running Single Experiment")
    print("=" * 60)

    try:
        # 检查数据路径
        if not check_data_path():
            print("Please fix data path issues and try again.")
            return None

        # 加载数据
        dataloaders = load_datasets()

        # 创建模型
        model = create_model()

        # 创建训练器
        trainer = create_trainer(model, dataloaders)

        # 训练模型
        print("\nStarting training...")
        start_time = time.time()

        trainer.fit(
            train_loader=dataloaders['source_train'],
            val_loader=dataloaders['source_test'],
            target_loader=dataloaders['target'] if config.training.domain_adaptation else None
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # 评估模型
        print("\nEvaluating model...")
        test_results = trainer.cross_domain_evaluate(
            source_loader=dataloaders['source_test'],
            target_loader=dataloaders['target']
        )

        # 保存最终结果
        final_results = {
            'source_accuracy': test_results['source']['accuracy'],
            'target_accuracy': test_results['target']['accuracy'],
            'domain_gap': test_results['domain_gap'],
            'transfer_ratio': test_results['transfer_ratio'],
            'training_time': training_time,
            'model_params': sum(p.numel() for p in model.parameters()),
            'config': config.get_experiment_name()
        }

        print(f"\n🎉 Final Results:")
        print(f"Source Accuracy: {final_results['source_accuracy']:.4f}")
        print(f"Target Accuracy: {final_results['target_accuracy']:.4f}")
        print(f"Domain Gap: {final_results['domain_gap']:.4f}")
        print(f"Transfer Ratio: {final_results['transfer_ratio']:.4f}")

        return final_results

    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_multiple_trials(num_trials: int = 5) -> Optional[Dict[str, Any]]:
    """运行多次实验获取稳定结果"""
    print("=" * 60)
    print(f"Running Multiple Trials (n={num_trials})")
    print("=" * 60)

    if not check_data_path():
        print("Please fix data path issues and try again.")
        return None

    all_results = []

    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")

        # 设置不同的随机种子
        set_seed(config.seed + trial)

        try:
            results = run_single_experiment()
            if results is not None:
                results['trial'] = trial + 1
                all_results.append(results)
                print(f"Trial {trial + 1} - Target Accuracy: {results['target_accuracy']:.4f}")
            else:
                print(f"Trial {trial + 1} failed")

        except Exception as e:
            print(f"Trial {trial + 1} failed: {str(e)}")
            continue

    # 计算统计结果
    if all_results:
        target_accuracies = [r['target_accuracy'] for r in all_results]
        source_accuracies = [r['source_accuracy'] for r in all_results]

        stats = {
            'mean_source_accuracy': np.mean(source_accuracies),
            'std_source_accuracy': np.std(source_accuracies),
            'mean_target_accuracy': np.mean(target_accuracies),
            'std_target_accuracy': np.std(target_accuracies),
            'num_trials': len(all_results)
        }

        print("\n" + "=" * 60)
        print("🎉 FINAL RESULTS")
        print("=" * 60)
        print(f"Mean Source Accuracy: {stats['mean_source_accuracy']:.4f} ± {stats['std_source_accuracy']:.4f}")
        print(f"Mean Target Accuracy: {stats['mean_target_accuracy']:.4f} ± {stats['std_target_accuracy']:.4f}")
        print(f"Successful trials: {stats['num_trials']}/{num_trials}")

        # 保存统计结果
        results_path = os.path.join(config.results_dir, 'final_results.npz')
        np.savez(results_path, stats=stats, all_results=all_results)
        print(f"Results saved to: {results_path}")

        return stats
    else:
        print("❌ All trials failed!")
        return None


def main():
    """主函数"""
    print("🚀 PG-MSAC-Net: Physical-Guided Multi-Scale Adaptive Cross-Domain Network")
    print("for Bearing Fault Diagnosis")
    print("=" * 80)

    # 解析命令行参数
    try:
        args = update_config_from_args()
        experiment_type = args.experiment_type
    except SystemExit:
        # 如果没有命令行参数，使用默认设置
        experiment_type = 'quick_test'

    # 设置随机种子
    set_seed(config.seed)

    # 设置设备
    device = setup_device()

    # 打印配置信息
    config.print_config()

    # 创建结果保存目录
    os.makedirs(config.experiment_dir, exist_ok=True)

    try:
        if experiment_type == 'quick_test':
            # 运行快速测试
            print("\n🔧 Running Quick Test...")
            success = run_quick_test()
            if success:
                print("\n✅ System check passed! You can now run full experiments.")
                print("\nNext steps:")
                print("• Place JNU dataset in:", config.data.data_root)
                print("• Run full training: python main.py --experiment_type cross_domain")
                return 0
            else:
                print("\n❌ System check failed. Please fix the issues above.")
                return 1

        elif experiment_type == 'cross_domain':
            # 运行跨域实验
            print("\n🚀 Running Cross-Domain Experiment...")
            results = run_multiple_trials(num_trials=3)
            if results:
                print("\n✅ Cross-domain experiment completed successfully!")
            else:
                print("\n❌ Cross-domain experiment failed.")
                return 1

        elif experiment_type == 'single':
            # 运行单次实验
            print("\n🚀 Running Single Experiment...")
            results = run_single_experiment()
            if results:
                print("\n✅ Single experiment completed successfully!")
            else:
                print("\n❌ Single experiment failed.")
                return 1

        else:
            print(f"❌ Unknown experiment type: {experiment_type}")
            print("Available types: quick_test, cross_domain, single")
            return 1

        print("\n🎉 Experiment completed!")
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user.")
        return 1

    except Exception as e:
        print(f"\n❌ Experiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())