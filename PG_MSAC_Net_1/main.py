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

warnings.filterwarnings('ignore')

# 导入项目模块 - 修复后的导入
from config import config, update_config_from_args
from models.PG_MSAC_Net import PG_MSAC_Net
from datasets.JNU_bearing_dataset import create_jnu_dataloaders, create_single_domain_dataloader
from training.trainer import PGMSACTrainer
from evaluation.evaluator import ModelEvaluator
from utils.visualization import plot_training_curves, plot_confusion_matrix_advanced
from utils.checkpoint import CheckpointManager


def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device():
    """设置计算设备"""
    device = config.device
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    return device


def load_datasets():
    """加载数据集"""
    print("Loading datasets...")

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

    print(f"Source train samples: {len(dataloaders['source_train'].dataset)}")
    print(f"Source test samples: {len(dataloaders['source_test'].dataset)}")
    print(f"Target samples: {len(dataloaders['target'].dataset)}")

    return dataloaders


def create_model():
    """创建模型"""
    print("Creating PG-MSAC-Net model...")

    model = PG_MSAC_Net(
        num_classes=config.data.num_classes,
        sample_len=config.data.sample_len,
        mpie_config=config.mpie,
        amscnn_config=config.amscnn,
        msgda_config=config.msgda
    )

    model = model.to(config.device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 ** 2:.2f} MB")

    return model


def create_trainer(model, dataloaders):
    """创建训练器"""
    print("Creating trainer...")

    trainer = PGMSACTrainer(
        model=model,
        config=config,
        device=config.device
    )

    return trainer


def run_single_experiment():
    """运行单次实验"""
    print("=" * 60)
    print("Running Single Experiment")
    print("=" * 60)

    # 加载数据
    dataloaders = load_datasets()

    # 创建模型
    model = create_model()

    # 创建训练器
    trainer = create_trainer(model, dataloaders)

    # 训练模型
    print("\nStarting training...")
    start_time = time.time()

    try:
        trainer.fit(
            train_loader=dataloaders['source_train'],
            val_loader=dataloaders['source_test'],
            target_loader=dataloaders['target'] if config.training.domain_adaptation else None
        )
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # 评估模型
    print("\nEvaluating model...")
    try:
        # 跨域评估
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

        print(f"\nFinal Results:")
        print(f"Source Accuracy: {final_results['source_accuracy']:.4f}")
        print(f"Target Accuracy: {final_results['target_accuracy']:.4f}")
        print(f"Domain Gap: {final_results['domain_gap']:.4f}")
        print(f"Transfer Ratio: {final_results['transfer_ratio']:.4f}")

        return final_results

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None


def run_multiple_trials(num_trials=5):
    """运行多次实验获取稳定结果"""
    print("=" * 60)
    print(f"Running Multiple Trials (n={num_trials})")
    print("=" * 60)

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
        print("FINAL RESULTS")
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
        print("All trials failed!")
        return None


def run_quick_test():
    """运行快速测试以验证系统工作"""
    print("=" * 60)
    print("Running Quick Test")
    print("=" * 60)

    try:
        # 测试数据加载
        print("Testing data loading...")
        dataloaders = load_datasets()
        print("✅ Data loading successful")

        # 测试模型创建
        print("Testing model creation...")
        model = create_model()
        print("✅ Model creation successful")

        # 测试前向传播
        print("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            # 从数据加载器获取一个批次
            for data, labels in dataloaders['source_train']:
                output = model(data)
                print(f"✅ Forward pass successful. Input: {data.shape}, Output: {output.shape}")
                break

        print("\n🎉 All tests passed! System is ready for training.")
        return True

    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("PG-MSAC-Net: Physical-Guided Multi-Scale Adaptive Cross-Domain Network")
    print("for Bearing Fault Diagnosis")
    print("=" * 80)

    # 解析命令行参数（如果有）
    try:
        args = update_config_from_args()
        experiment_type = args.experiment_type
    except:
        # 如果没有命令行参数，使用默认设置
        experiment_type = 'quick_test'  # 默认先运行快速测试

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
            print("\nRunning Quick Test...")
            success = run_quick_test()
            if success:
                print("\n✅ System check passed! You can now run full experiments.")
                print("Use --experiment_type cross_domain for full training.")
            else:
                print("\n❌ System check failed. Please fix the issues above.")

        elif experiment_type == 'cross_domain':
            # 运行跨域实验
            print("\nRunning Cross-Domain Experiment...")
            results = run_multiple_trials(num_trials=3)  # 减少试验次数用于测试

        elif experiment_type == 'single':
            # 运行单次实验
            print("\nRunning Single Experiment...")
            results = run_single_experiment()

        else:
            print(f"Unknown experiment type: {experiment_type}")
            print("Available types: quick_test, cross_domain, single")

        print("\nExperiment completed!")

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\nExperiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()