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

# 导入项目模块
from config import config, update_config_from_args
from models.PG_MSAC_Net import PG_MSAC_Net
from datasets.JNU_bearing import JNU_bearing_dataset
from training.trainer import CrossDomainTrainer
from evaluation.evaluator import ModelEvaluator
from utils.visualization import ResultVisualizer
from utils.checkpoint import ModelCheckpoint


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
    if torch.cuda.is_available():
        device = config.device
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    return device


def load_datasets():
    """加载数据集"""
    print("Loading datasets...")

    # 加载JNU轴承数据集
    dataset_loader = JNU_bearing_dataset(
        source_speed=config.data.source_speed,
        target_speed=config.data.target_speed,
        source_shot=config.data.source_shot,
        target_shot=config.data.target_shot,
        sample_len=config.data.sample_len,
        normalize_type=config.data.normalize_type
    )

    # 获取数据集
    datasets = dataset_loader.prepare_cross_domain_data()

    # 创建数据加载器
    dataloaders = {}

    # 源域数据加载器
    dataloaders['source_train'] = DataLoader(
        datasets['source_train'],
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )

    # 目标域数据加载器
    dataloaders['target_train'] = DataLoader(
        datasets['target_unlabeled'],
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )

    # 测试数据加载器
    dataloaders['target_test'] = DataLoader(
        datasets['target_test'],
        batch_size=config.training.batch_size * 2,  # 测试时可以用更大batch
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # 验证数据加载器（少量目标域标注数据）
    dataloaders['target_val'] = DataLoader(
        datasets['target_labeled'],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    print(f"Source train samples: {len(datasets['source_train'])}")
    print(f"Target unlabeled samples: {len(datasets['target_unlabeled'])}")
    print(f"Target labeled samples: {len(datasets['target_labeled'])}")
    print(f"Target test samples: {len(datasets['target_test'])}")

    return dataloaders


def create_model():
    """创建模型"""
    print("Creating PG-MSAC-Net model...")

    model = PG_MSAC_Net(
        num_classes=config.data.num_classes,
        sample_len=config.data.sample_len,
        mpie_config=config.model.MPIEConfig(),
        amscnn_config=config.model.AMSCNNConfig(),
        msgda_config=config.model.MSGDAConfig()
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

    trainer = CrossDomainTrainer(
        model=model,
        dataloaders=dataloaders,
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

    training_history = trainer.train()

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # 评估模型
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(model, dataloaders, config)

    # 在目标域测试集上评估
    test_results = evaluator.evaluate_cross_domain()

    # 可视化结果
    print("\nGenerating visualizations...")
    visualizer = ResultVisualizer(config)

    # 绘制训练曲线
    visualizer.plot_training_curves(
        training_history,
        save_path=config.paths.get_figure_path('training_curves')
    )

    # 绘制混淆矩阵
    visualizer.plot_confusion_matrix(
        test_results['predictions'],
        test_results['labels'],
        config.data.class_names,
        save_path=config.paths.get_figure_path('confusion_matrix')
    )

    # 绘制t-SNE聚类图
    visualizer.plot_tsne(
        test_results['features'],
        test_results['labels'],
        config.data.class_names,
        save_path=config.paths.get_figure_path('tsne_plot')
    )

    # 保存最终结果
    final_results = {
        'test_accuracy': test_results['accuracy'],
        'test_f1_score': test_results['f1_score'],
        'training_time': training_time,
        'model_params': sum(p.numel() for p in model.parameters()),
        'config': config.get_experiment_name()
    }

    return final_results


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
            results['trial'] = trial + 1
            all_results.append(results)

            print(f"Trial {trial + 1} - Accuracy: {results['test_accuracy']:.4f}")

        except Exception as e:
            print(f"Trial {trial + 1} failed: {str(e)}")
            continue

    # 计算统计结果
    if all_results:
        accuracies = [r['test_accuracy'] for r in all_results]
        f1_scores = [r['test_f1_score'] for r in all_results]

        stats = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1_score': np.mean(f1_scores),
            'std_f1_score': np.std(f1_scores),
            'num_trials': len(all_results)
        }

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"Mean F1-Score: {stats['mean_f1_score']:.4f} ± {stats['std_f1_score']:.4f}")
        print(f"Successful trials: {stats['num_trials']}/{num_trials}")

        # 保存统计结果
        results_path = config.paths.get_log_path('final_results')
        np.save(results_path.replace('.log', '.npy'), {'stats': stats, 'all_results': all_results})

        return stats
    else:
        print("All trials failed!")
        return None


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
        experiment_type = 'cross_domain'

    # 设置随机种子
    set_seed(config.seed)

    # 设置设备
    device = setup_device()

    # 打印配置信息
    config.print_config()

    # 创建结果保存目录
    os.makedirs(config.paths.experiment_dir, exist_ok=True)

    try:
        if experiment_type == 'cross_domain':
            # 运行跨域实验
            print("\nRunning Cross-Domain Experiment...")
            results = run_multiple_trials(num_trials=config.experiment.num_trials)

        elif experiment_type == 'ablation':
            # 运行消融实验
            print("\nRunning Ablation Study...")
            from experiments.ablation_study import run_ablation_study
            results = run_ablation_study(config)

        elif experiment_type == 'hyperparameter':
            # 运行超参数搜索
            print("\nRunning Hyperparameter Search...")
            from experiments.hyperparameter_search import run_hyperparameter_search
            results = run_hyperparameter_search(config)

        else:
            # 运行单次实验
            print("\nRunning Single Experiment...")
            results = run_single_experiment()

        print("\nExperiment completed successfully!")

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