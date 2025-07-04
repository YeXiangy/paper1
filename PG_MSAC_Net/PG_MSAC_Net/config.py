"""
Configuration file for PG-MSAC-Net
物理引导的多尺度自适应跨域轴承故障诊断网络配置文件
"""

import torch
import os


class Config:
    """配置类，包含所有训练和模型参数"""

    # ============================
    # 基础设置
    # ============================
    def __init__(self):
        # 项目基础设置
        self.project_name = "PG_MSAC_Net"
        self.model_name = "PG_MSAC_Net"

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 8
        self.pin_memory = True if self.device.type == 'cuda' else False

        # 随机种子
        self.seed = 42

    # ============================
    # 数据集配置
    # ============================
    class DataConfig:
        # 数据集基础设置
        dataset_name = 'JNU_bearing_dataset'
        data_root = 'D:\\00.资料\\03.Datasets\\01.bearing fault diagnosis\\JNU_bearing'

        # 样本设置
        sample_len = 1024  # 从2048减少到1024，适配更复杂的模型
        num_classes = 4  # JNU数据集：Normal, Inner, Ball, Outer
        in_channels = 1

        # 数据预处理
        normalize_type = 'mean~std'

        # 跨域实验设置
        source_speed = 800  # 源域转速
        target_speed = 1000  # 目标域转速
        source_shot = 50  # 源域每类样本数（充足）
        target_shot = 5  # 目标域每类样本数（小样本）
        target_unlabeled = 200  # 目标域无标注样本数

        # 故障类别名称
        class_names = ['Normal', 'Inner', 'Ball', 'Outer']

    # ============================
    # 模型配置
    # ============================
    class ModelConfig:
        # MPIE配置 - 多级物理信息编码器
        class MPIEConfig:
            # 物理特征维度
            time_features_dim = 8  # 时域特征维度
            freq_features_dim = 7  # 频域特征维度
            tf_features_dim = 5  # 时频域特征维度
            total_physical_dim = 20  # 总物理特征维度

            # 编码器网络结构
            time_hidden_dim = 48
            freq_hidden_dim = 40
            tf_hidden_dim = 40
            output_dim = 128  # 物理编码输出维度

            # 激活函数
            activation = 'relu'

        # AMSCNN配置 - 自适应多尺度CNN
        class AMSCNNConfig:
            # 多尺度卷积核
            kernel_sizes = [3, 7, 15]

            # 网络结构
            scale_channels = 64  # 每个尺度的通道数
            conv_channels = [64, 128, 256, 512]  # 卷积层通道数

            # 池化和正则化
            pool_size = 2
            dropout_rate = 0.2

            # 物理调制
            physical_modulation_dim = 192  # 64*3个尺度

        # MSGDA配置 - 多统计量引导域适应器
        class MSGDAConfig:
            # 统计特征数量
            num_statistical_features = 6

            # 权重网络结构
            weight_net_hidden = [32, 16]

            # 域判别器结构
            discriminator_hidden = [256, 128]
            discriminator_dropout = 0.2

            # MMD核参数
            mmd_sigma = 1.0

    # ============================
    # 训练配置
    # ============================
    class TrainingConfig:
        # 基础训练参数
        batch_size = 64
        max_epochs = 200

        # 学习率设置（分层学习率）
        base_lr = 1e-3
        physical_encoder_lr = 2e-3  # 物理编码器学习率更高
        domain_adapter_lr = 5e-4  # 域适应器学习率较低
        classifier_lr = 1e-3

        # 学习率调度
        lr_scheduler = 'cosine'  # 'step', 'cosine', 'exponential'
        lr_step_size = 50
        lr_gamma = 0.5

        # 优化器设置
        optimizer = 'adam'
        weight_decay = 1e-4
        momentum = 0.9  # 仅用于SGD

        # 损失函数权重
        lambda_classification = 1.0  # 分类损失权重
        lambda_physical = 0.1  # 物理一致性损失权重
        lambda_domain = 0.01  # 域适应损失权重

        # 早停设置
        early_stopping = True
        patience = 20
        min_delta = 1e-4

        # 梯度裁剪
        grad_clip = True
        max_grad_norm = 1.0

    # ============================
    # 实验配置
    # ============================
    class ExperimentConfig:
        # 实验类型
        experiment_types = ['ablation', 'cross_domain', 'hyperparameter']

        # 消融实验配置
        ablation_configs = [
            {'name': 'Full_Model', 'use_mpie': True, 'use_amscnn': True, 'use_msgda': True},
            {'name': 'w/o_MPIE', 'use_mpie': False, 'use_amscnn': True, 'use_msgda': True},
            {'name': 'w/o_AMSCNN', 'use_mpie': True, 'use_amscnn': False, 'use_msgda': True},
            {'name': 'w/o_MSGDA', 'use_mpie': True, 'use_amscnn': True, 'use_msgda': False},
            {'name': 'Only_CNN', 'use_mpie': False, 'use_amscnn': False, 'use_msgda': False},
        ]

        # 跨域实验配置
        cross_domain_configs = [
            {'name': 'Same_Device_Cross_Speed', 'source': 'JNU_800', 'target': 'JNU_1000'},
            {'name': 'Same_Device_Cross_Speed_2', 'source': 'JNU_800', 'target': 'JNU_600'},
            # 可以后续添加跨设备实验
            # {'name': 'Cross_Device', 'source': 'JNU_800', 'target': 'SEU_800'},
        ]

        # 重复实验次数
        num_trials = 5

        # 超参数搜索范围
        hyperparameter_ranges = {
            'learning_rate': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
            'batch_size': [32, 64, 128],
            'lambda_physical': [0.01, 0.05, 0.1, 0.2],
            'lambda_domain': [0.001, 0.005, 0.01, 0.05]
        }

    # ============================
    # 评估配置
    # ============================
    class EvaluationConfig:
        # 评估指标
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']

        # 可视化设置
        save_confusion_matrix = True
        save_tsne_plot = True
        save_loss_curves = True
        save_feature_visualization = True

        # t-SNE参数
        tsne_perplexity = 30
        tsne_n_iter = 1000

        # 混淆矩阵设置
        normalize_confusion_matrix = True

    # ============================
    # 文件路径配置
    # ============================
    class PathConfig:
        def __init__(self, config):
            # 基础路径
            self.base_dir = os.getcwd()

            # 结果保存路径
            self.results_dir = os.path.join(self.base_dir, 'results')
            self.models_dir = os.path.join(self.results_dir, 'models')
            self.logs_dir = os.path.join(self.results_dir, 'logs')
            self.figures_dir = os.path.join(self.results_dir, 'figures')

            # 实验特定路径
            experiment_name = f"{config.DataConfig.dataset_name}_{config.DataConfig.source_speed}to{config.DataConfig.target_speed}"
            self.experiment_dir = os.path.join(self.results_dir, experiment_name)

            # 创建必要的文件夹
            self._create_directories()

        def _create_directories(self):
            """创建必要的文件夹"""
            directories = [
                self.results_dir, self.models_dir, self.logs_dir,
                self.figures_dir, self.experiment_dir
            ]
            for directory in directories:
                os.makedirs(directory, exist_ok=True)

        def get_model_save_path(self, model_name, epoch=None):
            """获取模型保存路径"""
            if epoch is not None:
                filename = f"{model_name}_epoch_{epoch}.pth"
            else:
                filename = f"{model_name}_best.pth"
            return os.path.join(self.models_dir, filename)

        def get_log_path(self, log_name):
            """获取日志保存路径"""
            return os.path.join(self.logs_dir, f"{log_name}.log")

        def get_figure_path(self, figure_name):
            """获取图表保存路径"""
            return os.path.join(self.figures_dir, f"{figure_name}.png")

    # ============================
    # 初始化所有配置
    # ============================
    def __init__(self):
        super().__init__()
        self.data = self.DataConfig()
        self.model = self.ModelConfig()
        self.training = self.TrainingConfig()
        self.experiment = self.ExperimentConfig()
        self.evaluation = self.EvaluationConfig()
        self.paths = self.PathConfig(self)

    def get_experiment_name(self):
        """获取实验名称"""
        return f"{self.model_name}_{self.data.dataset_name}_{self.data.source_speed}to{self.data.target_speed}_shot{self.data.target_shot}"

    def print_config(self):
        """打印配置信息"""
        print("=" * 50)
        print(f"PG-MSAC-Net Configuration")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.data.dataset_name}")
        print(f"Cross-domain: {self.data.source_speed}rpm → {self.data.target_speed}rpm")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.training.batch_size}")
        print(f"Max epochs: {self.training.max_epochs}")
        print(f"Learning rate: {self.training.base_lr}")
        print(f"Sample length: {self.data.sample_len}")
        print(f"Target shot: {self.data.target_shot}")
        print("=" * 50)


# 全局配置实例
config = Config()


# 命令行参数解析（可选）
def update_config_from_args():
    """从命令行参数更新配置"""
    import argparse

    parser = argparse.ArgumentParser(description='PG-MSAC-Net Training')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default=config.data.dataset_name,
                        help='Dataset name')
    parser.add_argument('--source_speed', type=int, default=config.data.source_speed,
                        help='Source domain speed')
    parser.add_argument('--target_speed', type=int, default=config.data.target_speed,
                        help='Target domain speed')
    parser.add_argument('--shot', type=int, default=config.data.target_shot,
                        help='Number of samples per class in target domain')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=config.training.batch_size,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.training.max_epochs,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=config.training.base_lr,
                        help='Learning rate')

    # 损失权重
    parser.add_argument('--lambda_physical', type=float, default=config.training.lambda_physical,
                        help='Physical consistency loss weight')
    parser.add_argument('--lambda_domain', type=float, default=config.training.lambda_domain,
                        help='Domain adaptation loss weight')

    # 实验设置
    parser.add_argument('--experiment_type', type=str, default='cross_domain',
                        choices=['cross_domain', 'ablation', 'hyperparameter'],
                        help='Type of experiment to run')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')

    args = parser.parse_args()

    # 更新配置
    config.data.dataset_name = args.dataset
    config.data.source_speed = args.source_speed
    config.data.target_speed = args.target_speed
    config.data.target_shot = args.shot
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.epochs
    config.training.base_lr = args.lr
    config.training.lambda_physical = args.lambda_physical
    config.training.lambda_domain = args.lambda_domain

    # 设置GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        config.device = torch.device(f'cuda:{args.gpu}')

    return args


if __name__ == '__main__':
    # 测试配置
    config.print_config()
    print(f"Experiment name: {config.get_experiment_name()}")
    print(f"Model save path: {config.paths.get_model_save_path('test_model')}")