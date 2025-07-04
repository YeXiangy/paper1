"""
PG-MSAC-Net: Physical-Guided Multi-Scale Adaptive Cross-Domain Network
物理引导的多尺度自适应跨域轴承故障诊断网络

主模型文件 - 集成三个创新模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .MPIE import MultiLevelPhysicalEncoder
from .AMSCNN import AdaptiveMultiScaleCNN
from .MSGDA import MultiStatGuidedDomainAdapter


class PG_MSAC_Net(nn.Module):
    """
    PG-MSAC-Net主模型

    架构：
    输入信号 → MPIE → AMSCNN → MSGDA → 分类器

    创新点：
    1. MPIE: 多级物理信息编码器
    2. AMSCNN: 自适应多尺度CNN
    3. MSGDA: 多统计量引导域适应器
    """

    def __init__(self, num_classes, sample_len, mpie_config, amscnn_config, msgda_config):
        super(PG_MSAC_Net, self).__init__()

        self.num_classes = num_classes
        self.sample_len = sample_len

        # 创新点1: 多级物理信息编码器
        self.physical_encoder = MultiLevelPhysicalEncoder(mpie_config)

        # 创新点2: 自适应多尺度CNN
        self.adaptive_cnn = AdaptiveMultiScaleCNN(
            amscnn_config,
            physical_dim=mpie_config.output_dim
        )

        # 创新点3: 多统计量引导域适应器
        self.domain_adapter = MultiStatGuidedDomainAdapter(
            msgda_config,
            feature_dim=amscnn_config.conv_channels[-1]
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(amscnn_config.conv_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # 特征维度
        self.feature_dim = amscnn_config.conv_channels[-1]

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化分类器权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, signal, target_features=None, domain_adapt=False, alpha=1.0):
        """
        前向传播

        Args:
            signal: 输入振动信号 [batch_size, 1, sample_len]
            target_features: 目标域特征（域适应时使用）
            domain_adapt: 是否进行域适应
            alpha: 梯度反转参数

        Returns:
            如果domain_adapt=False:
                logits: 分类结果 [batch_size, num_classes]
            如果domain_adapt=True:
                logits: 分类结果
                features: 提取的特征
                domain_loss: 域适应损失
                loss_info: 损失信息字典
        """
        # 1. 物理特征编码
        physical_code = self.physical_encoder(signal)  # [B, 128]

        # 2. 自适应多尺度CNN特征提取
        deep_features = self.adaptive_cnn(signal, physical_code)  # [B, 512]

        # 3. 分类
        logits = self.classifier(deep_features)  # [B, num_classes]

        if not domain_adapt:
            # 普通前向传播，不进行域适应
            return logits
        else:
            # 域适应模式
            if target_features is None:
                raise ValueError("target_features is required when domain_adapt=True")

            # 4. 域适应
            domain_loss, adaptive_weight, loss_components = self.domain_adapter(
                deep_features, target_features, alpha
            )

            # 构建损失信息
            loss_info = {
                'adaptive_weight': adaptive_weight,
                'loss_components': loss_components,
                'physical_code_norm': torch.norm(physical_code, dim=1).mean().item(),
                'feature_norm': torch.norm(deep_features, dim=1).mean().item()
            }

            return logits, deep_features, domain_loss, loss_info

    def extract_features(self, signal):
        """
        提取特征（用于可视化和分析）

        Args:
            signal: 输入信号 [batch_size, 1, sample_len]

        Returns:
            feature_dict: 特征字典
        """
        with torch.no_grad():
            # 物理特征编码
            physical_code = self.physical_encoder(signal)

            # CNN特征提取
            deep_features = self.adaptive_cnn(signal, physical_code)

            # 分类特征
            classifier_input = deep_features
            for i, layer in enumerate(self.classifier[:-1]):  # 除了最后的分类层
                classifier_input = layer(classifier_input)

            return {
                'physical_code': physical_code.detach(),
                'deep_features': deep_features.detach(),
                'classifier_features': classifier_input.detach(),
                'raw_signal': signal.detach()
            }

    def get_model_complexity(self):
        """
        获取模型复杂度信息

        Returns:
            complexity_info: 复杂度信息字典
        """
        # 计算各模块参数量
        mpie_params = sum(p.numel() for p in self.physical_encoder.parameters())
        amscnn_params = sum(p.numel() for p in self.adaptive_cnn.parameters())
        msgda_params = sum(p.numel() for p in self.domain_adapter.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        total_params = mpie_params + amscnn_params + msgda_params + classifier_params

        # 计算模型大小（MB）
        model_size_mb = total_params * 4 / (1024 ** 2)  # 假设float32

        return {
            'total_parameters': total_params,
            'mpie_parameters': mpie_params,
            'amscnn_parameters': amscnn_params,
            'msgda_parameters': msgda_params,
            'classifier_parameters': classifier_params,
            'model_size_mb': model_size_mb,
            'parameter_distribution': {
                'MPIE': mpie_params / total_params * 100,
                'AMSCNN': amscnn_params / total_params * 100,
                'MSGDA': msgda_params / total_params * 100,
                'Classifier': classifier_params / total_params * 100
            }
        }

    def enable_domain_adaptation(self, enable=True):
        """启用/禁用域适应"""
        for param in self.domain_adapter.parameters():
            param.requires_grad = enable

    def freeze_feature_extractor(self, freeze=True):
        """冻结/解冻特征提取器"""
        for param in self.physical_encoder.parameters():
            param.requires_grad = not freeze
        for param in self.adaptive_cnn.parameters():
            param.requires_grad = not freeze

    def get_attention_maps(self, signal):
        """
        获取注意力图（用于可视化）

        Args:
            signal: 输入信号

        Returns:
            attention_maps: 注意力图字典
        """
        physical_code = self.physical_encoder(signal)
        attention_maps = self.adaptive_cnn.get_attention_weights(signal, physical_code)

        # 添加物理特征重要性
        physical_importance = self.physical_encoder.get_feature_importance(signal)
        attention_maps['physical_importance'] = physical_importance

        return attention_maps

    def get_domain_adaptation_info(self, source_features, target_features):
        """
        获取域适应信息（用于分析）

        Args:
            source_features: 源域特征
            target_features: 目标域特征

        Returns:
            domain_info: 域适应信息字典
        """
        with torch.no_grad():
            _, adaptive_weight, loss_components = self.domain_adapter(
                source_features, target_features
            )

            disc_acc = self.domain_adapter.get_domain_discriminator_accuracy(
                source_features, target_features
            )

            return {
                'adaptive_weight': adaptive_weight.item(),
                'discriminator_accuracy': disc_acc,
                'loss_components': loss_components
            }


# 模型工厂函数
def create_pg_msac_net(config):
    """
    创建PG-MSAC-Net模型的工厂函数

    Args:
        config: 配置对象

    Returns:
        model: PG-MSAC-Net模型实例
    """
    model = PG_MSAC_Net(
        num_classes=config.data.num_classes,
        sample_len=config.data.sample_len,
        mpie_config=config.model.MPIEConfig(),
        amscnn_config=config.model.AMSCNNConfig(),
        msgda_config=config.model.MSGDAConfig()
    )

    return model


# 测试代码
if __name__ == '__main__':
    # 测试完整模型
    from types import SimpleNamespace

    # 创建测试配置
    mpie_config = SimpleNamespace(
        time_features_dim=8,
        freq_features_dim=7,
        tf_features_dim=5,
        time_hidden_dim=48,
        freq_hidden_dim=40,
        tf_hidden_dim=40,
        output_dim=128
    )

    amscnn_config = SimpleNamespace(
        kernel_sizes=[3, 7, 15],
        scale_channels=64,
        conv_channels=[64, 128, 256, 512],
        physical_modulation_dim=192,
        dropout_rate=0.2
    )

    msgda_config = SimpleNamespace(
        num_statistical_features=6,
        weight_net_hidden=[32, 16],
        discriminator_hidden=[256, 128],
        discriminator_dropout=0.2,
        mmd_sigma=1.0
    )

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PG_MSAC_Net(
        num_classes=4,
        sample_len=1024,
        mpie_config=mpie_config,
        amscnn_config=amscnn_config,
        msgda_config=msgda_config
    ).to(device)

    # 测试输入
    batch_size = 8
    source_signal = torch.randn(batch_size, 1, 1024).to(device)
    target_signal = torch.randn(batch_size, 1, 1024).to(device)

    print("Testing PG-MSAC-Net...")
    print(f"Input shape: {source_signal.shape}")

    # 测试普通前向传播
    logits = model(source_signal)
    print(f"Classification output shape: {logits.shape}")

    # 测试域适应模式
    # 首先获取目标域特征
    with torch.no_grad():
        target_features = model.extract_features(target_signal)['deep_features']

    logits, features, domain_loss, loss_info = model(
        source_signal, target_features, domain_adapt=True
    )

    print(f"Domain adaptation - Features shape: {features.shape}")
    print(f"Domain adaptation - Domain loss: {domain_loss.item():.4f}")
    print(f"Domain adaptation - Adaptive weight: {loss_info['adaptive_weight'].item():.4f}")

    # 测试模型复杂度
    complexity = model.get_model_complexity()
    print(f"\nModel Complexity:")
    print(f"Total parameters: {complexity['total_parameters']:,}")
    print(f"Model size: {complexity['model_size_mb']:.2f} MB")
    print("Parameter distribution:")
    for module, percentage in complexity['parameter_distribution'].items():
        print(f"  {module}: {percentage:.1f}%")

    # 测试特征提取
    feature_dict = model.extract_features(source_signal)
    print(f"\nFeature extraction:")
    for name, features in feature_dict.items():
        if name != 'raw_signal':
            print(f"  {name}: {features.shape}")

    # 测试注意力图
    attention_maps = model.get_attention_maps(source_signal[:2])  # 只用前2个样本测试
    print(f"\nAttention maps:")
    for name, attention in attention_maps.items():
        if isinstance(attention, dict):
            print(f"  {name}: dict with keys {list(attention.keys())}")
        else:
            print(f"  {name}: {attention.shape}")

    # 测试域适应信息
    domain_info = model.get_domain_adaptation_info(features, target_features)
    print(f"\nDomain adaptation info:")
    print(f"  Adaptive weight: {domain_info['adaptive_weight']:.4f}")
    print(f"  Discriminator accuracy: {domain_info['discriminator_accuracy']:.4f}")

    print("\nPG-MSAC-Net test completed successfully!")

    # 保存模型结构信息
    print(f"\nModel Summary:")
    print("=" * 50)
    print(f"Model: PG-MSAC-Net")
    print(f"Input: [batch_size, 1, {1024}]")
    print(f"Output: [batch_size, {4}]")
    print(f"Parameters: {complexity['total_parameters']:,}")
    print(f"Size: {complexity['model_size_mb']:.2f} MB")
    print("=" * 50)