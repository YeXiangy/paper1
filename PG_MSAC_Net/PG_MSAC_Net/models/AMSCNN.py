class AdaptiveMultiScaleCNN(nn.Module):
    """
    自适应多尺度CNN (AMSCNN)
    通过物理调制机制实现多尺度特征学习
    """

    def __init__(self, physical_dim=128, output_dim=512):
        super(AdaptiveMultiScaleCNN, self).__init__()

        # 多尺度并行分支
        self.scale1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.scale2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.scale3 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 物理特征调制器
        self.physical_modulator = nn.Sequential(
            nn.Linear(physical_dim, 192),  # 64*3 = 192
            nn.ReLU(inplace=True),
            nn.Linear(192, 192),
            nn.Sigmoid()  # 生成[0,1]的调制权重
        )

        # 门控融合机制
        self.gate_network = nn.Sequential(
            nn.Linear(physical_dim, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 192),
            nn.Sigmoid()
        )

        # 后续卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(512, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )

        self.flatten = nn.Flatten()

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(output_dim + physical_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, signal, physical_code):
        """
        前向传播
        Args:
            signal: 输入信号 [B, 1, 1024]
            physical_code: 物理编码 [B, 128]
        Returns:
            deep_features: 深度特征 [B, 512]
        """
        batch_size = signal.shape[0]

        # 多尺度特征提取
        feat1 = self.scale1(signal)  # [B, 64, 512]
        feat2 = self.scale2(signal)  # [B, 64, 512]
        feat3 = self.scale3(signal)  # [B, 64, 512]

        # 多尺度特征拼接
        multi_scale_feat = torch.cat([feat1, feat2, feat3], dim=1)  # [B, 192, 512]

        # 物理特征调制
        modulation_weights = self.physical_modulator(physical_code)  # [B, 192]
        gate_weights = self.gate_network(physical_code)  # [B, 192]

        # 应用调制权重 (广播机制)
        modulated_feat = multi_scale_feat * modulation_weights.unsqueeze(-1)  # [B, 192, 512]
        gated_feat = modulated_feat * gate_weights.unsqueeze(-1)  # [B, 192, 512]

        # 残差连接
        enhanced_feat = multi_scale_feat + gated_feat  # [B, 192, 512]

        # 深度特征提取
        deep_features = self.conv_layers(enhanced_feat)  # [B, 512, 1]
        deep_features = self.flatten(deep_features)  # [B, 512]

        # 最终特征融合
        combined_features = torch.cat([deep_features, physical_code], dim=1)  # [B, 512+128]
        final_features = self.feature_fusion(combined_features)  # [B, 512]

        return final_features