#!/usr/bin/env python3
"""
Model definitions: MultiScaleRetention, DAMSRBlock, CNNDAMSRBiGRUModel, FocalLoss.
"""

import torch
import torch.nn as nn

from .config import get_runtime_device


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced data"""
    def __init__(self, alpha=0.5, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiScaleRetention(nn.Module):
    """
    多尺度保留机制 (Multi-Scale Retention)

    论文: Retentive Network: A Successor to Transformer for Large Language Models (NeurIPS 2023)

    创新点：
    1. 训练并行，推理递归（O(1)复杂度）
    2. 多尺度建模（短期、中期、长期依赖）
    3. 性能超过Transformer，速度更快
    4. 结合了RNN的效率和Transformer的表达能力
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(MultiScaleRetention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 衰减因子（固定值，适配短序列）
        # 方案B：保持2头和4头不变 + 地板值0.75
        # 原始公式：gamma = 0.90 - 0.05 * i
        # 地板值：0.75（确保所有头都有效，避免"瞎子头"）
        #
        # 不同头数的gamma值：
        # 2头: [0.90, 0.85] ← 与阶段1-3完全一致
        # 4头: [0.90, 0.85, 0.80, 0.75] ← 保持原始公式
        # 8头: [0.90, 0.879, 0.857, 0.836, 0.814, 0.793, 0.771, 0.75] ← 重分配

        base_gamma = [0.90 - 0.05 * i for i in range(num_heads)]
        floor = 0.75  # 地板值，确保最弱的头仍有5.6%权重@10bp

        if min(base_gamma) >= floor:
            # 没跌破地板，保持原始公式（2头和4头走这里）
            gamma_values = torch.tensor(base_gamma)
        else:
            # 跌破地板，在 [0.90, floor] 之间均匀分布（5头及以上走这里）
            gamma_values = torch.tensor([
                0.90 - ((0.90 - floor) / (num_heads - 1)) * i
                for i in range(num_heads)
            ])

        self.register_buffer('gamma', gamma_values)  # 固定值，不可训练

        # 群归一化（每个头一个组）
        self.group_norm = nn.GroupNorm(num_heads, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, length, d_model)
        返回: (batch, length, d_model)
        """
        batch, length, _ = x.shape

        # 投影到 Q, K, V
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.num_heads, self.head_dim)

        # 构建距离衰减权重（双向，不是因果）
        decay_weights = self._build_decay_weights(length, x.device)  # (length, length, num_heads)

        # 计算注意力分数
        # q: (batch, length, num_heads, head_dim)
        # k: (batch, length, num_heads, head_dim)
        scores = torch.einsum('blhd,bLhd->blLh', q, k)  # (batch, length, length, num_heads)

        # 缩放
        scores = scores / (self.head_dim ** 0.5)

        # 应用距离衰减权重
        scores = scores * decay_weights.unsqueeze(0)  # (batch, length, length, num_heads)

        # Softmax归一化（每个查询位置对所有键位置）
        attention = torch.softmax(scores, dim=2)  # 在键的维度上softmax
        attention = self.dropout(attention)

        # 加权求和
        output = torch.einsum('blLh,bLhd->blhd', attention, v)  # (batch, length, num_heads, head_dim)

        # 合并多头
        output = output.reshape(batch, length, self.d_model)

        # 群归一化
        output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)

        # 输出投影
        output = self.out_proj(output)

        return output

    def _build_decay_weights(self, length, device):
        """
        构建距离衰减权重矩阵（双向）

        返回: (length, length, num_heads)
        weights[i, j, h] = gamma[h]^|i-j|

        解释：
        - 距离越远，权重越小
        - 不同的头有不同的衰减率（多尺度）
        - 双向：可以看到前后的位置（适合IRES位点预测）
        """
        # 创建位置索引
        i_idx = torch.arange(length, device=device).unsqueeze(1)  # (length, 1)
        j_idx = torch.arange(length, device=device).unsqueeze(0)  # (1, length)

        # 计算绝对距离（双向）
        distance = torch.abs(i_idx - j_idx).float()  # (length, length)

        # 对每个头应用不同的衰减率
        weights = torch.zeros(length, length, self.num_heads, device=device)
        for h in range(self.num_heads):
            # gamma^|distance|
            # 距离为0（自己）：权重=1
            # 距离越大：权重越小
            weights[:, :, h] = torch.pow(self.gamma[h], distance)

        return weights


class DAMSRBlock(nn.Module):
    """
    DAMSR块 = MultiScaleRetention + FFN

    包含：
    1. 多尺度保留层（替代自注意力）
    2. 前馈网络（FFN）
    3. 残差连接和层归一化
    """
    def __init__(self, d_model, num_heads=4, ffn_dim=None, dropout=0.1):
        super(DAMSRBlock, self).__init__()

        if ffn_dim is None:
            ffn_dim = d_model * 4

        # 多尺度保留层
        self.retention = MultiScaleRetention(d_model, num_heads, dropout)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )

        # 层归一化
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, length, d_model)
        """
        # 保留层 + 残差
        x = x + self.retention(self.ln1(x))

        # FFN + 残差
        x = x + self.ffn(self.ln2(x))

        return x


class CNNDAMSRBiGRUModel(nn.Module):
    """
    CNN + DAMSR + BiGRU 双分支混合模型（优化版：不同聚合策略）

    架构：
    1. 共享CNN：提取局部序列特征（motif）
    2. 分支1 - DAMSR：多尺度衰减注意力（显式位置关系，并行计算）
    3. 分支2 - BiGRU：双向递归累积（隐式全局上下文，序列建模）
    4. 特征融合：拼接两个分支的特征（不同聚合策略）
    5. 分类头：预测中心位置是否是IRES位点

    创新点：
    - DAMSR（NeurIPS 2023）首次用于circRNA分析
    - 现代注意力机制 + 经典递归网络的互补
    - 显式多尺度衰减 + 隐式递归累积
    - 两种不同的序列建模范式融合

    互补性优化（关键改进）：
    - DAMSR：中心加权距离衰减池化 → 捕捉多尺度全局模式（整个窗口的统计特征）
    - BiGRU：中心位置特征 → 捕捉局部双向焦点（中心位点的上下文）
    - 避免特征重叠：不同聚合策略确保两个分支提取不同视角的信息
    """
    def __init__(self, window_size=151, num_filters=128, damsr_layers=1, num_heads=2,
                 bigru_hidden_size=128, bigru_num_layers=2, context_pool_gamma=0.97):
        super(CNNDAMSRBiGRUModel, self).__init__()

        self.window_size = window_size
        self.context_pool_gamma = context_pool_gamma
        self.bigru_hidden_size = bigru_hidden_size
        self.bigru_num_layers = bigru_num_layers

        # ===== 共享CNN部分（局部特征提取）=====
        self.conv1 = nn.Conv1d(4, num_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(num_filters*2, num_filters*2, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(num_filters*2)
        self.dropout3 = nn.Dropout(0.3)

        # ===== 分支1: DAMSR（多尺度衰减注意力）- 轻量化 =====
        self.damsr_blocks = nn.ModuleList([
            DAMSRBlock(
                d_model=num_filters*2,
                num_heads=num_heads,
                ffn_dim=num_filters*4,
                dropout=0.1
            )
            for _ in range(damsr_layers)
        ])

        # ===== 分支2: BiGRU（双向递归）=====
        gru_dropout = 0.0 if bigru_num_layers == 1 else 0.25
        self.bigru = nn.GRU(
            input_size=num_filters*2,
            hidden_size=bigru_hidden_size,
            num_layers=bigru_num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=True
        )

        # ===== 特征融合 + 分类头 =====
        # DAMSR分支维度: num_filters*2
        # BiGRU分支维度: bigru_hidden_size*2 (双向)
        fused_dim = num_filters*2 + bigru_hidden_size*2
        self.fc1 = nn.Linear(fused_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """
        输入: x (batch, window_size, 4) - one-hot编码
        输出: (batch, 1) - 预测概率
        """
        # ===== 共享CNN部分 =====
        x = x.transpose(1, 2)  # (batch, 4, window_size)

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # ⭐ 分叉点：DAMSR使用conv2的输出（浅层特征，更多局部细节）
        damsr_input = x.transpose(1, 2)  # (batch, window_size, num_filters*2)

        # CNN第3层（只给BiGRU使用）
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        # ⭐ BiGRU使用conv3的输出（深层特征，更抽象的模式）
        bigru_input = x.transpose(1, 2)  # (batch, window_size, num_filters*2)

        # ===== 分支1: DAMSR（显式注意力，处理浅层特征）=====
        damsr_out = damsr_input
        for damsr_block in self.damsr_blocks:
            damsr_out = damsr_block(damsr_out)

        # DAMSR 分支作为上下文建模器，不直接承担中心判别。
        # 因此保留窗口级汇总，但用"以中心为基准的距离衰减汇总"替代简单平均，
        # 让中心附近上下文获得更大权重，同时保留远端辅助信息。
        center_idx = self.window_size // 2
        positions = torch.arange(self.window_size, device=damsr_out.device)
        distance = torch.abs(positions - center_idx).float()
        pool_weights = torch.pow(self.context_pool_gamma, distance)
        pool_weights = pool_weights / pool_weights.sum()
        damsr_feat = torch.sum(damsr_out * pool_weights.view(1, self.window_size, 1), dim=1)

        # ===== 分支2: BiGRU（递归累积，处理深层特征）=====
        bigru_out, _ = self.bigru(bigru_input)  # (batch, window_size, bigru_hidden_size*2)

        # ⭐ BiGRU特征：中心位置（捕捉局部双向焦点）
        bigru_feat = bigru_out[:, center_idx, :]  # (batch, bigru_hidden_size*2) - 局部焦点

        # ===== 特征融合 =====
        combined = torch.cat([damsr_feat, bigru_feat], dim=1)  # (batch, fused_dim)

        # ===== 分类头（渐进降维）=====
        x = self.fc1(combined)
        x = self.bn_fc1(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = torch.relu(x)
        x = self.dropout5(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x


def build_cnn_model(window_size=151, num_filters=128,
                    bigru_hidden_size=128, bigru_num_layers=2,
                    damsr_layers=1, damsr_num_heads=2, context_pool_gamma=0.97):
    """
    构建模型：CNN + DAMSR + BiGRU 双分支（优化版：不同聚合策略）

    参数:
        window_size: 窗口大小
        num_filters: CNN滤波器数量
        bigru_hidden_size: BiGRU隐藏层大小
        bigru_num_layers: BiGRU层数
        damsr_layers: DAMSR层数
        damsr_num_heads: DAMSR多头注意力数量
        context_pool_gamma: 中心加权池化衰减率
    """
    device = get_runtime_device()

    model = CNNDAMSRBiGRUModel(
        window_size=window_size,
        num_filters=num_filters,
        damsr_layers=damsr_layers,
        num_heads=damsr_num_heads,
        bigru_hidden_size=bigru_hidden_size,
        bigru_num_layers=bigru_num_layers,
        context_pool_gamma=context_pool_gamma
    )

    model = model.to(device)

    return model