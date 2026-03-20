"""
Severity Conditioner — Clinical Feature Encoder (Eq. 2 in the paper)

Maps patient-level clinical variables to a fixed-dimensional condition vector c
that drives FiLM modulation. The encoding pipeline:

    c = MLP( [z_norm(AHI, age, BMI) ; Emb(severity) ; Emb(sex)] )     (Eq. 2)

where z_norm is z-score normalization using training-set statistics,
Emb is a learned embedding lookup, and [;] denotes concatenation.

将患者级别的临床特征（AHI值、OSA严重程度类别、年龄、性别、BMI）
编码为固定维度的条件向量，用于FiLM适配器的特征调制。
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SeverityConditioner(nn.Module):
    """
    Encodes patient clinical features into a condition vector c (Eq. 2).

    Input features:
    - AHI (continuous, z-normalized)
    - OSA severity category (4-class embedding: Normal/Mild/Moderate/Severe)
    - Age (continuous, z-normalized)
    - Sex (2-class embedding: Female/Male)
    - BMI (continuous, z-normalized)

    Output: condition vector c ∈ R^{condition_dim} (default 64)

    The two-layer MLP provides sufficient capacity to capture nonlinear
    interactions between clinical variables while keeping parameter count low
    (~5K params), consistent with the ACM hypothesis.
    """

    def __init__(
        self,
        condition_dim: int = 64,
        severity_classes: int = 4,
        sex_classes: int = 2,
        embedding_dim: int = 16,
    ):
        """
        Args:
            condition_dim: Output condition vector dimensionality.
            severity_classes: Number of OSA severity levels (Normal/Mild/Moderate/Severe).
            sex_classes: Number of sex categories (Female/Male).
            embedding_dim: Embedding dimension for categorical features.
        """
        super().__init__()
        self.condition_dim = condition_dim

        # Categorical feature embeddings (Eq. 2: Emb(severity), Emb(sex))
        self.severity_embedding = nn.Embedding(severity_classes, embedding_dim)
        self.sex_embedding = nn.Embedding(sex_classes, embedding_dim)

        # Input dim: 3 continuous (AHI, age, BMI) + 2 × embedding_dim (severity, sex)
        input_dim = 3 + 2 * embedding_dim

        # Two-layer MLP: concat features → condition vector c (Eq. 2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        # 连续特征标准化参数 (训练时通过 set_normalization_stats 设置)
        self.register_buffer("continuous_mean", torch.zeros(3))
        self.register_buffer("continuous_std", torch.ones(3))

    def set_normalization_stats(
        self, means: torch.Tensor, stds: torch.Tensor
    ) -> None:
        """设置连续特征的标准化参数（来自训练集统计量）

        Args:
            means: 连续特征均值, shape [3] (AHI, 年龄, BMI)
            stds: 连续特征标准差, shape [3]
        """
        self.continuous_mean = means
        self.continuous_std = stds.clamp(min=1e-6)

    def _handle_missing_values(
        self, continuous: torch.Tensor
    ) -> torch.Tensor:
        """检测并填充连续特征中的缺失值 (NaN)

        使用 continuous_mean (可作为训练集中位数的默认值) 填充缺失值，
        并记录警告日志。

        Args:
            continuous: 连续特征张量, shape [B, 3]

        Returns:
            填充后的连续特征张量
        """
        nan_mask = torch.isnan(continuous)
        if nan_mask.any():
            num_nan = nan_mask.sum().item()
            feature_names = ["AHI", "age", "BMI"]
            nan_per_feature = nan_mask.sum(dim=0)
            details = ", ".join(
                f"{feature_names[i]}={int(nan_per_feature[i].item())}"
                for i in range(3)
                if nan_per_feature[i] > 0
            )
            logger.warning(
                "SeverityConditioner: 检测到 %d 个缺失值 (%s)，"
                "使用训练集统计量填充",
                num_nan,
                details,
            )
            # 用 continuous_mean 填充 NaN
            fill_values = self.continuous_mean.expand_as(continuous)
            continuous = torch.where(nan_mask, fill_values, continuous)
        return continuous

    def forward(
        self,
        ahi: torch.Tensor,
        severity: torch.Tensor,
        age: torch.Tensor,
        sex: torch.Tensor,
        bmi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ahi: AHI值, shape [B]
            severity: OSA严重程度类别, shape [B] (int, 0-3)
            age: 年龄, shape [B]
            sex: 性别, shape [B] (int, 0=Female, 1=Male)
            bmi: BMI值, shape [B]

        Returns:
            条件向量, shape [B, condition_dim]
        """
        # 组合连续特征
        continuous = torch.stack([ahi, age, bmi], dim=-1)  # [B, 3]

        # 缺失值处理 (Req 2.4)
        continuous = self._handle_missing_values(continuous)

        # 标准化连续特征 (Req 2.2)
        continuous = (continuous - self.continuous_mean) / self.continuous_std

        # 嵌入分类特征 (Req 2.3)
        sev_emb = self.severity_embedding(severity)  # [B, emb_dim]
        sex_emb = self.sex_embedding(sex)  # [B, emb_dim]

        # 拼接并通过MLP映射 (Req 2.5)
        combined = torch.cat([continuous, sev_emb, sex_emb], dim=-1)
        return self.mlp(combined)
