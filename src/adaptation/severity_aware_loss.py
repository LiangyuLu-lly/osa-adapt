"""
严重程度感知的N1 Focal Loss

在N1AwareFocalLoss基础上，根据OSA严重程度动态调整focal gamma：
- Normal (0): gamma_n1 = gamma_n1_base (默认2.5)
- Mild (1):   gamma_n1 = gamma_n1_base + 1 * increment (默认3.0)
- Moderate (2): gamma_n1 = gamma_n1_base + 2 * increment (默认3.5)
- Severe (3): gamma_n1 = gamma_n1_base + 3 * increment (默认4.0)

严重OSA患者的N1更难检测（更多碎片化），需要更强的聚焦。

Requirements: 3.1, 3.2, 3.3, 3.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SeverityAwareN1Loss(nn.Module):
    """
    严重程度感知的N1 Focal Loss

    核心特性：
    1. N1使用独立的focal gamma，且gamma随OSA严重程度递增 (Req 3.1, 3.3)
    2. N1类别权重 >= 其他类别平均权重的n1_weight_multiplier倍 (Req 3.2)
    3. 全无效标签batch返回零损失，不产生梯度错误 (Req 3.4)

    参数:
        num_classes: 睡眠分期类别数（AASM: W, N1, N2, N3, REM = 5）
        n1_class_index: N1在AASM标准中的索引（默认1）
        gamma_default: 非N1类别的focal gamma
        gamma_n1_base: N1类别的基础focal gamma（Normal严重程度时使用）
        gamma_n1_increment: 每增加一级严重程度，gamma增加的量
        n1_weight_multiplier: N1权重至少为其他类别平均权重的倍数
        label_smoothing: 标签平滑系数
    """

    def __init__(
        self,
        num_classes: int = 5,
        n1_class_index: int = 1,
        gamma_default: float = 2.0,
        gamma_n1_base: float = 2.5,
        gamma_n1_increment: float = 0.5,
        n1_weight_multiplier: float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n1_class_index = n1_class_index
        self.gamma_default = gamma_default
        self.gamma_n1_base = gamma_n1_base
        self.gamma_n1_increment = gamma_n1_increment
        self.n1_weight_multiplier = n1_weight_multiplier
        self.label_smoothing = label_smoothing

        # 类别权重，可通过set_class_weights更新
        self.register_buffer("class_weights", torch.ones(num_classes))

    def set_class_weights(self, class_counts: torch.Tensor) -> None:
        """
        根据类别分布设置权重，并确保N1权重满足约束。

        使用逆频率权重的平方根，然后强制N1权重 >= 其他类别平均权重 × n1_weight_multiplier。

        参数:
            class_counts: 每个类别的样本数, shape [num_classes]
        """
        assert len(class_counts) == self.num_classes, (
            f"class_counts长度({len(class_counts)})与num_classes({self.num_classes})不匹配"
        )
        total = class_counts.sum().float()
        n_classes = self.num_classes

        # 基础权重：逆频率的平方根
        raw_weights = total / (n_classes * class_counts.float().clamp(min=1))
        weights = raw_weights ** 0.5
        # 归一化使均值为1
        weights = weights / weights.mean()

        # 强制N1权重约束
        weights = self._enforce_n1_weight_constraint(weights)

        self.class_weights = weights.to(self.class_weights.device)

    def _enforce_n1_weight_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """
        确保N1权重 >= 其他类别平均权重 × n1_weight_multiplier。

        参数:
            weights: 类别权重, shape [num_classes]

        返回:
            调整后的权重
        """
        weights = weights.clone()
        mask = torch.ones(self.num_classes, dtype=torch.bool)
        mask[self.n1_class_index] = False
        other_mean = weights[mask].mean()

        min_n1_weight = other_mean * self.n1_weight_multiplier
        if weights[self.n1_class_index] < min_n1_weight:
            weights[self.n1_class_index] = min_n1_weight

        return weights

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        severity: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算严重程度感知的N1 Focal Loss。

        参数:
            inputs: 模型预测logits, shape [B, C]
            targets: 真实标签, shape [B]，-1表示无效标签
            severity: OSA严重程度, shape [B]，int 0-3
                      (0=Normal, 1=Mild, 2=Moderate, 3=Severe)

        返回:
            loss: 标量损失值
        """
        # 过滤无效标签 (Req 3.4)
        valid_mask = targets >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        valid_inputs = inputs[valid_mask]
        valid_targets = targets[valid_mask]
        valid_severity = severity[valid_mask]

        # 标准交叉熵（不reduction）
        ce_loss = F.cross_entropy(
            valid_inputs,
            valid_targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # p_t: 正确类别的概率
        p_t = torch.exp(-ce_loss)

        # 严重程度依赖的N1 gamma (Req 3.1, 3.3)
        # gamma_n1 = gamma_n1_base + severity * gamma_n1_increment
        severity_gamma = (
            self.gamma_n1_base + valid_severity.float() * self.gamma_n1_increment
        )

        # N1样本使用severity_gamma，非N1样本使用gamma_default
        gamma_per_sample = torch.where(
            valid_targets == self.n1_class_index,
            severity_gamma,
            torch.full_like(severity_gamma, self.gamma_default),
        )

        # Focal调制因子
        focal_weight = (1 - p_t) ** gamma_per_sample

        # 应用类别权重 (Req 3.2)
        # 确保 class_weights 与 inputs 在同一设备上
        class_weights = self.class_weights.to(valid_inputs.device)
        alpha_t = class_weights[valid_targets]

        return (alpha_t * focal_weight * ce_loss).mean()
