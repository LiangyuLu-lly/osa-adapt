"""
Severity-Aware N1 Focal Loss (Eq. 3 in the paper)

Extends standard focal loss with severity-dependent focusing for N1 detection:

    L = -α_c · (1 - p_t)^{γ(c,s)} · log(p_t)                    (Eq. 3)

where:
    γ(c,s) = γ_base + s · Δγ    for class c = N1                  (Eq. 4)
    γ(c,s) = γ_default           for class c ≠ N1

    s ∈ {0,1,2,3} is the OSA severity level (Normal/Mild/Moderate/Severe)
    Δγ (gamma_n1_increment) controls how much harder the loss focuses on
    N1 epochs from more severe patients.

Rationale: Severe OSA causes more sleep fragmentation, making N1 detection
harder. Higher γ for severe patients forces the model to focus more on
these difficult-to-classify N1 epochs.

在N1AwareFocalLoss基础上，根据OSA严重程度动态调整focal gamma：
- Normal (0): gamma_n1 = gamma_n1_base (默认2.5)
- Mild (1):   gamma_n1 = gamma_n1_base + 1 × increment (默认3.0)
- Moderate (2): gamma_n1 = gamma_n1_base + 2 × increment (默认3.5)
- Severe (3): gamma_n1 = gamma_n1_base + 3 × increment (默认4.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SeverityAwareN1Loss(nn.Module):
    """
    Severity-Aware N1 Focal Loss (Eq. 3-4).

    Key properties:
    1. N1 uses a severity-dependent focal γ that increases with OSA severity (Eq. 4)
    2. N1 class weight ≥ mean(other weights) × n1_weight_multiplier (class balancing)
    3. Batches with all-invalid labels return zero loss gracefully
    4. Label smoothing (default 0.05) for regularization

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

        # Standard cross-entropy (unreduced) with label smoothing
        ce_loss = F.cross_entropy(
            valid_inputs,
            valid_targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # p_t = probability assigned to the correct class (Eq. 3)
        p_t = torch.exp(-ce_loss)

        # Severity-dependent N1 gamma (Eq. 4):
        #   γ_N1(s) = γ_base + s · Δγ
        # where s ∈ {0,1,2,3} is the severity level
        severity_gamma = (
            self.gamma_n1_base + valid_severity.float() * self.gamma_n1_increment
        )

        # Per-sample γ: N1 samples use severity_gamma, others use γ_default
        gamma_per_sample = torch.where(
            valid_targets == self.n1_class_index,
            severity_gamma,
            torch.full_like(severity_gamma, self.gamma_default),
        )

        # Focal modulation factor: (1 - p_t)^γ  (Eq. 3)
        focal_weight = (1 - p_t) ** gamma_per_sample

        # Apply class weights α_c (Eq. 3)
        class_weights = self.class_weights.to(valid_inputs.device)
        alpha_t = class_weights[valid_targets]

        # Final loss: L = mean( α_c · (1-p_t)^γ · CE )  (Eq. 3)
        return (alpha_t * focal_weight * ce_loss).mean()
