"""
FiLM (Feature-wise Linear Modulation) Adapter

Implements the core FiLM transformation (Eq. 1 in the paper):

    h' = γ(c) ⊙ h + β(c)

where h is the intermediate feature, c is the severity condition vector,
and γ, β are generated from c via learned linear projections.

通过条件向量生成per-channel的缩放(γ)和偏移(β)参数，
对特征进行仿射变换: γ(c) ⊙ x + β(c)

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning
Layer", AAAI 2018.
"""

import torch
import torch.nn as nn


class FiLMAdapter(nn.Module):
    """
    Feature-wise Linear Modulation adapter.

    Given input features h and condition vector c, computes (Eq. 1):
        γ = W_γ · c + b_γ          (scale generator)
        β = W_β · c + b_β          (shift generator)
        h' = γ ⊙ h + β             (affine modulation)

    Identity initialization: γ bias=1, β bias=0, all weights=0.
    This ensures the adapter acts as an identity mapping before training,
    preserving the pretrained model's behavior at initialization.

    A scale_factor (default 0.1) further dampens the modulation magnitude,
    which is critical for stable adaptation on small clinical datasets
    (see ACM hypothesis, Section 3.2 in the paper).
    """

    def __init__(self, feature_dim: int, condition_dim: int, scale_factor: float = 0.1):
        """
        Args:
            feature_dim: Number of channels in the input feature map.
            condition_dim: Dimensionality of the condition vector c.
            scale_factor: Dampening factor for FiLM modulation (default 0.1).
                          Smaller values → gentler adaptation → less overfitting risk.
        """
        super().__init__()
        self.scale_factor = scale_factor

        # Linear projections: c → γ and c → β  (Eq. 1)
        self.gamma_layer = nn.Linear(condition_dim, feature_dim)
        self.beta_layer = nn.Linear(condition_dim, feature_dim)

        # Identity init: γ=1, β=0 → h' = 1·h + 0 = h (no-op before training)
        nn.init.ones_(self.gamma_layer.bias)
        nn.init.zeros_(self.gamma_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)
        nn.init.zeros_(self.beta_layer.weight)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation (Eq. 1).

        Args:
            x: Feature tensor, shape [B, C] or [B, C, T] (conv features).
            condition: Condition vector from SeverityConditioner, shape [B, condition_dim].

        Returns:
            Modulated features, same shape as x.
        """
        # Generate γ and β, dampened by scale_factor for stable adaptation
        # γ_eff = 1 + s·(γ_raw - 1), so at init γ_eff = 1 (identity)
        gamma = 1.0 + self.scale_factor * (self.gamma_layer(condition) - 1.0)  # [B, C]
        beta = self.scale_factor * self.beta_layer(condition)    # [B, C]

        if x.dim() == 3:
            # Conv features: [B, C, T] → expand γ/β to [B, C, 1] for broadcasting
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        # Eq. 1: h' = γ(c) ⊙ h + β(c)
        return gamma * x + beta
