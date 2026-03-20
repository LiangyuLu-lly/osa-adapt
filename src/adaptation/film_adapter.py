"""
FiLM (Feature-wise Linear Modulation) 适配器

通过条件向量生成per-channel的缩放(γ)和偏移(β)参数，
对特征进行仿射变换: γ(c) ⊙ x + β(c)
"""

import torch
import torch.nn as nn


class FiLMAdapter(nn.Module):
    """
    Feature-wise Linear Modulation适配器
    
    给定输入特征x和条件向量c，输出: γ(c) ⊙ x + β(c)
    其中γ和β由条件向量通过线性层生成。
    
    初始化时γ偏置为1、β偏置为0，确保未训练时为恒等映射。
    """
    
    def __init__(self, feature_dim: int, condition_dim: int, scale_factor: float = 0.1):
        """
        Args:
            feature_dim: 输入特征的通道数
            condition_dim: 条件向量的维度
            scale_factor: FiLM调制的缩放因子（默认0.1，使调制更温和）
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.gamma_layer = nn.Linear(condition_dim, feature_dim)
        self.beta_layer = nn.Linear(condition_dim, feature_dim)
        
        # 初始化为恒等映射: γ=1, β=0
        # 使用更小的初始化，让FiLM的影响更温和
        nn.init.ones_(self.gamma_layer.bias)
        nn.init.zeros_(self.gamma_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)
        nn.init.zeros_(self.beta_layer.weight)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特征张量, shape [B, C] 或 [B, C, T]
            condition: 条件向量, shape [B, condition_dim]
        
        Returns:
            调制后的特征, 与x同shape
        """
        # 生成gamma和beta，并缩放以减小影响
        gamma = 1.0 + self.scale_factor * (self.gamma_layer(condition) - 1.0)  # [B, C]
        beta = self.scale_factor * self.beta_layer(condition)    # [B, C]
        
        if x.dim() == 3:
            # 卷积特征: [B, C, T] -> 扩展gamma/beta为 [B, C, 1]
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        
        return gamma * x + beta
