"""
增强版渐进式适应器 (EnhancedProgressiveAdapter)

在 ProgressiveAdapter 基础上增强适应能力：
1. Phase 2 中除 FiLM 外，额外解冻最后 1-2 个卷积块（部分微调）
2. Cosine annealing 学习率调度 + warmup
3. 更长训练周期 (max_epochs=100, patience=15)
4. 差异化学习率：FiLM 层 1e-3, 解冻层 1e-4

Requirements: 3.1, 3.2, 3.3
"""

import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .indomain_pretrainer import CosineAnnealingWithWarmup
from .progressive_adapter import ProgressiveAdapter

logger = logging.getLogger(__name__)


def _get_conv_blocks(model: nn.Module) -> Optional[nn.ModuleList]:
    """从模型中提取卷积块列表。

    支持 FiLMWrappedChambon（base_model.conv_blocks）和
    直接具有 conv_blocks 属性的模型。

    Args:
        model: FiLM 包装模型或基础模型

    Returns:
        卷积块的 ModuleList，若无法提取则返回 None
    """
    # FiLMWrappedChambon: model.base_model.conv_blocks
    base = getattr(model, "base_model", model)
    blocks = getattr(base, "conv_blocks", None)
    if blocks is not None and isinstance(blocks, nn.ModuleList):
        return blocks
    return None


def _get_cnn_module(model: nn.Module) -> Optional[nn.Module]:
    """从模型中提取 CNN 模块（用于 TinySleepNet 类架构）。

    Args:
        model: FiLM 包装模型或基础模型

    Returns:
        CNN 模块，若无法提取则返回 None
    """
    base = getattr(model, "base_model", model)
    cnn = getattr(base, "cnn", None)
    return cnn


class EnhancedProgressiveAdapter(ProgressiveAdapter):
    """增强版渐进式适应器

    关键改进（Req 3.1, 3.2, 3.3）：
    1. Phase 2 中除 FiLM 外，额外解冻最后 N 个卷积块（部分微调）
    2. Cosine annealing 学习率调度 + warmup（前 10% epochs）
    3. 更长训练周期 (max_epochs=100, patience=15)
    4. 差异化学习率：FiLM 层 lr_film, 解冻层 lr_backbone
    """

    def __init__(
        self,
        model: nn.Module,
        conditioner: nn.Module,
        loss_fn: nn.Module,
        lr_film: float = 1e-3,
        lr_backbone: float = 1e-4,
        patience: int = 15,
        max_epochs: int = 100,
        warmup_ratio: float = 0.1,
        unfreeze_last_n: int = 1,
    ):
        """
        Args:
            model: FiLM 包装后的模型（需有 get_trainable_params() 方法）
            conditioner: SeverityConditioner（已包含在 model 中，此处保留引用）
            loss_fn: SeverityAwareN1Loss，forward(inputs, targets, severity)
            lr_film: FiLM 层和 Conditioner 的学习率（Req 3.1 差异化学习率）
            lr_backbone: 解冻卷积块的学习率（Req 3.1 差异化学习率）
            patience: 早停耐心值（Req 3.3）
            max_epochs: 最大训练轮数（Req 3.3）
            warmup_ratio: warmup 占总 epoch 的比例（Req 3.2）
            unfreeze_last_n: 解冻最后 N 个卷积块（Req 3.1）
        """
        # 使用 lr_film 作为基类的 lr 参数
        super().__init__(
            model=model,
            conditioner=conditioner,
            loss_fn=loss_fn,
            lr=lr_film,
            patience=patience,
            max_epochs=max_epochs,
        )
        self.lr_film = lr_film
        self.lr_backbone = lr_backbone
        self.warmup_ratio = warmup_ratio
        self.unfreeze_last_n = unfreeze_last_n

    def _get_backbone_params_to_unfreeze(self) -> List[nn.Parameter]:
        """获取需要解冻的最后 N 个卷积块的参数。

        支持两种模型架构：
        - Chambon2018 类（有 conv_blocks）：解冻最后 N 个 conv_block
        - TinySleepNet 类（有 cnn）：解冻 CNN 中最后 N 个卷积层

        Req 3.1: 额外解冻 Base_Model 最后 1-2 个卷积块

        Returns:
            需要解冻的参数列表
        """
        params: List[nn.Parameter] = []

        # 尝试 Chambon2018 类架构（conv_blocks）
        conv_blocks = _get_conv_blocks(self.model)
        if conv_blocks is not None and len(conv_blocks) > 0:
            n = min(self.unfreeze_last_n, len(conv_blocks))
            if n > 0:
                for block in conv_blocks[-n:]:
                    params.extend(block.parameters())
            return params

        # 尝试 TinySleepNet 类架构（cnn Sequential）
        cnn = _get_cnn_module(self.model)
        if cnn is not None:
            # 从 Sequential 中提取包含可学习参数的子模块
            conv_layers = [
                m for m in cnn.modules()
                if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d))
            ]
            if conv_layers:
                # 按逆序取最后 N 个卷积+BN 对
                # 每个"块"包含一个 Conv + 一个 BN
                n_layers = min(self.unfreeze_last_n * 2, len(conv_layers))
                if n_layers > 0:
                    for layer in conv_layers[-n_layers:]:
                        params.extend(layer.parameters())
            return params

        logger.warning(
            "无法从模型中提取卷积块，unfreeze_last_n=%d 将不生效",
            self.unfreeze_last_n,
        )
        return params

    def _setup_param_groups(self) -> List[Dict]:
        """设置差异化学习率的参数组。

        Req 3.1: FiLM 层 lr_film, 解冻卷积块 lr_backbone

        Returns:
            优化器参数组列表
        """
        # 1. 先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 解冻 FiLM + Conditioner 参数
        film_params = self.model.get_trainable_params()
        for param in film_params:
            param.requires_grad = True

        # 3. 解冻最后 N 个卷积块（Req 3.1）
        backbone_params = self._get_backbone_params_to_unfreeze()
        for param in backbone_params:
            param.requires_grad = True

        # 4. 构建参数组（去重：backbone 参数可能与 film_params 重叠）
        backbone_param_ids = {id(p) for p in backbone_params}
        film_only_params = [
            p for p in film_params if id(p) not in backbone_param_ids
        ]

        param_groups = []
        if film_only_params:
            param_groups.append({
                "params": film_only_params,
                "lr": self.lr_film,
                "name": "film",
            })
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": self.lr_backbone,
                "name": "backbone",
            })

        return param_groups

    def phase2_enhanced_finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """增强版 Phase 2：部分微调 + 差异化学习率 + LR 调度。

        Req 3.1: 解冻最后 N 个卷积块 + 差异化学习率
        Req 3.2: Cosine annealing + warmup（前 10% epochs）
        Req 3.3: max_epochs=100, patience=15

        Args:
            train_loader: 训练数据 DataLoader
            val_loader: 验证数据 DataLoader

        Returns:
            包含训练历史的字典
        """
        # 设置参数组（差异化学习率）
        param_groups = self._setup_param_groups()

        if not param_groups:
            logger.warning("没有可训练参数，跳过 Phase 2")
            return {
                "phase": 2,
                "history": [],
                "best_val_accuracy": 0.0,
                "total_epochs": 0,
                "early_stopped": False,
            }

        # 优化器
        optimizer = torch.optim.Adam(param_groups)

        # Cosine annealing + warmup 学习率调度（Req 3.2）
        warmup_epochs = max(1, int(self.max_epochs * self.warmup_ratio))
        scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            total_epochs=self.max_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=1e-6,
        )

        # 记录可训练参数统计
        total_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            "Enhanced Phase 2: lr_film=%.1e, lr_backbone=%.1e, "
            "unfreeze_last_n=%d, warmup=%d epochs, "
            "max_epochs=%d, patience=%d, trainable_params=%d",
            self.lr_film, self.lr_backbone, self.unfreeze_last_n,
            warmup_epochs, self.max_epochs, self.patience, total_trainable,
        )

        # 训练历史
        history: List[Dict] = []

        # 早停状态（Req 3.3）
        best_val_accuracy = -1.0
        best_state_dict = None
        epochs_without_improvement = 0

        self.model.train()

        for epoch in range(self.max_epochs):
            # 更新学习率（Req 3.2）
            scheduler.step(epoch)
            current_lrs = scheduler.get_last_lr()

            # --- 训练阶段 ---
            train_loss = self._train_one_epoch(train_loader, optimizer)

            # --- 验证阶段 ---
            val_accuracy, n1_recall = self._validate(val_loader)

            # 记录历史
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "n1_recall": n1_recall,
                "lrs": current_lrs,
            }
            history.append(epoch_record)

            logger.info(
                "Enhanced Phase 2 Epoch %d: loss=%.4f, val_acc=%.4f, "
                "n1_recall=%.4f, lrs=%s",
                epoch, train_loss, val_accuracy, n1_recall,
                [f"{lr:.2e}" for lr in current_lrs],
            )

            # 早停检查（Req 3.3: patience=15）
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                logger.info(
                    "早停触发: %d epochs 无改善 (patience=%d), "
                    "最佳 val_accuracy=%.4f",
                    epochs_without_improvement,
                    self.patience,
                    best_val_accuracy,
                )
                break

        # 恢复最佳模型权重
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return {
            "phase": 2,
            "history": history,
            "best_val_accuracy": best_val_accuracy,
            "total_epochs": len(history),
            "early_stopped": epochs_without_improvement >= self.patience,
            "total_trainable_params": total_trainable,
        }
