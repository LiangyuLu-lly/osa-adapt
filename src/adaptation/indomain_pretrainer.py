"""
域内预训练器 (InDomainPretrainer)

在 328 名临床 OSA 患者数据上对 Chambon2018 和 TinySleepNet 进行充分的域内预训练，
建立消除域差异的强基线。

关键改进（相比当前预训练）：
- max_epochs: 30 → 100
- 学习率调度: cosine annealing with warmup
- 数据增强: 时间偏移、高斯噪声、幅度缩放
- 类别平衡: 加权采样
- patience: 7 → 20

Requirements: 2.1, 2.2, 2.3, 2.4
"""

import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model_builder import build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据增强配置
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """数据增强配置

    Attributes:
        time_shift_max: 最大时间偏移（采样点数）
        gaussian_noise_std: 高斯噪声标准差
        amplitude_scale_range: 幅度缩放范围 (min, max)
        enabled: 是否启用数据增强
    """
    time_shift_max: int = 50
    gaussian_noise_std: float = 0.01
    amplitude_scale_range: Tuple[float, float] = (0.9, 1.1)
    enabled: bool = True


# ---------------------------------------------------------------------------
# 数据增强函数
# ---------------------------------------------------------------------------

def apply_augmentation(
    signal: torch.Tensor,
    config: AugmentationConfig,
) -> torch.Tensor:
    """对单个 EEG 信号应用数据增强。

    增强方式（Req 2.3）：
    1. 时间偏移：随机循环移位
    2. 高斯噪声：叠加随机噪声
    3. 幅度缩放：随机缩放信号幅度

    Args:
        signal: 形状 [C, T] 的 EEG 信号张量
        config: 增强配置

    Returns:
        增强后的信号张量，形状不变
    """
    if not config.enabled:
        return signal

    augmented = signal.clone()

    # 1. 时间偏移（循环移位）
    if config.time_shift_max > 0:
        shift = torch.randint(
            -config.time_shift_max, config.time_shift_max + 1, (1,)
        ).item()
        if shift != 0:
            augmented = torch.roll(augmented, shifts=int(shift), dims=-1)

    # 2. 高斯噪声
    if config.gaussian_noise_std > 0:
        noise = torch.randn_like(augmented) * config.gaussian_noise_std
        augmented = augmented + noise

    # 3. 幅度缩放
    lo, hi = config.amplitude_scale_range
    if lo != 1.0 or hi != 1.0:
        scale = torch.empty(1).uniform_(lo, hi).item()
        augmented = augmented * scale

    return augmented


# ---------------------------------------------------------------------------
# 带增强的 Dataset 包装器
# ---------------------------------------------------------------------------

class AugmentedDataset(Dataset):
    """为已有 Dataset 添加在线数据增强的包装器。

    仅在训练时启用增强，验证时直接返回原始数据。
    """

    def __init__(
        self,
        base_dataset: Dataset,
        augmentation_config: AugmentationConfig,
        training: bool = True,
    ):
        self.base_dataset = base_dataset
        self.aug_config = augmentation_config
        self.training = training

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]

        if len(item) == 3:
            signal, label, features = item
        elif len(item) == 2:
            signal, label = item
            features = None
        else:
            return item

        # 训练时应用增强
        if self.training and self.aug_config.enabled:
            signal = apply_augmentation(signal, self.aug_config)

        if features is not None:
            return signal, label, features
        return signal, label


# ---------------------------------------------------------------------------
# Cosine Annealing 学习率调度（含 warmup）
# ---------------------------------------------------------------------------

class CosineAnnealingWithWarmup:
    """Cosine annealing 学习率调度器，含线性 warmup 阶段。

    Warmup 阶段（epoch < warmup_epochs）：学习率从 0 线性递增到 base_lr
    Cosine 阶段（epoch >= warmup_epochs）：学习率按余弦曲线从 base_lr 递减到 min_lr

    Req 2.3: 使用学习率调度（cosine annealing）
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        # 记录每个参数组的初始学习率
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._current_epoch = 0

    def step(self, epoch: Optional[int] = None) -> None:
        """更新学习率。

        Args:
            epoch: 当前 epoch 编号（从 0 开始）。若为 None 则自动递增。
        """
        if epoch is not None:
            self._current_epoch = epoch
        else:
            self._current_epoch += 1

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self.get_lr(base_lr, self._current_epoch)

    def get_lr(self, base_lr: float, epoch: int) -> float:
        """计算指定 epoch 的学习率。"""
        if epoch < self.warmup_epochs:
            # 线性 warmup：从 0 递增到 base_lr
            return base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing：从 base_lr 递减到 min_lr
            cosine_epochs = self.total_epochs - self.warmup_epochs
            if cosine_epochs <= 0:
                return base_lr
            progress = (epoch - self.warmup_epochs) / cosine_epochs
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (base_lr - self.min_lr) * cosine_decay

    def get_last_lr(self) -> List[float]:
        """返回当前各参数组的学习率。"""
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# 类别平衡采样器
# ---------------------------------------------------------------------------

def create_class_balanced_sampler(
    dataset: Dataset,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """创建类别平衡的加权随机采样器。

    为每个样本分配权重 = 1 / (该类别的样本数)，使得每个类别被采样的概率相等。

    Req 2.3: 类别平衡采样

    Args:
        dataset: 数据集，每个样本的第二个元素为标签
        num_samples: 每个 epoch 的采样数量，默认为数据集大小

    Returns:
        WeightedRandomSampler 实例
    """
    # 收集所有标签
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        if len(item) >= 2:
            label = item[1]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(int(label))
        else:
            labels.append(0)

    labels = np.array(labels)
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # 每个类别的权重 = 1 / 类别样本数
    class_weight_map = {
        cls: 1.0 / count for cls, count in zip(unique_classes, class_counts)
    }

    # 为每个样本分配权重
    sample_weights = np.array([class_weight_map[l] for l in labels])
    sample_weights = torch.from_numpy(sample_weights).double()

    if num_samples is None:
        num_samples = len(dataset)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


# ---------------------------------------------------------------------------
# InDomainPretrainer 主类
# ---------------------------------------------------------------------------

class InDomainPretrainer:
    """域内预训练器

    在临床 OSA 患者数据上对 Chambon2018 和 TinySleepNet 进行充分的域内预训练。
    使用 5 折患者级别交叉验证，每折独立训练并保存最佳检查点。

    关键改进（相比当前预训练）：
    - max_epochs: 30 → 100
    - 学习率调度: cosine annealing with warmup
    - 数据增强: 时间偏移、高斯噪声、幅度缩放
    - 类别平衡: 加权采样
    - patience: 7 → 20

    Requirements: 2.1, 2.2, 2.3, 2.4
    """

    def __init__(
        self,
        model_name: str,
        n_folds: int = 5,
        max_epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 20,
        batch_size: int = 256,
        warmup_ratio: float = 0.1,
        min_lr: float = 1e-6,
        checkpoint_dir: str = "weights/rescue_pretrained",
        augmentation_config: Optional[AugmentationConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: 模型名称，"Chambon2018" 或 "TinySleepNet"
            n_folds: 交叉验证折数（Req 2.1）
            max_epochs: 最大训练轮数
            lr: 初始学习率
            patience: 早停耐心值
            batch_size: 批大小
            warmup_ratio: warmup 占总 epoch 的比例
            min_lr: cosine annealing 的最小学习率
            checkpoint_dir: 检查点保存目录（Req 2.4）
            augmentation_config: 数据增强配置（Req 2.3），None 则使用默认配置
            device: 计算设备，None 则自动检测
        """
        self.model_name = model_name
        self.n_folds = n_folds
        self.max_epochs = max_epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr
        self.checkpoint_dir = checkpoint_dir
        self.aug_config = augmentation_config or AugmentationConfig()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_fold(
        self,
        fold_idx: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> Dict:
        """训练单折，返回训练历史和最佳检查点路径。

        Req 2.1: 在每折的训练集上训练模型
        Req 2.2: 验证集上评估准确率
        Req 2.3: 使用 LR 调度、数据增强、类别平衡采样
        Req 2.4: 保存最佳模型检查点和训练日志

        Args:
            fold_idx: 折索引（0-based）
            train_dataset: 训练集 Dataset
            val_dataset: 验证集 Dataset

        Returns:
            包含训练历史、最佳准确率、检查点路径等信息的字典
        """
        logger.info(
            "开始训练 %s 第 %d 折 (max_epochs=%d, lr=%.1e, patience=%d)",
            self.model_name, fold_idx, self.max_epochs, self.lr, self.patience,
        )
        start_time = time.time()

        # 构建模型
        model = build_model(self.model_name).to(self.device)

        # 包装训练集以添加数据增强（Req 2.3）
        aug_train_dataset = AugmentedDataset(
            train_dataset, self.aug_config, training=True
        )
        # 验证集不做增强
        val_dataset_wrapped = AugmentedDataset(
            val_dataset, self.aug_config, training=False
        )

        # 类别平衡采样器（Req 2.3）
        sampler = create_class_balanced_sampler(train_dataset)

        train_loader = DataLoader(
            aug_train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset_wrapped,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Cosine annealing 学习率调度 + warmup（Req 2.3）
        warmup_epochs = max(1, int(self.max_epochs * self.warmup_ratio))
        scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            total_epochs=self.max_epochs,
            warmup_epochs=warmup_epochs,
            min_lr=self.min_lr,
        )

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        # 训练状态
        history: List[Dict] = []
        best_val_acc = -1.0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            # 更新学习率
            scheduler.step(epoch)
            current_lr = scheduler.get_last_lr()[0]

            # --- 训练阶段 ---
            train_loss = self._train_one_epoch(
                model, train_loader, optimizer, criterion
            )

            # --- 验证阶段 ---
            val_acc, val_loss, per_class_acc = self._validate(
                model, val_loader, criterion
            )

            # 记录历史（Req 2.4）
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "per_class_accuracy": per_class_acc,
                "lr": current_lr,
            }
            history.append(epoch_record)

            logger.info(
                "Fold %d Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, "
                "val_acc=%.4f, lr=%.2e",
                fold_idx, epoch, self.max_epochs,
                train_loss, val_loss, val_acc, current_lr,
            )

            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                logger.info(
                    "Fold %d 早停触发: %d epochs 无改善 (patience=%d), "
                    "最佳 val_acc=%.4f",
                    fold_idx, epochs_without_improvement,
                    self.patience, best_val_acc,
                )
                break

        # 恢复最佳模型权重
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # 保存检查点（Req 2.4）
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{self.model_name}_fold{fold_idx}_best.pt",
        )
        torch.save(
            {
                "model_state_dict": best_state_dict or model.state_dict(),
                "model_name": self.model_name,
                "fold_idx": fold_idx,
                "best_val_acc": best_val_acc,
                "total_epochs": len(history),
                "config": {
                    "max_epochs": self.max_epochs,
                    "lr": self.lr,
                    "patience": self.patience,
                    "batch_size": self.batch_size,
                    "warmup_ratio": self.warmup_ratio,
                    "augmentation": asdict(self.aug_config),
                },
            },
            checkpoint_path,
        )

        # 保存训练日志（Req 2.4）
        log_path = os.path.join(
            self.checkpoint_dir,
            f"{self.model_name}_fold{fold_idx}_history.json",
        )
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        logger.info(
            "Fold %d 训练完成: best_val_acc=%.4f, epochs=%d, 耗时=%.1fs",
            fold_idx, best_val_acc, len(history), elapsed,
        )

        return {
            "fold_idx": fold_idx,
            "best_val_accuracy": best_val_acc,
            "total_epochs": len(history),
            "early_stopped": epochs_without_improvement >= self.patience,
            "checkpoint_path": checkpoint_path,
            "log_path": log_path,
            "history": history,
            "training_time_seconds": elapsed,
        }

    def train_all_folds(
        self,
        fold_datasets: List[Tuple[Dataset, Dataset]],
    ) -> Dict:
        """训练所有折，返回汇总结果。

        Args:
            fold_datasets: 长度为 n_folds 的列表，
                每个元素为 (train_dataset, val_dataset) 元组

        Returns:
            包含所有折结果和汇总统计的字典
        """
        if len(fold_datasets) != self.n_folds:
            raise ValueError(
                f"fold_datasets 长度 ({len(fold_datasets)}) "
                f"与 n_folds ({self.n_folds}) 不一致"
            )

        all_results = []
        for fold_idx, (train_ds, val_ds) in enumerate(fold_datasets):
            result = self.train_fold(fold_idx, train_ds, val_ds)
            all_results.append(result)

        # 汇总统计
        val_accs = [r["best_val_accuracy"] for r in all_results]
        mean_acc = float(np.mean(val_accs))
        std_acc = float(np.std(val_accs))

        summary = {
            "model_name": self.model_name,
            "n_folds": self.n_folds,
            "fold_results": all_results,
            "mean_val_accuracy": mean_acc,
            "std_val_accuracy": std_acc,
            "min_val_accuracy": float(np.min(val_accs)),
            "max_val_accuracy": float(np.max(val_accs)),
            "target_met": mean_acc >= 0.65,  # Req 2.2: val_acc >= 65%
        }

        # 保存汇总结果
        summary_path = os.path.join(
            self.checkpoint_dir,
            f"{self.model_name}_summary.json",
        )
        # 保存时排除 history（太大）
        summary_for_save = {
            k: v for k, v in summary.items() if k != "fold_results"
        }
        summary_for_save["fold_val_accuracies"] = val_accs
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_for_save, f, ensure_ascii=False, indent=2)

        logger.info(
            "%s 全部 %d 折训练完成: mean_acc=%.4f ± %.4f, 目标达成=%s",
            self.model_name, self.n_folds, mean_acc, std_acc,
            summary["target_met"],
        )

        return summary

    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """执行一个训练 epoch。

        Returns:
            平均训练损失
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if len(batch) == 3:
                x, targets, _features = batch
            elif len(batch) == 2:
                x, targets = batch
            else:
                continue

            x = x.to(self.device)
            targets = targets.to(self.device)
            if isinstance(targets, torch.Tensor) and targets.dtype == torch.float32:
                targets = targets.long()

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float, Dict[int, float]]:
        """在验证集上评估模型。

        Returns:
            (val_accuracy, val_loss, per_class_accuracy) 元组
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        for batch in val_loader:
            if len(batch) == 3:
                x, targets, _features = batch
            elif len(batch) == 2:
                x, targets = batch
            else:
                continue

            x = x.to(self.device)
            targets = targets.to(self.device)
            if isinstance(targets, torch.Tensor) and targets.dtype == torch.float32:
                targets = targets.long()

            outputs = model(x)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

            preds = outputs.argmax(dim=1)
            # 仅保留有效标签（>= 0）
            valid_mask = targets >= 0
            all_preds.append(preds[valid_mask].cpu())
            all_targets.append(targets[valid_mask].cpu())

        val_loss = total_loss / max(num_batches, 1)

        if not all_preds:
            return 0.0, val_loss, {}

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if len(all_targets) == 0:
            return 0.0, val_loss, {}

        # 总体准确率
        val_acc = (all_preds == all_targets).float().mean().item()

        # 每类准确率
        per_class_acc = {}
        for cls in range(5):  # W, N1, N2, N3, REM
            mask = all_targets == cls
            if mask.any():
                per_class_acc[cls] = (
                    (all_preds[mask] == cls).float().mean().item()
                )
            else:
                per_class_acc[cls] = 0.0

        model.train()
        return val_acc, val_loss, per_class_acc
