"""
渐进式适应协议 (Progressive Adapter)

两阶段渐进式域适应：
- Phase 1 (BN适应): 冻结所有参数，仅用前向传播更新BatchNorm统计量（无标签）
- Phase 2 (FiLM微调): 冻结基础模型，仅训练FiLM和SeverityConditioner参数（有标签+早停）

Two-Pass Inference（解决AHI循环依赖，审稿人意见#1）：
- Pass 1: 使用基础模型（无FiLM条件化）进行粗略分期 → 估计AHI
- Pass 2: 使用估计的AHI进行FiLM条件化分期

数据泄漏说明（审稿人意见#2）：
  Phase 1 BN适应采用归纳式(inductive)设置：仅使用训练折(training fold)的
  未标注数据更新BN统计量，测试折数据不参与Phase 1。这确保了临床现实性——
  在实际部署中，BN统计量仅从已收集的历史数据中计算，不包含待预测的新患者数据。

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .ahi_estimator import AHIEstimator

logger = logging.getLogger(__name__)


class ProgressiveAdapter:
    """
    两阶段渐进式适应

    Phase 1 (BN适应): 冻结所有参数，仅用前向传播更新BatchNorm统计量
    Phase 2 (FiLM微调): 冻结基础模型，仅训练FiLM和SeverityConditioner参数

    模型需要是FiLM包装模型，提供 get_trainable_params() 方法。
    DataLoader 产出 (x, targets, patient_features)，其中 patient_features
    是包含 ahi, severity, age, sex, bmi 键的字典。
    """

    def __init__(
        self,
        model: nn.Module,
        conditioner: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-3,
        patience: int = 5,
        max_epochs: int = 50,
        bn_momentum: float = 0.01,  # 新增: BN momentum 参数
    ):
        """
        Args:
            model: FiLM包装后的模型（需有 get_trainable_params() 方法）
            conditioner: SeverityConditioner（已包含在model中，此处保留引用）
            loss_fn: SeverityAwareN1Loss，forward(inputs, targets, severity)
            lr: Phase 2 学习率
            patience: 早停耐心值（连续多少个epoch验证准确率不提升则停止）
            max_epochs: Phase 2 最大训练轮数
            bn_momentum: BN层的momentum参数(默认0.01,比PyTorch默认0.1更温和)
        """
        self.model = model
        self.conditioner = conditioner
        self.loss_fn = loss_fn
        self.lr = lr
        self.patience = patience
        self.max_epochs = max_epochs
        self.bn_momentum = bn_momentum
        self.device = next(model.parameters()).device

    def phase1_bn_adapt(self, unlabeled_loader: DataLoader) -> Dict:
        """
        Phase 1: BatchNorm统计量适应（无标签）

        将模型设为train模式（仅影响BN层的running_mean/running_var更新），
        冻结所有参数的梯度，对未标注数据执行前向传播。

        Req 4.1: 仅更新BN running_mean/running_var，冻结所有其他参数
        Req 4.2: 使用全部可用的未标注数据进行前向传播

        Args:
            unlabeled_loader: 未标注数据的DataLoader，
                产出 (x, targets, patient_features) 或 (x, patient_features)

        Returns:
            包含BN适应统计信息的字典
        """
        # 设置 BN momentum (更温和的统计量更新)
        original_momentums = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                original_momentums[name] = module.momentum
                module.momentum = self.bn_momentum
        
        logger.info(
            "Phase 1: 设置 BN momentum = %.4f (原始值通常为 0.1)",
            self.bn_momentum
        )
        
        # 冻结所有参数 (Req 4.1)
        for param in self.model.parameters():
            param.requires_grad = False

        # 设为train模式，使BN层更新running统计量
        self.model.train()

        num_batches = 0
        num_samples = 0

        with torch.no_grad():
            for batch in unlabeled_loader:
                # 支持两种DataLoader格式
                if len(batch) == 3:
                    x, _targets, patient_features = batch
                elif len(batch) == 2:
                    x, patient_features = batch
                else:
                    raise ValueError(
                        f"DataLoader batch应包含2或3个元素，实际得到{len(batch)}个"
                    )

                x = x.to(self.device)
                patient_features = {
                    k: v.to(self.device) for k, v in patient_features.items()
                }

                # 前向传播以更新BN统计量 (Req 4.2)
                self.model(x, patient_features)

                num_batches += 1
                num_samples += x.size(0)
        
        # 恢复原始 momentum
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if name in original_momentums:
                    module.momentum = original_momentums[name]

        logger.info(
            "Phase 1 BN适应完成: %d batches, %d samples",
            num_batches,
            num_samples,
        )

        return {
            "phase": 1,
            "num_batches": num_batches,
            "num_samples": num_samples,
            "bn_momentum_used": self.bn_momentum,
        }

    def phase2_film_finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """
        Phase 2: FiLM参数微调（有标签）

        冻结基础模型参数，仅优化FiLM层和SeverityConditioner参数。
        使用早停策略（patience）防止过拟合。

        Req 4.3: 仅更新FiLM和SeverityConditioner参数
        Req 4.4: 早停策略 (patience=5)
        Req 4.5: 记录每个epoch的训练损失、验证准确率和N1召回率

        Args:
            train_loader: 训练数据DataLoader，产出 (x, targets, patient_features)
            val_loader: 验证数据DataLoader，产出 (x, targets, patient_features)

        Returns:
            包含训练历史的字典
        """
        # 冻结所有参数，然后仅解冻FiLM+Conditioner (Req 4.3)
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_params = self.model.get_trainable_params()
        for param in trainable_params:
            param.requires_grad = True
        
        # 【关键修复】冻结所有BN层，防止统计量被破坏
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()  # 强制eval模式，不更新running stats
                module.track_running_stats = False  # 禁用统计量跟踪

        # 设置优化器（仅优化可训练参数）
        # 使用权重衰减防止过拟合
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=1e-4)

        # 训练历史 (Req 4.5)
        history: List[Dict] = []

        # 早停状态 (Req 4.4)
        best_val_accuracy = -1.0
        best_state_dict = None
        epochs_without_improvement = 0

        self.model.train()

        for epoch in range(self.max_epochs):
            # --- 训练阶段 ---
            train_loss = self._train_one_epoch(train_loader, optimizer)

            # --- 验证阶段 ---
            val_accuracy, n1_recall = self._validate(val_loader)

            # 记录历史 (Req 4.5)
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "n1_recall": n1_recall,
            }
            history.append(epoch_record)

            logger.info(
                "Phase 2 Epoch %d: loss=%.4f, val_acc=%.4f, n1_recall=%.4f",
                epoch,
                train_loss,
                val_accuracy,
                n1_recall,
            )

            # 早停检查 (Req 4.4)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                logger.info(
                    "早停触发: %d epochs无改善 (patience=%d), "
                    "最佳val_accuracy=%.4f at epoch %d",
                    epochs_without_improvement,
                    self.patience,
                    best_val_accuracy,
                    epoch - epochs_without_improvement,
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
        }

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """执行一个训练epoch。

        Args:
            train_loader: 训练数据DataLoader
            optimizer: 优化器

        Returns:
            平均训练损失
        """
        # 设置为训练模式，但BN层保持eval（已在phase2_film_finetune中设置）
        self.model.train()
        
        # 确保BN层保持eval模式
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
        
        total_loss = 0.0
        num_batches = 0

        for x, targets, patient_features in train_loader:
            x = x.to(self.device)
            targets = targets.to(self.device)
            patient_features = {
                k: v.to(self.device) for k, v in patient_features.items()
            }

            optimizer.zero_grad()
            outputs = self.model(x, patient_features)
            loss = self.loss_fn(outputs, targets, patient_features["severity"])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> tuple:
        """在验证集上评估模型。

        Args:
            val_loader: 验证数据DataLoader

        Returns:
            (val_accuracy, n1_recall) 元组
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        for x, targets, patient_features in val_loader:
            x = x.to(self.device)
            targets = targets.to(self.device)
            patient_features = {
                k: v.to(self.device) for k, v in patient_features.items()
            }

            outputs = self.model(x, patient_features)
            preds = outputs.argmax(dim=1)

            # 仅保留有效标签
            valid_mask = targets >= 0
            all_preds.append(preds[valid_mask])
            all_targets.append(targets[valid_mask])

        if not all_preds:
            return 0.0, 0.0

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if len(all_targets) == 0:
            return 0.0, 0.0

        # 总体准确率
        val_accuracy = (all_preds == all_targets).float().mean().item()

        # N1召回率 (N1 class index = 1)
        n1_mask = all_targets == 1
        if n1_mask.any():
            n1_recall = (all_preds[n1_mask] == 1).float().mean().item()
        else:
            n1_recall = 0.0

        # 恢复训练模式
        self.model.train()

        return val_accuracy, n1_recall

    # ================================================================
    # Two-Pass Inference（审稿人意见 #1: 解决AHI循环依赖）
    # ================================================================

    @torch.no_grad()
    def two_pass_inference(
        self,
        data_loader: DataLoader,
        base_model: nn.Module,
        ahi_estimator: AHIEstimator,
        patient_features_template: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict:
        """
        Two-Pass Inference: 解决AHI循环依赖问题。

        在临床部署场景中，AHI在分期完成前不可用。Two-Pass方案：
        - Pass 1: 使用基础模型（无FiLM条件化）进行粗略分期，
                   从分期结果通过AHIEstimator估计AHI
        - Pass 2: 使用估计的AHI替换patient_features中的ahi字段，
                   进行FiLM条件化分期

        Args:
            data_loader: 数据DataLoader，产出 (x, targets, patient_features)
            base_model: 基础模型（无FiLM包装），用于Pass 1粗略分期
            ahi_estimator: 已拟合的AHI估计器
            patient_features_template: 可选的患者特征模板（用于提供age/sex/bmi，
                当data_loader中已包含这些字段时可为None）

        Returns:
            包含两次推理结果的字典:
                pass1_predictions: Pass 1的粗略分期预测
                pass2_predictions: Pass 2的FiLM条件化分期预测
                estimated_ahi: 从Pass 1估计的AHI值
                estimated_severity: 从估计AHI推导的严重程度类别
        """
        base_model.eval()
        self.model.eval()

        # --- Pass 1: 基础模型粗略分期 ---
        pass1_all_preds = []
        pass1_all_targets = []
        all_patient_features = []

        for x, targets, patient_features in data_loader:
            x = x.to(self.device)
            targets = targets.to(self.device)

            # Pass 1: 基础模型前向传播（不使用patient_features）
            # 基础模型可能不接受patient_features参数
            try:
                outputs = base_model(x)
            except TypeError:
                # 如果基础模型也需要patient_features（如已包装模型）
                pf_device = {
                    k: v.to(self.device) for k, v in patient_features.items()
                }
                outputs = base_model(x, pf_device)

            preds = outputs.argmax(dim=1)
            pass1_all_preds.append(preds.cpu())
            pass1_all_targets.append(targets.cpu())
            all_patient_features.append(patient_features)

        pass1_preds = torch.cat(pass1_all_preds).numpy()

        # --- 从Pass 1结果估计AHI ---
        estimated_ahi = ahi_estimator.estimate(pass1_preds)
        estimated_severity = ahi_estimator.ahi_to_severity(estimated_ahi)

        # --- Pass 2: 使用估计AHI进行FiLM条件化分期 ---
        pass2_all_preds = []

        batch_idx = 0
        for x, targets, patient_features in data_loader:
            x = x.to(self.device)
            batch_size = x.size(0)

            # 用估计的AHI和severity替换原始值
            modified_features = {}
            for k, v in patient_features.items():
                modified_features[k] = v.to(self.device)

            modified_features["ahi"] = torch.full(
                (batch_size,), estimated_ahi, device=self.device
            )
            modified_features["severity"] = torch.full(
                (batch_size,), estimated_severity,
                dtype=torch.long, device=self.device,
            )

            outputs = self.model(x, modified_features)
            preds = outputs.argmax(dim=1)
            pass2_all_preds.append(preds.cpu())
            batch_idx += 1

        pass2_preds = torch.cat(pass2_all_preds).numpy()
        all_targets = torch.cat(pass1_all_targets).numpy()

        return {
            "pass1_predictions": pass1_preds,
            "pass2_predictions": pass2_preds,
            "targets": all_targets,
            "estimated_ahi": float(estimated_ahi),
            "estimated_severity": int(estimated_severity),
        }
