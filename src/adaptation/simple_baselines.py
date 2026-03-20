"""
简化的基线方法 - 用于快速实验验证
"""

import copy
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class FullFineTuneBaseline:
    """全参数微调基线 - 简化版本用于快速验证"""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-3,
        patience: int = 5,
        max_epochs: int = 30,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.patience = patience
        self.max_epochs = max_epochs
        self.device = next(model.parameters()).device
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """训练模型"""
        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 【关键修复】冻结BN层，防止统计量被破坏
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()  # 强制eval模式
                module.track_running_stats = False  # 禁用统计量跟踪
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        history: List[Dict] = []
        best_val_accuracy = -1.0
        best_state_dict = None
        epochs_without_improvement = 0
        
        for epoch in range(self.max_epochs):
            # 训练
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
                
                valid_mask = targets >= 0
                if valid_mask.any():
                    loss = self.loss_fn(
                        outputs[valid_mask],
                        targets[valid_mask],
                        patient_features["severity"][valid_mask]
                    )
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            # 验证
            val_accuracy, n1_recall = self._validate(val_loader)
            
            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_accuracy": val_accuracy,
                "n1_recall": n1_recall,
            })
            
            logger.info(
                "Epoch %d: loss=%.4f, val_acc=%.4f, n1_recall=%.4f",
                epoch, avg_loss, val_accuracy, n1_recall,
            )
            
            # 早停
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= self.patience:
                logger.info(
                    "早停触发: %d epochs无改善 (patience=%d)",
                    epochs_without_improvement, self.patience,
                )
                break
        
        # 恢复最佳模型
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        
        return {
            "history": history,
            "total_epochs": len(history),
            "best_val_accuracy": best_val_accuracy,
            "early_stopped": epochs_without_improvement >= self.patience,
        }
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> tuple:
        """验证"""
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
            
            valid_mask = targets >= 0
            all_preds.append(preds[valid_mask])
            all_targets.append(targets[valid_mask])
        
        if not all_preds:
            return 0.0, 0.0
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        if len(all_targets) == 0:
            return 0.0, 0.0
        
        val_accuracy = (all_preds == all_targets).float().mean().item()
        
        n1_mask = all_targets == 1
        if n1_mask.any():
            n1_recall = (all_preds[n1_mask] == 1).float().mean().item()
        else:
            n1_recall = 0.0
        
        self.model.train()
        
        return val_accuracy, n1_recall
