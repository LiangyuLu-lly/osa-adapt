"""
基线适应方法 (Baseline Adaptation Methods)

实现6种基线适应方法用于与OSA-Adapt进行公平对比：
1. NoAdaptation: 不做任何适应，直接使用预训练模型
2. FullFinetune: 解冻所有参数进行微调
3. LastLayerFinetune: 仅微调最后一层（分类器）
4. LoRAAdaptation: 使用LoRA（rank=4）进行低秩适应
5. StandardFiLM: 标准FiLM（无严重程度条件化，使用固定/可学习条件向量）
6. BNOnlyAdaptation: 仅更新BatchNorm统计量

所有方法共享相同的训练/评估接口 (BaseAdaptationMethod)。

Requirements: 7.1, 7.2, 7.3
"""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .film_adapter import FiLMAdapter

logger = logging.getLogger(__name__)


class BaseAdaptationMethod(ABC):
    """基线适应方法的基类。

    所有基线方法共享相同的 adapt() 接口 (Req 7.3)。
    adapt() 接收模型、训练/验证DataLoader，执行适应并返回结果字典。
    """

    @abstractmethod
    def adapt(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        """执行适应并返回结果字典。

        Args:
            model: 待适应的预训练模型
            train_loader: 训练数据DataLoader，产出 (x, targets, patient_features)
            val_loader: 验证数据DataLoader，产出 (x, targets, patient_features)
            **kwargs: 额外参数

        Returns:
            结果字典，至少包含:
                method: str - 方法名称
                history: list - 训练历史（每epoch记录）
                total_epochs: int - 总训练轮数
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """方法名称"""
        raise NotImplementedError


def _get_device(model: nn.Module) -> torch.device:
    """获取模型所在设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """在验证集上评估模型，返回准确率和N1召回率。

    Args:
        model: 模型
        val_loader: 验证DataLoader，产出 (x, targets, patient_features)
        device: 计算设备

    Returns:
        包含 val_accuracy 和 n1_recall 的字典
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, targets, patient_features in val_loader:
            x = x.to(device)
            targets = targets.to(device)
            patient_features = {
                k: v.to(device) for k, v in patient_features.items()
            }

            outputs = model(x, patient_features)
            preds = outputs.argmax(dim=1)

            valid_mask = targets >= 0
            all_preds.append(preds[valid_mask])
            all_targets.append(targets[valid_mask])

    if not all_preds:
        return {"val_accuracy": 0.0, "n1_recall": 0.0}

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    if len(all_targets) == 0:
        return {"val_accuracy": 0.0, "n1_recall": 0.0}

    val_accuracy = (all_preds == all_targets).float().mean().item()

    n1_mask = all_targets == 1
    n1_recall = (
        (all_preds[n1_mask] == 1).float().mean().item() if n1_mask.any() else 0.0
    )

    return {"val_accuracy": val_accuracy, "n1_recall": n1_recall}


def _train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 5,
) -> Dict:
    """通用训练循环，带早停。

    Args:
        model: 模型
        train_loader: 训练DataLoader
        val_loader: 验证DataLoader
        optimizer: 优化器
        loss_fn: 损失函数，接受 (outputs, targets) 或 (outputs, targets, severity)
        device: 计算设备
        max_epochs: 最大训练轮数
        patience: 早停耐心值

    Returns:
        包含 history, total_epochs, best_val_accuracy 的字典
    """
    history: List[Dict] = []
    best_val_accuracy = -1.0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # 训练
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x, targets, patient_features in train_loader:
            x = x.to(device)
            targets = targets.to(device)
            patient_features = {
                k: v.to(device) for k, v in patient_features.items()
            }

            optimizer.zero_grad()
            outputs = model(x, patient_features)

            # 标准交叉熵（基线方法不使用严重程度感知损失）
            valid_mask = targets >= 0
            if valid_mask.any():
                loss = loss_fn(outputs[valid_mask], targets[valid_mask])
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # 验证
        metrics = _evaluate(model, val_loader, device)
        val_accuracy = metrics["val_accuracy"]
        n1_recall = metrics["n1_recall"]

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
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                "早停触发: %d epochs无改善 (patience=%d)",
                epochs_without_improvement, patience,
            )
            break

    # 恢复最佳模型
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "history": history,
        "total_epochs": len(history),
        "best_val_accuracy": best_val_accuracy,
        "early_stopped": epochs_without_improvement >= patience,
    }


class NoAdaptation(BaseAdaptationMethod):
    """无适应基线：直接使用预训练模型进行推理，不做任何修改。"""

    def adapt(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        device = _get_device(model)
        metrics = _evaluate(model, val_loader, device)
        return {
            "method": self.name,
            "history": [],
            "total_epochs": 0,
            "best_val_accuracy": metrics["val_accuracy"],
            "n1_recall": metrics["n1_recall"],
        }

    @property
    def name(self) -> str:
        return "no_adaptation"


class FullFinetune(BaseAdaptationMethod):
    """全参数微调：解冻所有参数进行端到端微调。"""

    def __init__(
        self,
        lr: float = 1e-4,
        max_epochs: int = 50,
        patience: int = 5,
    ):
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience

    def adapt(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        device = _get_device(model)

        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        result = _train_with_early_stopping(
            model, train_loader, val_loader, optimizer, loss_fn,
            device, self.max_epochs, self.patience,
        )
        result["method"] = self.name
        return result

    @property
    def name(self) -> str:
        return "full_finetune"


class LastLayerFinetune(BaseAdaptationMethod):
    """最后一层微调：仅微调分类器（最后一层），冻结其余参数。"""

    def __init__(
        self,
        lr: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 5,
    ):
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience

    def adapt(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        device = _get_device(model)

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 仅解冻最后一层（分类器）
        classifier = self._find_classifier(model)
        for param in classifier.parameters():
            param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        result = _train_with_early_stopping(
            model, train_loader, val_loader, optimizer, loss_fn,
            device, self.max_epochs, self.patience,
        )
        result["method"] = self.name
        return result

    @staticmethod
    def _find_classifier(model: nn.Module) -> nn.Module:
        """查找模型的分类器层。

        按优先级查找: classifier, fc, head, output_layer 属性。
        如果都没有，返回模型最后一个子模块。
        """
        for attr_name in ("classifier", "fc", "head", "output_layer"):
            classifier = getattr(model, attr_name, None)
            if classifier is not None and isinstance(classifier, nn.Module):
                return classifier

        # 回退：返回最后一个子模块
        children = list(model.children())
        if children:
            return children[-1]

        raise ValueError("无法找到模型的分类器层")

    @property
    def name(self) -> str:
        return "last_layer_finetune"


class LoRALayer(nn.Module):
    """LoRA低秩适应层。

    对原始权重矩阵W添加低秩分解 ΔW = A @ B，
    其中 A: [out, rank], B: [rank, in]。

    前向传播: y = W @ x + (A @ B) @ x

    Req 7.1: 使用rank=4的低秩分解

    注意：original_layer 不注册为子模块，避免替换后产生循环引用。
    原始层的权重和偏置直接复制为frozen buffer。
    """

    def __init__(self, original_layer: nn.Linear, rank: int = 4):
        """
        Args:
            original_layer: 原始线性层
            rank: 低秩分解的秩
        """
        super().__init__()
        self.rank = rank
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # 将原始权重/偏置复制为frozen buffer（不注册为子模块）
        self.register_buffer("weight", original_layer.weight.data.clone())
        self.register_buffer(
            "bias",
            original_layer.bias.data.clone() if original_layer.bias is not None else None,
        )

        # LoRA参数: A初始化为随机高斯，B初始化为零（确保初始ΔW=0）
        self.lora_A = nn.Parameter(torch.randn(self.out_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 原始输出 + LoRA增量"""
        original_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = x @ self.lora_B.T @ self.lora_A.T
        return original_out + lora_out


class LoRAAdaptation(BaseAdaptationMethod):
    """LoRA适应：使用rank=4的低秩分解适应线性层。

    Req 7.1: 使用rank=4的低秩分解
    Req 7.3: 仅适应最后的全连接层（或所有线性层中的最后几个）
    """

    def __init__(
        self,
        rank: int = 4,
        lr: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 5,
    ):
        self.rank = rank
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience

    def adapt(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        device = _get_device(model)

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 对最后的线性层应用LoRA
        self._apply_lora(model)

        # 确保新创建的 LoRA 层在正确的 device 上
        model = model.to(device)

        # 仅优化LoRA参数
        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(lora_params, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        result = _train_with_early_stopping(
            model, train_loader, val_loader, optimizer, loss_fn,
            device, self.max_epochs, self.patience,
        )
        result["method"] = self.name
        return result

    def _apply_lora(self, model: nn.Module) -> None:
        """对模型中的线性层应用LoRA。

        查找分类器中的线性层，或模型最后的线性层。
        """
        # 优先查找分类器属性名
        classifier_attr = None
        for attr_name in ("classifier", "fc", "head", "output_layer"):
            candidate = getattr(model, attr_name, None)
            if candidate is not None and isinstance(candidate, nn.Module):
                classifier_attr = attr_name
                break

        if classifier_attr is not None:
            classifier = getattr(model, classifier_attr)
            if isinstance(classifier, nn.Linear):
                # 分类器本身就是线性层，直接替换
                lora_layer = LoRALayer(classifier, rank=self.rank)
                setattr(model, classifier_attr, lora_layer)
                return
            # 分类器是复合模块，替换其中的线性层
            applied = False
            for name, module in list(classifier.named_modules()):
                if isinstance(module, nn.Linear) and name != "":
                    lora_layer = LoRALayer(module, rank=self.rank)
                    self._replace_module(classifier, name, lora_layer)
                    applied = True
            if applied:
                return

        # 回退：对模型中最后一个线性层应用LoRA
        last_linear_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name != "":
                last_linear_name = name
        if last_linear_name is not None:
            module = dict(model.named_modules())[last_linear_name]
            lora_layer = LoRALayer(module, rank=self.rank)
            self._replace_module(model, last_linear_name, lora_layer)

    @staticmethod
    def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
        """替换模块中的子模块。支持嵌套名称（如 'layer.0.fc'）。"""
        parts = name.split(".")
        current = parent
        for part in parts[:-1]:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)

        last_part = parts[-1]
        if last_part.isdigit():
            current[int(last_part)] = new_module
        else:
            setattr(current, last_part, new_module)

    @property
    def name(self) -> str:
        return "lora_adaptation"


class StandardFiLM(BaseAdaptationMethod):
    """标准FiLM基线：使用FiLM适配器但无严重程度条件化。

    与OSA-Adapt的区别：使用固定/可学习的条件向量，
    而非由患者临床特征生成的条件向量。
    所有样本共享同一个可学习条件向量。
    """

    def __init__(
        self,
        condition_dim: int = 64,
        lr: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 5,
    ):
        self.condition_dim = condition_dim
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience

    def adapt(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        device = _get_device(model)

        # 在模型中插入FiLM层（使用固定可学习条件向量）
        wrapped = StandardFiLMWrapper(model, self.condition_dim).to(device)

        # 仅优化FiLM参数和可学习条件向量
        trainable = list(wrapped.get_trainable_params())
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        result = _train_with_early_stopping(
            wrapped, train_loader, val_loader, optimizer, loss_fn,
            device, self.max_epochs, self.patience,
        )
        result["method"] = self.name
        result["adapted_model"] = wrapped
        return result

    @property
    def name(self) -> str:
        return "standard_film"


class StandardFiLMWrapper(nn.Module):
    """标准FiLM包装器：使用可学习的固定条件向量。

    与 FiLMWrappedChambon 不同，此包装器不使用
    SeverityConditioner，而是使用一个可学习的条件向量
    （所有样本共享同一个条件向量）。
    """

    def __init__(self, base_model: nn.Module, condition_dim: int = 64):
        super().__init__()
        self.base_model = base_model
        self.condition_dim = condition_dim
        self._has_conv_blocks = False
        self._has_cnn_lstm = False

        # 可学习的固定条件向量（初始化为零，使FiLM初始为恒等映射）
        self.learned_condition = nn.Parameter(
            torch.zeros(condition_dim)
        )

        # 查找卷积块并为每个块创建FiLM层
        self.film_layers = nn.ModuleList()
        conv_blocks = getattr(base_model, "conv_blocks", None)

        if conv_blocks is not None:
            self._has_conv_blocks = True
            for block in conv_blocks:
                out_ch = self._detect_out_channels(block)
                self.film_layers.append(
                    FiLMAdapter(feature_dim=out_ch, condition_dim=condition_dim)
                )
        elif hasattr(base_model, "cnn") and hasattr(base_model, "lstm"):
            # TinySleepNet 风格: CNN + LSTM + classifier
            self._has_cnn_lstm = True
            cnn_out_ch = self._detect_cnn_out_channels(base_model.cnn)
            self.film_layers.append(
                FiLMAdapter(feature_dim=cnn_out_ch, condition_dim=condition_dim)
            )
        else:
            # 通用回退：在分类器前插入单个FiLM层
            last_dim = self._detect_last_feature_dim(base_model)
            self.film_layers.append(
                FiLMAdapter(feature_dim=last_dim, condition_dim=condition_dim)
            )

        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False

    @staticmethod
    def _detect_out_channels(block: nn.Module) -> int:
        """检测卷积块的输出通道数"""
        out_ch = None
        for module in block.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                out_ch = module.out_channels
        return out_ch if out_ch is not None else 128

    @staticmethod
    def _detect_cnn_out_channels(cnn: nn.Module) -> int:
        """检测CNN模块的最后一个卷积层输出通道数"""
        out_ch = None
        for module in cnn.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                out_ch = module.out_channels
        return out_ch if out_ch is not None else 128

    @staticmethod
    def _detect_last_feature_dim(model: nn.Module) -> int:
        """检测模型最后特征维度"""
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                return module.in_features
        return 128

    def forward(
        self, x: torch.Tensor, patient_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """前向传播：使用固定可学习条件向量进行FiLM调制。

        Args:
            x: 输入张量
            patient_features: 患者特征字典（此基线中不使用）
        Returns:
            分类logits
        """
        batch_size = x.size(0)
        # 扩展条件向量到batch维度
        condition = self.learned_condition.unsqueeze(0).expand(
            batch_size, -1
        )

        if self._has_conv_blocks:
            for conv_block, film in zip(self.base_model.conv_blocks, self.film_layers):
                x = conv_block(x)
                x = film(x, condition)
            x = x.flatten(1)
            # Dropout（如果模型有的话）
            if hasattr(self.base_model, '_dropout'):
                x = self.base_model._dropout(x)
            x = self.base_model.classifier(x)
        elif self._has_cnn_lstm:
            # TinySleepNet: CNN → FiLM → flatten → LSTM → classifier
            cnn_out = self.base_model.cnn(x)
            cnn_out = self.film_layers[0](cnn_out, condition)
            # 如果是 3D，需要 flatten 后再送入 LSTM
            if cnn_out.dim() == 3:
                cnn_out = cnn_out.flatten(1)  # [B, C*T]
            if cnn_out.dim() == 2:
                cnn_out = cnn_out.unsqueeze(1)  # [B, 1, feat]
            lstm_out, _ = self.base_model.lstm(cnn_out)
            if lstm_out.dim() == 3:
                lstm_out = lstm_out[:, -1, :]
            # Dropout（如果模型有的话）
            if hasattr(self.base_model, '_rnn_dropout'):
                lstm_out = self.base_model._rnn_dropout(lstm_out)
            x = self.base_model.classifier(lstm_out)
        else:
            # 通用回退：提取分类器前的特征，做FiLM调制，再分类
            classifier = getattr(self.base_model, "classifier", None)
            if classifier is not None:
                features = []
                def hook_fn(module, inp, out):
                    features.append(inp[0])
                handle = classifier.register_forward_pre_hook(hook_fn)
                with torch.no_grad():
                    self.base_model(x, patient_features)
                handle.remove()
                if features:
                    feat = features[0].detach()
                    feat = self.film_layers[0](feat, condition)
                    x = classifier(feat)
                else:
                    with torch.no_grad():
                        x = self.base_model(x, patient_features)
            else:
                with torch.no_grad():
                    x = self.base_model(x, patient_features)

        return x

    def get_trainable_params(self) -> List[nn.Parameter]:
        """返回可训练参数：FiLM层 + 可学习条件向量"""
        params = list(self.film_layers.parameters())
        params.append(self.learned_condition)
        return params


class BNOnlyAdaptation(BaseAdaptationMethod):
    """仅BatchNorm适应：仅更新BN层的running统计量。

    类似于ProgressiveAdapter的Phase 1，但作为独立的基线方法。
    不进行任何参数训练，仅通过前向传播更新BN统计量。
    """

    def adapt(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        device = _get_device(model)

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 设为train模式以更新BN统计量
        model.train()

        num_batches = 0
        num_samples = 0

        with torch.no_grad():
            for batch in train_loader:
                x, targets, patient_features = batch
                x = x.to(device)
                patient_features = {
                    k: v.to(device) for k, v in patient_features.items()
                }

                model(x, patient_features)
                num_batches += 1
                num_samples += x.size(0)

        logger.info(
            "BN适应完成: %d batches, %d samples",
            num_batches, num_samples,
        )

        # 评估
        metrics = _evaluate(model, val_loader, device)

        return {
            "method": self.name,
            "history": [],
            "total_epochs": 0,
            "best_val_accuracy": metrics["val_accuracy"],
            "n1_recall": metrics["n1_recall"],
            "bn_batches": num_batches,
            "bn_samples": num_samples,
        }

    @property
    def name(self) -> str:
        return "bn_only_adaptation"


# 基线方法注册表
BASELINE_METHODS = {
    "no_adaptation": NoAdaptation,
    "full_finetune": FullFinetune,
    "last_layer_finetune": LastLayerFinetune,
    "lora_adaptation": LoRAAdaptation,
    "standard_film": StandardFiLM,
    "bn_only_adaptation": BNOnlyAdaptation,
}


def create_baseline(method_name: str, **kwargs) -> BaseAdaptationMethod:
    """工厂函数：根据名称创建基线适应方法。

    Args:
        method_name: 方法名称，可选值见 BASELINE_METHODS
        **kwargs: 传递给方法构造函数的参数

    Returns:
        BaseAdaptationMethod 实例

    Raises:
        ValueError: 未知的方法名称
    """
    if method_name not in BASELINE_METHODS:
        available = ", ".join(BASELINE_METHODS.keys())
        raise ValueError(
            f"未知的基线方法: '{method_name}'。可选: {available}"
        )
    return BASELINE_METHODS[method_name](**kwargs)


# ================================================================
# 分布对齐基线（审稿人意见 #4: CORAL / MMD）
# ================================================================


class CORALAdaptation(BaseAdaptationMethod):
    """CORAL (CORrelation ALignment) 域适应基线。

    通过对齐源域和目标域特征的二阶统计量（协方差矩阵）进行适应。
    不使用任何严重程度信息——这是关键对比点：
    展示"盲目"分布对齐无法解决严重程度相关的N1检测问题。

    CORAL Loss = ||C_s - C_t||²_F / (4 * d²)
    其中 C_s, C_t 分别是源域和目标域特征的协方差矩阵。
    """

    def __init__(
        self,
        coral_weight: float = 1.0,
        lr: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 5,
    ):
        self.coral_weight = coral_weight
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience

    @staticmethod
    def _coral_loss(
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """计算CORAL损失。

        Args:
            source_features: 源域特征 [B_s, D]
            target_features: 目标域特征 [B_t, D]

        Returns:
            CORAL损失标量
        """
        d = source_features.size(1)

        # 中心化
        src_centered = source_features - source_features.mean(dim=0, keepdim=True)
        tgt_centered = target_features - target_features.mean(dim=0, keepdim=True)

        # 协方差矩阵
        n_s = max(source_features.size(0) - 1, 1)
        n_t = max(target_features.size(0) - 1, 1)
        cov_src = (src_centered.T @ src_centered) / n_s
        cov_tgt = (tgt_centered.T @ tgt_centered) / n_t

        # Frobenius范数
        loss = torch.sum((cov_src - cov_tgt) ** 2) / (4 * d * d)
        return loss

    def adapt(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        device = _get_device(model)

        # 解冻最后一层用于适应
        for param in model.parameters():
            param.requires_grad = False

        classifier = LastLayerFinetune._find_classifier(model)
        for param in classifier.parameters():
            param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        ce_loss_fn = nn.CrossEntropyLoss()

        history: List[Dict] = []
        best_val_accuracy = -1.0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for x, targets, patient_features in train_loader:
                x = x.to(device)
                targets = targets.to(device)
                pf = {k: v.to(device) for k, v in patient_features.items()}

                optimizer.zero_grad()
                outputs = model(x, pf)

                valid_mask = targets >= 0
                if valid_mask.any():
                    ce = ce_loss_fn(outputs[valid_mask], targets[valid_mask])
                else:
                    ce = torch.tensor(0.0, device=device, requires_grad=True)

                # CORAL: 对齐不同严重程度组的特征分布
                # 使用输出logits作为特征的代理
                severity = pf.get("severity")
                coral = torch.tensor(0.0, device=device)
                if severity is not None and outputs.size(0) > 2:
                    mild_mask = severity <= 1
                    severe_mask = severity >= 2
                    if mild_mask.sum() > 1 and severe_mask.sum() > 1:
                        coral = self._coral_loss(
                            outputs[mild_mask], outputs[severe_mask]
                        )

                loss = ce + self.coral_weight * coral
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            metrics = _evaluate(model, val_loader, device)
            val_accuracy = metrics["val_accuracy"]

            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_accuracy": val_accuracy,
                "n1_recall": metrics["n1_recall"],
            })

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        return {
            "method": self.name,
            "history": history,
            "total_epochs": len(history),
            "best_val_accuracy": best_val_accuracy,
        }

    @property
    def name(self) -> str:
        return "coral_adaptation"


class MMDAdaptation(BaseAdaptationMethod):
    """MMD (Maximum Mean Discrepancy) 域适应基线。

    通过最小化源域和目标域特征在RKHS中的MMD距离进行适应。
    使用高斯核（RBF kernel）。与CORAL类似，不使用严重程度信息。

    MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    其中 k 是高斯核 k(x,y) = exp(-||x-y||² / (2σ²))
    """

    def __init__(
        self,
        mmd_weight: float = 1.0,
        kernel_bandwidth: float = 1.0,
        lr: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 5,
    ):
        self.mmd_weight = mmd_weight
        self.kernel_bandwidth = kernel_bandwidth
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience

    def _gaussian_kernel(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """高斯核矩阵 k(x_i, y_j) = exp(-||x_i - y_j||² / (2σ²))"""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)

        x = x.unsqueeze(1)  # [N, 1, D]
        y = y.unsqueeze(0)  # [1, M, D]

        kernel = torch.exp(
            -torch.sum((x - y) ** 2, dim=-1)
            / (2 * self.kernel_bandwidth ** 2)
        )
        return kernel

    def _mmd_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """计算MMD²损失。"""
        k_ss = self._gaussian_kernel(source, source)
        k_tt = self._gaussian_kernel(target, target)
        k_st = self._gaussian_kernel(source, target)

        n_s = source.size(0)
        n_t = target.size(0)

        mmd = (
            k_ss.sum() / max(n_s * n_s, 1)
            + k_tt.sum() / max(n_t * n_t, 1)
            - 2 * k_st.sum() / max(n_s * n_t, 1)
        )
        return mmd

    def adapt(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        device = _get_device(model)

        for param in model.parameters():
            param.requires_grad = False

        classifier = LastLayerFinetune._find_classifier(model)
        for param in classifier.parameters():
            param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        ce_loss_fn = nn.CrossEntropyLoss()

        history: List[Dict] = []
        best_val_accuracy = -1.0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for x, targets, patient_features in train_loader:
                x = x.to(device)
                targets = targets.to(device)
                pf = {k: v.to(device) for k, v in patient_features.items()}

                optimizer.zero_grad()
                outputs = model(x, pf)

                valid_mask = targets >= 0
                if valid_mask.any():
                    ce = ce_loss_fn(outputs[valid_mask], targets[valid_mask])
                else:
                    ce = torch.tensor(0.0, device=device, requires_grad=True)

                severity = pf.get("severity")
                mmd = torch.tensor(0.0, device=device)
                if severity is not None and outputs.size(0) > 2:
                    mild_mask = severity <= 1
                    severe_mask = severity >= 2
                    if mild_mask.sum() > 1 and severe_mask.sum() > 1:
                        mmd = self._mmd_loss(
                            outputs[mild_mask], outputs[severe_mask]
                        )

                loss = ce + self.mmd_weight * mmd
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            metrics = _evaluate(model, val_loader, device)
            val_accuracy = metrics["val_accuracy"]

            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_accuracy": val_accuracy,
                "n1_recall": metrics["n1_recall"],
            })

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        return {
            "method": self.name,
            "history": history,
            "total_epochs": len(history),
            "best_val_accuracy": best_val_accuracy,
        }

    @property
    def name(self) -> str:
        return "mmd_adaptation"


# 更新注册表，加入CORAL和MMD
BASELINE_METHODS["coral_adaptation"] = CORALAdaptation
BASELINE_METHODS["mmd_adaptation"] = MMDAdaptation
