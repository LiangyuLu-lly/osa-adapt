"""
模型架构构建模块：创建与 FiLM 包装器兼容的 Chambon2018 和 TinySleepNet 模型。

优先使用 PhysioEx 原生架构（可加载预训练权重），
回退到自定义轻量实现（随机初始化）。

两个模型的 forward 签名均为 forward(x, patient_features=None)，
其中 x 的形状为 [batch, n_channels, sequence_length]。

属性要求（FiLM 包装器兼容）：
- Chambon2018: conv_blocks (list-like) + classifier (nn.Linear)
- TinySleepNet: cnn (callable) + lstm (nn.LSTM) + classifier (nn.Linear)
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PhysioEx 包装器
# ---------------------------------------------------------------------------

def _try_import_physioex():
    """尝试导入 PhysioEx 模块，返回 (chambon_Net, tiny_Net) 或 None。"""
    try:
        from physioex.train.networks.chambon2018 import Net as CNet
        from physioex.train.networks.tinysleepnet import Net as TNet
        return CNet, TNet
    except ImportError:
        return None, None


class _Squeeze2Dto3D(nn.Module):
    """将 4D [B, C, 1, T] 压缩为 3D [B, C, T]。"""
    def forward(self, x):
        if x.dim() == 4 and x.shape[2] == 1:
            return x.squeeze(2)
        return x


class _Unsqueeze3Dto4D(nn.Module):
    """将 3D [B, C, T] 扩展为 4D [B, C, 1, T]（Conv2d 需要）。"""
    def forward(self, x):
        if x.dim() == 3:
            return x.unsqueeze(2)
        return x


class PhysioExChambon2018Wrapper(nn.Module):
    """包装 PhysioEx 的 Chambon2018 Net，暴露 FiLM 兼容接口。

    PhysioEx 的 Chambon2018 内部使用 Conv2d (braindecode)，
    输出 4D [B, C, 1, T]。本包装器在每个 conv_block 末尾加入
    squeeze 操作，使输出为 3D [B, C, T]，兼容 FiLMAdapter。

    暴露属性：
    - conv_blocks: nn.ModuleList，每块输出 3D [B, C, T]
    - classifier: nn.Linear
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 5):
        super().__init__()
        CNet, _ = _try_import_physioex()
        if CNet is None:
            raise ImportError("PhysioEx not available")

        mc = {
            'in_channels': n_channels,
            'sf': 100,
            'n_classes': n_classes,
            'sequence_length': 1,
            'n_times': 3000,
        }
        self._pex_net = CNet(mc)

        # 拆分 feature_extractor 为块，每块末尾加 squeeze
        feat_ext = self._pex_net.epoch_encoder.feature_extractor
        children = list(feat_ext.children())
        mid = len(children) // 2

        # 第一块: [Conv2d, Identity, ReLU, MaxPool2d] + squeeze → 输出 3D
        # 第二块: unsqueeze + [Conv2d, Identity, ReLU, MaxPool2d] + squeeze → 输出 3D
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                _Unsqueeze3Dto4D(),  # 输入可能是 3D（来自 FiLM 调制后）
                *children[:mid],
                _Squeeze2Dto3D(),
            ),
            nn.Sequential(
                _Unsqueeze3Dto4D(),
                *children[mid:],
                _Squeeze2Dto3D(),
            ),
        ])

        # 暴露 classifier
        self.classifier = self._pex_net.clf
        self._dropout = self._pex_net.drop

        # 保存 spatial_conv 引用
        self._spatial_conv = self._pex_net.epoch_encoder.spatial_conv

    def forward(
        self,
        x: torch.Tensor,
        patient_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, n_channels, sequence_length] (e.g. [B, 1, 3000])
        Returns:
            logits: [batch, n_classes]
        """
        # spatial_conv 处理: [B, C, T] → unsqueeze → [B, 1, C, T] → spatial → 可能还是 [B, 1, C, T]
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self._spatial_conv(x)
        # squeeze 回 3D 给第一个 conv_block
        if x.dim() == 4 and x.shape[2] == 1:
            x = x.squeeze(2)  # [B, C, T]

        # 逐块前向传播（每块输出 3D）
        for block in self.conv_blocks:
            x = block(x)

        # flatten + dropout + classifier
        x = x.flatten(1)
        x = self._dropout(x)
        return self.classifier(x)


class PhysioExTinySleepNetWrapper(nn.Module):
    """包装 PhysioEx 的 TinySleepNet Net，暴露 FiLM 兼容接口。

    PhysioEx TinySleepNet CNN 结构:
    [ConvBlock, Pad, MaxPool, Dropout, ConvBlock×3, Pad, MaxPool, Flatten, Dropout]

    我们将 CNN 拆分为：
    - cnn: 不含 Flatten 和最后 Dropout 的卷积部分（输出 3D [B, C, T]）
    - lstm: LSTM 层
    - classifier: 分类头

    FiLM 在 cnn 输出（3D）上操作，然后 flatten → LSTM → classifier。
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 5):
        super().__init__()
        _, TNet = _try_import_physioex()
        if TNet is None:
            raise ImportError("PhysioEx not available")

        mc = {
            'in_channels': n_channels,
            'sf': 100,
            'n_classes': n_classes,
            'sequence_length': 1,
            'n_rnn_units': 128,
            'n_rnn_layers': 1,
        }
        self._pex_net = TNet(mc)

        # 拆分 CNN：去掉最后的 Flatten 和 Dropout
        orig_cnn = self._pex_net.feature_extractor.cnn
        cnn_children = list(orig_cnn.children())
        # 去掉最后两层 (Flatten, Dropout)
        self.cnn = nn.Sequential(*cnn_children[:-2])
        self._cnn_dropout = cnn_children[-1]  # Dropout

        # 暴露 lstm 和 classifier
        self.lstm = self._pex_net.clf.rnn
        self.classifier = self._pex_net.clf.clf
        self._rnn_dropout = self._pex_net.clf.rnn_dropout

    def forward(
        self,
        x: torch.Tensor,
        patient_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, n_channels, sequence_length] (e.g. [B, 1, 3000])
        Returns:
            logits: [batch, n_classes]
        """
        # CNN: [B, 1, 3000] → [B, 128, T] (3D, 不含 Flatten)
        cnn_out = self.cnn(x)

        # Flatten + Dropout
        cnn_flat = cnn_out.flatten(1)  # [B, 128*T] = [B, 2048]
        cnn_flat = self._cnn_dropout(cnn_flat)

        # LSTM 期望 [B, seq_len, feat]: 添加 seq 维度
        cnn_flat = cnn_flat.unsqueeze(1)  # [B, 1, 2048]
        lstm_out, _ = self.lstm(cnn_flat)  # [B, 1, 128]
        lstm_out = lstm_out[:, -1, :]  # [B, 128]
        lstm_out = self._rnn_dropout(lstm_out)

        return self.classifier(lstm_out)  # [B, 5]


# ---------------------------------------------------------------------------
# 自定义轻量模型（回退方案）
# ---------------------------------------------------------------------------

class Chambon2018Net(nn.Module):
    """增强版 Chambon2018-style CNN 睡眠分期模型。

    4 个卷积块，通道递增: 1 → 64 → 128 → 128 → 64
    总参数量 ~200K-300K，足以学习 5 类睡眠分期。

    相比 PhysioEx 原版 Chambon2018（仅 3,616 参数），本模型：
    - 使用更多卷积滤波器（64/128 vs 8）
    - 增加到 4 个卷积块
    - 添加 BatchNorm 提升训练稳定性
    - 保持与 FiLMWrappedChambon 兼容的接口

    属性（FiLMWrappedChambon 兼容）：
    - conv_blocks: nn.ModuleList
    - classifier: nn.Linear
    """

    def __init__(self, n_channels: int = 1, sequence_length: int = 3000, n_classes: int = 5):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=50, stride=6, padding=24),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(kernel_size=8, stride=8), nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=8, padding=4),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=8, padding=4),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=8, padding=4),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Conv1d(128, 64, kernel_size=8, padding=4),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Dropout(0.5),
            ),
        ])
        self._flat_dim = self._compute_flat_dim(n_channels, sequence_length)
        self.classifier = nn.Linear(self._flat_dim, n_classes)

    def _compute_flat_dim(self, n_channels: int, sequence_length: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, n_channels, sequence_length)
            for block in self.conv_blocks:
                x = block(x)
            return x.numel()

    def forward(self, x, patient_features=None):
        for block in self.conv_blocks:
            x = block(x)
        x = x.flatten(1)
        return self.classifier(x)


class TinySleepNetModel(nn.Module):
    """自定义 TinySleepNet CNN-LSTM 模型（回退方案）。

    属性（FiLMWrappedTinySleepNet 兼容）：
    - cnn: nn.Sequential
    - lstm: nn.LSTM
    - classifier: nn.Linear
    """

    def __init__(self, n_channels: int = 1, sequence_length: int = 3000,
                 n_classes: int = 5, lstm_hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(0.5),
            nn.Conv1d(32, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=25, padding=12),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(8),
        )
        self._cnn_out_channels = 128
        self._cnn_out_time = 8
        self.lstm = nn.LSTM(input_size=8, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x, patient_features=None):
        cnn_out = self.cnn(x)
        lstm_out, _ = self.lstm(cnn_out)
        return self.classifier(lstm_out[:, -1, :])


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_model(model_name: str, use_physioex: bool = True, **kwargs) -> nn.Module:
    """工厂函数：根据名称创建模型。

    Chambon2018: 始终使用增强版自定义架构（~200K+ 参数），
                 PhysioEx 原版仅 3,616 参数，不足以学习 5 类分期。
    TinySleepNet: 优先使用 PhysioEx 原生架构（1.5M 参数，足够大）。

    Args:
        model_name: "Chambon2018" 或 "TinySleepNet"
        use_physioex: 是否优先使用 PhysioEx 架构（仅对 TinySleepNet 生效）
        **kwargs: 传递给模型构造函数的参数
    Returns:
        nn.Module: 模型实例
    """
    # Chambon2018 始终使用增强版自定义架构
    if model_name == "Chambon2018":
        logger.info("使用增强版自定义 Chambon2018 架构 (~200K+ 参数)")
        return Chambon2018Net(**kwargs)

    # TinySleepNet 优先使用 PhysioEx
    if model_name == "TinySleepNet" and use_physioex:
        try:
            model = PhysioExTinySleepNetWrapper(**kwargs)
            logger.info("使用 PhysioEx 原生 TinySleepNet 架构")
            return model
        except (ImportError, Exception) as e:
            logger.warning("PhysioEx TinySleepNet 创建失败，回退到自定义实现: %s", e)

    # 回退到自定义实现
    fallback = {
        "Chambon2018": Chambon2018Net,
        "TinySleepNet": TinySleepNetModel,
    }
    if model_name not in fallback:
        raise ValueError(f"未知模型: {model_name}，支持: {list(fallback.keys())}")
    logger.info("使用自定义 %s 架构", model_name)
    return fallback[model_name](**kwargs)
