"""
模型包装器：将FiLM适配器集成到预训练睡眠分期模型中

提供三种包装器：
- FiLMWrappedChambon: Chambon2018 CNN模型 + FiLM（每个卷积块后）
- FiLMWrappedTinySleepNet: TinySleepNet CNN-LSTM模型 + FiLM（CNN与LSTM之间）
- ONNXFeatureAdapter: U-Sleep ONNX特征提取 + PyTorch FiLM调制 + 分类头

所有包装器冻结基础模型参数，仅FiLM层和条件化模块可训练。
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .film_adapter import FiLMAdapter
from .severity_conditioner import SeverityConditioner


class FiLMWrappedChambon(nn.Module):
    """Chambon2018 + FiLM适配器。在每个卷积块后插入FiLM层。

    假设 base_model 具有以下属性：
    - conv_blocks: 卷积块列表（3个）
    - classifier: 最终全连接分类层
    """

    def __init__(self, base_model: nn.Module, conditioner: SeverityConditioner):
        """
        Args:
            base_model: 预训练的Chambon2018模型，需有 conv_blocks 和 classifier 属性
            conditioner: 严重程度条件化模块
        """
        super().__init__()
        self.base_model = base_model
        self.conditioner = conditioner

        # 检测每个卷积块的输出通道数，为每个块创建对应的FiLM层
        self.film_layers = nn.ModuleList()
        for block in self.base_model.conv_blocks:
            out_channels = self._detect_out_channels(block)
            self.film_layers.append(
                FiLMAdapter(
                    feature_dim=out_channels,
                    condition_dim=conditioner.condition_dim,
                )
            )

        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False

    @staticmethod
    def _detect_out_channels(block: nn.Module) -> int:
        """从卷积块中检测输出通道数。

        遍历模块查找最后一个 Conv1d/Conv2d 层的 out_channels。
        """
        out_ch = None
        for module in block.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                out_ch = module.out_channels
        if out_ch is None:
            raise ValueError(
                "无法从卷积块中检测输出通道数，"
                "请确保块中包含 Conv1d 或 Conv2d 层"
            )
        return out_ch

    def forward(
        self, x: torch.Tensor, patient_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: EEG输入张量
            patient_features: 患者临床特征字典，包含键:
                ahi, severity, age, sex, bmi
        Returns:
            分类logits, shape [B, num_classes]
        """
        # 生成条件向量
        condition = self.conditioner(
            ahi=patient_features["ahi"],
            severity=patient_features["severity"],
            age=patient_features["age"],
            sex=patient_features["sex"],
            bmi=patient_features["bmi"],
        )

        # 逐块前向传播，每块后插入FiLM调制
        for conv_block, film in zip(self.base_model.conv_blocks, self.film_layers):
            x = conv_block(x)
            x = film(x, condition)

        # 展平后通过分类头
        x = x.flatten(1)
        x = self.base_model.classifier(x)
        return x

    def get_trainable_params(self) -> List[nn.Parameter]:
        """返回可训练参数列表（仅FiLM层 + 条件化模块）"""
        params = list(self.film_layers.parameters())
        params += list(self.conditioner.parameters())
        return params

    def get_film_param_count(self) -> int:
        """返回FiLM层参数数量"""
        return sum(p.numel() for p in self.film_layers.parameters())

    def get_base_param_count(self) -> int:
        """返回基础模型参数数量"""
        return sum(p.numel() for p in self.base_model.parameters())


class FiLMWrappedTinySleepNet(nn.Module):
    """TinySleepNet + FiLM适配器。在CNN特征提取器和LSTM之间插入FiLM层。

    假设 base_model 具有以下属性：
    - cnn: CNN特征提取器（或 cnn_forward 方法）
    - lstm: LSTM序列编码器
    - classifier: 最终分类层
    """

    def __init__(self, base_model: nn.Module, conditioner: SeverityConditioner):
        """
        Args:
            base_model: 预训练的TinySleepNet模型
            conditioner: 严重程度条件化模块
        """
        super().__init__()
        self.base_model = base_model
        self.conditioner = conditioner

        # 检测CNN输出特征维度
        cnn_out_dim = self._detect_cnn_out_dim()
        self.film_cnn = FiLMAdapter(
            feature_dim=cnn_out_dim,
            condition_dim=conditioner.condition_dim,
        )

        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _detect_cnn_out_dim(self) -> int:
        """检测CNN输出特征维度。

        尝试从 base_model.cnn 中查找最后一个线性层或卷积层的输出维度。
        如果无法自动检测，默认使用128。
        """
        cnn = getattr(self.base_model, "cnn", None)
        if cnn is None:
            return 128

        out_dim = None
        for module in cnn.modules():
            if isinstance(module, nn.Linear):
                out_dim = module.out_features
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                out_dim = module.out_channels
        return out_dim if out_dim is not None else 128

    def forward(
        self, x: torch.Tensor, patient_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: EEG输入张量
            patient_features: 患者临床特征字典
        Returns:
            分类logits
        """
        # 生成条件向量
        condition = self.conditioner(
            ahi=patient_features["ahi"],
            severity=patient_features["severity"],
            age=patient_features["age"],
            sex=patient_features["sex"],
            bmi=patient_features["bmi"],
        )

        # CNN特征提取（支持 cnn_forward 方法或 cnn 属性）
        if hasattr(self.base_model, "cnn_forward"):
            cnn_features = self.base_model.cnn_forward(x)
        else:
            cnn_features = self.base_model.cnn(x)

        # FiLM调制CNN特征（可能是 2D [B, F] 或 3D [B, C, T]）
        cnn_features = self.film_cnn(cnn_features, condition)

        # 如果是 3D，需要 flatten 后再送入 LSTM
        if cnn_features.dim() == 3:
            cnn_features = cnn_features.flatten(1)  # [B, C*T]

        # LSTM 期望 3D [B, seq_len, feat]
        if cnn_features.dim() == 2:
            cnn_features = cnn_features.unsqueeze(1)  # [B, 1, feat]

        # LSTM序列编码
        lstm_out, _ = self.base_model.lstm(cnn_features)

        # 取最后一个时间步（如果是序列输出）
        if lstm_out.dim() == 3:
            lstm_out = lstm_out[:, -1, :]

        # Dropout（如果模型有的话）
        if hasattr(self.base_model, '_rnn_dropout'):
            lstm_out = self.base_model._rnn_dropout(lstm_out)

        # 分类
        output = self.base_model.classifier(lstm_out)
        return output

    def get_trainable_params(self) -> List[nn.Parameter]:
        """返回可训练参数列表（仅FiLM层 + 条件化模块）"""
        params = list(self.film_cnn.parameters())
        params += list(self.conditioner.parameters())
        return params

    def get_film_param_count(self) -> int:
        """返回FiLM层参数数量"""
        return sum(p.numel() for p in self.film_cnn.parameters())

    def get_base_param_count(self) -> int:
        """返回基础模型参数数量"""
        return sum(p.numel() for p in self.base_model.parameters())



class ONNXFeatureAdapter(nn.Module):
    """U-Sleep ONNX + PyTorch FiLM后处理。

    U-Sleep使用ONNX格式，无法直接插入FiLM层。
    方案：接收预提取的ONNX中间特征，在PyTorch中通过FiLM调制后
    接新的分类头进行预测。

    不在 __init__ 中创建ONNX session，避免强制依赖 onnxruntime。
    可通过 set_onnx_session() 后续设置。
    """

    def __init__(
        self,
        feature_dim: int,
        conditioner: SeverityConditioner,
        num_classes: int = 5,
    ):
        """
        Args:
            feature_dim: ONNX模型提取的特征维度
            conditioner: 严重程度条件化模块
            num_classes: 睡眠分期类别数（默认5: W/N1/N2/N3/REM）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.conditioner = conditioner
        self.num_classes = num_classes

        # FiLM调制层
        self.film = FiLMAdapter(
            feature_dim=feature_dim,
            condition_dim=conditioner.condition_dim,
        )

        # 新的分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes),
        )

        # ONNX session（可选，后续设置）
        self.ort_session = None

    def set_onnx_session(self, session) -> None:
        """设置ONNX Runtime推理会话。

        Args:
            session: onnxruntime.InferenceSession 实例
        """
        self.ort_session = session

    def forward(
        self,
        features: torch.Tensor,
        patient_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """对预提取的特征进行FiLM调制和分类。

        Args:
            features: ONNX模型预提取的特征, shape [B, feature_dim]
                      或 [B, feature_dim, T]
            patient_features: 患者临床特征字典
        Returns:
            分类logits, shape [B, num_classes]
        """
        # 生成条件向量
        condition = self.conditioner(
            ahi=patient_features["ahi"],
            severity=patient_features["severity"],
            age=patient_features["age"],
            sex=patient_features["sex"],
            bmi=patient_features["bmi"],
        )

        # FiLM调制
        modulated = self.film(features, condition)

        # 如果是3D特征，先做全局平均池化
        if modulated.dim() == 3:
            modulated = modulated.mean(dim=-1)  # [B, feature_dim]

        # 分类
        return self.classifier(modulated)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """返回可训练参数列表（FiLM + 条件化模块 + 分类头）"""
        params = list(self.film.parameters())
        params += list(self.conditioner.parameters())
        params += list(self.classifier.parameters())
        return params

    def get_film_param_count(self) -> int:
        """返回FiLM层参数数量"""
        return sum(p.numel() for p in self.film.parameters())

    def get_base_param_count(self) -> int:
        """返回基础模型参数数量（ONNX模型无PyTorch参数，返回0）"""
        return 0
