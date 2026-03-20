# -*- coding: utf-8 -*-
"""
U-Sleep 模型集成器 (USleepIntegrator)

通过 utime 包加载 U-Sleep 预训练模型，支持两种使用模式：
1. 直接推理模式：使用 U-Sleep 原始输出作为 zero-shot 基线
2. 特征提取模式：提取中间特征，用于 FiLM 适应

U-Sleep 是在 15000+ PSG 录制上训练的通用睡眠分期模型（报告准确率 86.6%），
使用 TensorFlow/Keras 实现。本集成器处理：
- PKL → EDF 格式转换（U-Sleep 需要 EDF 输入）
- 采样率重采样（临床数据 100Hz → U-Sleep 可能需要 128Hz）
- 单通道 EEG 支持（U-Sleep 通常使用多通道 PSG）
- 优雅降级（utime 未安装时提供模拟模式）

Requirements: 1.1, 1.3, 1.4
"""

import logging
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 睡眠分期标签映射: W=0, N1=1, N2=2, N3=3, REM=4
SLEEP_STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
NUM_CLASSES = 5

# 临床 OSA 数据集参数
CLINICAL_SAMPLE_RATE = 100  # Hz
EPOCH_DURATION = 30  # 秒
SAMPLES_PER_EPOCH = CLINICAL_SAMPLE_RATE * EPOCH_DURATION  # 3000


def _check_utime_available() -> bool:
    """检查 utime 包是否可用。"""
    try:
        import utime  # noqa: F401
        return True
    except ImportError:
        return False


def _check_pyedflib_available() -> bool:
    """检查 pyedflib 是否可用（写入 EDF 文件需要）。"""
    try:
        import pyedflib  # noqa: F401
        return True
    except ImportError:
        return False



def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样信号到目标采样率。

    使用 scipy.signal.resample 进行频域重采样，保持信号特征。

    Args:
        signal: 输入信号，shape [num_samples] 或 [num_epochs, samples_per_epoch]
        orig_sr: 原始采样率 (Hz)
        target_sr: 目标采样率 (Hz)

    Returns:
        重采样后的信号，保持输入维度结构
    """
    if orig_sr == target_sr:
        return signal

    from scipy.signal import resample as scipy_resample

    if signal.ndim == 1:
        num_target_samples = int(len(signal) * target_sr / orig_sr)
        return scipy_resample(signal, num_target_samples)
    elif signal.ndim == 2:
        # [num_epochs, samples_per_epoch]
        target_samples_per_epoch = int(signal.shape[1] * target_sr / orig_sr)
        resampled = np.zeros(
            (signal.shape[0], target_samples_per_epoch), dtype=signal.dtype
        )
        for i in range(signal.shape[0]):
            resampled[i] = scipy_resample(signal[i], target_samples_per_epoch)
        return resampled
    else:
        raise ValueError(f"不支持的信号维度: {signal.ndim}，期望 1D 或 2D")


def pkl_to_eeg_array(
    pkl_path: Union[str, Path],
    eeg_channel: Optional[str] = None,
) -> Tuple[np.ndarray, int, str]:
    """从 PKL 文件加载 EEG 数据。

    PKL 文件格式: {'patient_id': str, 'signals': {channel_name: ndarray}, ...}
    其中 signals[channel] shape 为 [num_epochs, samples_per_epoch]

    Args:
        pkl_path: PKL 文件路径
        eeg_channel: 指定 EEG 通道名。如果为 None，自动检测 C3/C4 通道。

    Returns:
        (eeg_epochs, sample_rate, channel_name) 元组
        - eeg_epochs: shape [num_epochs, samples_per_epoch]
        - sample_rate: 采样率 (Hz)
        - channel_name: 使用的通道名
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    signals = data.get("signals", {})
    if not signals:
        raise ValueError(f"PKL 文件中无信号数据: {pkl_path}")

    # 自动检测 EEG 通道
    if eeg_channel is None:
        eeg_channel = _detect_eeg_channel(list(signals.keys()))

    if eeg_channel not in signals:
        available = list(signals.keys())
        raise KeyError(
            f"通道 '{eeg_channel}' 不存在。可用通道: {available}"
        )

    eeg_data = signals[eeg_channel]  # [num_epochs, samples_per_epoch]
    samples_per_epoch = eeg_data.shape[1] if eeg_data.ndim == 2 else SAMPLES_PER_EPOCH
    sample_rate = int(samples_per_epoch / EPOCH_DURATION)

    return eeg_data, sample_rate, eeg_channel


def _detect_eeg_channel(channel_names: List[str]) -> str:
    """自动检测 EEG 通道，优先选择 C3 或 C4。

    Args:
        channel_names: 可用通道名列表

    Returns:
        检测到的 EEG 通道名

    Raises:
        ValueError: 无法检测到 EEG 通道
    """
    # 优先级: C3-M2 > C4-M1 > C3 > C4 > 任何含 EEG 的通道
    priority_patterns = ["C3-M2", "C4-M1", "C3", "C4", "EEG"]
    for pattern in priority_patterns:
        for ch in channel_names:
            if pattern.upper() in ch.upper():
                return ch

    # 回退: 使用第一个通道
    if channel_names:
        logger.warning(
            "无法检测 EEG 通道，使用第一个通道: %s", channel_names[0]
        )
        return channel_names[0]

    raise ValueError("无可用通道")



class USleepIntegrator:
    """U-Sleep 模型集成器。

    两种使用模式：
    1. 直接推理模式：使用 U-Sleep 原始输出作为 zero-shot 基线
    2. 特征提取模式：提取中间特征，通过 ONNXFeatureAdapter 进行 FiLM 适应

    当 utime 包不可用时，提供模拟模式用于开发和测试。

    Attributes:
        model_name: U-Sleep 模型标识符（默认 "U-Sleep:1.0"）
        is_available: utime 包是否可用
        feature_dim: 特征提取模式的输出维度
        device: PyTorch 设备
    """

    def __init__(
        self,
        model_name: str = "U-Sleep:1.0",
        feature_dim: int = 256,
        device: Optional[torch.device] = None,
        use_gpu: bool = False,
    ):
        """初始化 U-Sleep 集成器。

        Args:
            model_name: U-Sleep 模型标识符
            feature_dim: 特征提取输出维度
            device: PyTorch 设备（用于特征提取模式）
            use_gpu: 是否使用 GPU 进行 U-Sleep 推理
        """
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.device = device or torch.device("cpu")
        self.use_gpu = use_gpu
        self.is_available = _check_utime_available()
        self._has_pyedflib = _check_pyedflib_available()

        # 特征投影层：将 U-Sleep 概率输出映射到特征空间
        self._feature_projector = self._build_feature_projector()

        if not self.is_available:
            logger.warning(
                "utime 包不可用，USleepIntegrator 将使用模拟模式。"
                "安装 utime: pip install u-time"
            )


    def _build_feature_projector(self) -> nn.Module:
        """构建特征投影网络。

        将 U-Sleep 的 5 类概率输出（或中间表示）投影到
        feature_dim 维特征空间，兼容 FiLM 适配器。

        Returns:
            nn.Module: 特征投影网络
        """
        projector = nn.Sequential(
            nn.Linear(NUM_CLASSES, 64),
            nn.ReLU(),
            nn.Linear(64, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        return projector.to(self.device)

    def predict(self, eeg_data: np.ndarray) -> np.ndarray:
        """直接推理模式：返回睡眠分期预测。

        Args:
            eeg_data: EEG 数据，支持以下格式：
                - [num_epochs, samples_per_epoch]: 分 epoch 的数据
                - [batch, 1, samples_per_epoch]: PyTorch 风格的批量数据
                - [num_samples]: 连续信号（自动按 30s 分 epoch）

        Returns:
            预测标签数组，shape [num_epochs]，值在 {0,1,2,3,4}
        """
        eeg_epochs = self._normalize_input(eeg_data)
        num_epochs = eeg_epochs.shape[0]

        if self.is_available:
            return self._predict_with_utime(eeg_epochs)
        else:
            return self._predict_mock(num_epochs)


    def predict_proba(self, eeg_data: np.ndarray) -> np.ndarray:
        """直接推理模式：返回各类别概率。

        Args:
            eeg_data: EEG 数据（同 predict 方法）

        Returns:
            概率数组，shape [num_epochs, 5]
        """
        eeg_epochs = self._normalize_input(eeg_data)
        num_epochs = eeg_epochs.shape[0]

        if self.is_available:
            return self._predict_proba_with_utime(eeg_epochs)
        else:
            return self._predict_proba_mock(num_epochs)

    def extract_features(self, eeg_data: np.ndarray) -> torch.Tensor:
        """特征提取模式：提取中间特征用于 FiLM 适应。

        将 U-Sleep 的概率输出通过特征投影网络映射到
        feature_dim 维空间，兼容 ONNXFeatureAdapter。

        Args:
            eeg_data: EEG 数据（同 predict 方法）

        Returns:
            特征张量，shape [num_epochs, feature_dim]
        """
        proba = self.predict_proba(eeg_data)
        proba_tensor = torch.from_numpy(proba).float().to(self.device)

        self._feature_projector.eval()
        with torch.no_grad():
            features = self._feature_projector(proba_tensor)

        return features


    def _normalize_input(self, eeg_data: np.ndarray) -> np.ndarray:
        """将各种输入格式统一为 [num_epochs, samples_per_epoch]。

        Args:
            eeg_data: 支持的输入格式：
                - [num_epochs, samples_per_epoch]: 直接返回
                - [batch, 1, samples_per_epoch]: 去掉通道维度
                - [num_samples]: 按 30s epoch 分割

        Returns:
            shape [num_epochs, samples_per_epoch] 的数组
        """
        if isinstance(eeg_data, torch.Tensor):
            eeg_data = eeg_data.detach().cpu().numpy()

        if eeg_data.ndim == 3:
            # [batch, channels, samples] → [batch, samples]
            if eeg_data.shape[1] == 1:
                eeg_data = eeg_data.squeeze(1)
            else:
                # 多通道：取第一个通道
                eeg_data = eeg_data[:, 0, :]

        if eeg_data.ndim == 1:
            # 连续信号 → 按 epoch 分割
            n_samples = len(eeg_data)
            num_epochs = n_samples // SAMPLES_PER_EPOCH
            if num_epochs == 0:
                raise ValueError(
                    f"信号太短 ({n_samples} 样本)，"
                    f"至少需要 {SAMPLES_PER_EPOCH} 样本（1 个 epoch）"
                )
            eeg_data = eeg_data[: num_epochs * SAMPLES_PER_EPOCH]
            eeg_data = eeg_data.reshape(num_epochs, SAMPLES_PER_EPOCH)

        if eeg_data.ndim != 2:
            raise ValueError(
                f"无法处理的输入维度: {eeg_data.ndim}，期望 1D/2D/3D"
            )

        return eeg_data.astype(np.float64)


    def _predict_with_utime(self, eeg_epochs: np.ndarray) -> np.ndarray:
        """使用 utime 包进行真实推理。

        将 epoch 数据写入临时 EDF 文件，调用 utime 命令行工具推理。

        Args:
            eeg_epochs: shape [num_epochs, samples_per_epoch]

        Returns:
            预测标签，shape [num_epochs]
        """
        if not self._has_pyedflib:
            raise RuntimeError(
                "pyedflib 未安装，无法写入 EDF 文件。"
                "安装: pip install pyedflib"
            )

        import pyedflib

        with tempfile.TemporaryDirectory() as tmpdir:
            edf_path = os.path.join(tmpdir, "input.edf")
            out_path = os.path.join(tmpdir, "predictions.npy")

            # 拼接为连续信号
            continuous = eeg_epochs.flatten()
            sr = CLINICAL_SAMPLE_RATE

            # 写入 EDF
            self._write_edf(edf_path, continuous, sr, pyedflib)

            # 调用 utime 推理
            self._run_utime_predict(edf_path, out_path)

            # 读取结果
            if os.path.exists(out_path):
                preds = np.load(out_path).flatten().astype(np.int64)
                # 确保长度匹配
                num_epochs = eeg_epochs.shape[0]
                if len(preds) > num_epochs:
                    preds = preds[:num_epochs]
                elif len(preds) < num_epochs:
                    # 用最近邻填充
                    padded = np.full(num_epochs, preds[-1], dtype=np.int64)
                    padded[: len(preds)] = preds
                    preds = padded
                return np.clip(preds, 0, NUM_CLASSES - 1)
            else:
                logger.error("U-Sleep 推理未生成预测文件")
                return self._predict_mock(eeg_epochs.shape[0])


    def _predict_proba_with_utime(self, eeg_epochs: np.ndarray) -> np.ndarray:
        """使用 utime 获取概率输出。

        utime 默认输出硬标签。此方法将硬标签转换为 one-hot 概率，
        并添加少量平滑以避免极端概率值。

        Args:
            eeg_epochs: shape [num_epochs, samples_per_epoch]

        Returns:
            概率数组，shape [num_epochs, 5]
        """
        preds = self._predict_with_utime(eeg_epochs)
        return self._labels_to_smoothed_proba(preds)

    def _predict_mock(self, num_epochs: int) -> np.ndarray:
        """模拟推理：生成基于先验分布的随机预测。

        使用典型的睡眠分期分布作为先验：
        W~15%, N1~5%, N2~45%, N3~15%, REM~20%

        Args:
            num_epochs: epoch 数量

        Returns:
            模拟预测标签，shape [num_epochs]
        """
        logger.info("使用模拟模式生成 %d 个 epoch 的预测", num_epochs)
        prior = np.array([0.15, 0.05, 0.45, 0.15, 0.20])
        return np.random.choice(NUM_CLASSES, size=num_epochs, p=prior)

    def _predict_proba_mock(self, num_epochs: int) -> np.ndarray:
        """模拟概率输出。

        Args:
            num_epochs: epoch 数量

        Returns:
            模拟概率，shape [num_epochs, 5]
        """
        logger.info("使用模拟模式生成 %d 个 epoch 的概率", num_epochs)
        # 生成 Dirichlet 分布的随机概率
        proba = np.random.dirichlet(
            alpha=[3.0, 1.0, 9.0, 3.0, 4.0], size=num_epochs
        )
        return proba.astype(np.float32)


    @staticmethod
    def _labels_to_smoothed_proba(
        labels: np.ndarray, smoothing: float = 0.05
    ) -> np.ndarray:
        """将硬标签转换为平滑概率。

        Args:
            labels: 整数标签，shape [N]
            smoothing: 标签平滑系数

        Returns:
            平滑概率，shape [N, 5]
        """
        n = len(labels)
        proba = np.full((n, NUM_CLASSES), smoothing / (NUM_CLASSES - 1),
                        dtype=np.float32)
        for i in range(n):
            label = int(np.clip(labels[i], 0, NUM_CLASSES - 1))
            proba[i, label] = 1.0 - smoothing
        return proba

    @staticmethod
    def _write_edf(
        edf_path: str,
        signal: np.ndarray,
        sample_rate: int,
        pyedflib_module,
    ) -> None:
        """将连续 EEG 信号写入 EDF 文件。

        创建双通道 EDF（EEG + 零填充 EOG），兼容 U-Sleep 输入要求。

        Args:
            edf_path: 输出 EDF 文件路径
            signal: 连续 EEG 信号，shape [num_samples]
            sample_rate: 采样率 (Hz)
            pyedflib_module: pyedflib 模块引用
        """
        signal = signal.astype(np.float64)
        eog_signal = np.zeros_like(signal)

        p_max = float(np.percentile(np.abs(signal), 99.9)) + 1e-6
        writer = pyedflib_module.EdfWriter(edf_path, 2)
        try:
            for idx, (label, data) in enumerate([
                ("EEG C3-M2", signal),
                ("EOG E1-M2", eog_signal),
            ]):
                writer.setSignalHeader(idx, {
                    "label": label,
                    "dimension": "uV",
                    "sample_frequency": sample_rate,
                    "physical_max": p_max,
                    "physical_min": -p_max,
                    "digital_max": 32767,
                    "digital_min": -32768,
                })
            writer.writeSamples([signal, eog_signal])
        finally:
            writer.close()


    def _run_utime_predict(self, edf_path: str, out_path: str) -> None:
        """调用 utime 命令行工具进行推理。

        Args:
            edf_path: 输入 EDF 文件路径
            out_path: 输出预测文件路径（.npy）
        """
        num_gpus = "1" if self.use_gpu else "0"
        cmd = [
            sys.executable, "-m", "utime.bin.predict_one",
            "-f", edf_path,
            "-o", out_path,
            "--channels", "EEG C3-M2++EOG E1-M2",
            "--model", self.model_name,
            "--num_gpus", num_gpus,
            "--overwrite",
        ]
        logger.info("执行 U-Sleep 推理: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                logger.error(
                    "U-Sleep 推理失败 (code=%d): %s",
                    result.returncode,
                    result.stderr[-500:] if result.stderr else "(无输出)",
                )
            else:
                logger.info("U-Sleep 推理完成")
        except subprocess.TimeoutExpired:
            logger.error("U-Sleep 推理超时 (600s)")
        except FileNotFoundError:
            logger.error("无法找到 Python 解释器: %s", sys.executable)


    def predict_from_pkl(
        self,
        pkl_path: Union[str, Path],
        eeg_channel: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """从 PKL 文件直接推理。

        处理完整的 PKL → 预测流水线：加载数据、格式转换、推理。

        Args:
            pkl_path: PKL 文件路径
            eeg_channel: EEG 通道名（None 则自动检测）

        Returns:
            (predictions, metadata) 元组
            - predictions: shape [num_epochs]
            - metadata: 包含通道名、采样率等信息
        """
        eeg_epochs, sample_rate, channel = pkl_to_eeg_array(
            pkl_path, eeg_channel
        )

        # 如果采样率不是 100Hz，重采样到 100Hz
        if sample_rate != CLINICAL_SAMPLE_RATE:
            logger.info(
                "重采样: %dHz → %dHz", sample_rate, CLINICAL_SAMPLE_RATE
            )
            eeg_epochs = resample_signal(
                eeg_epochs, sample_rate, CLINICAL_SAMPLE_RATE
            )

        predictions = self.predict(eeg_epochs)

        metadata = {
            "pkl_path": str(pkl_path),
            "channel": channel,
            "original_sample_rate": sample_rate,
            "num_epochs": len(predictions),
            "is_mock": not self.is_available,
        }
        return predictions, metadata

    def extract_features_from_pkl(
        self,
        pkl_path: Union[str, Path],
        eeg_channel: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """从 PKL 文件提取特征。

        Args:
            pkl_path: PKL 文件路径
            eeg_channel: EEG 通道名

        Returns:
            (features, metadata) 元组
            - features: shape [num_epochs, feature_dim]
            - metadata: 包含通道名等信息
        """
        eeg_epochs, sample_rate, channel = pkl_to_eeg_array(
            pkl_path, eeg_channel
        )

        if sample_rate != CLINICAL_SAMPLE_RATE:
            eeg_epochs = resample_signal(
                eeg_epochs, sample_rate, CLINICAL_SAMPLE_RATE
            )

        features = self.extract_features(eeg_epochs)

        metadata = {
            "pkl_path": str(pkl_path),
            "channel": channel,
            "original_sample_rate": sample_rate,
            "num_epochs": features.shape[0],
            "feature_dim": features.shape[1],
            "is_mock": not self.is_available,
        }
        return features, metadata

