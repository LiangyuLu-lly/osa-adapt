"""从 PKL 文件加载 PSG 信号的 PyTorch Dataset。

每个样本为一个 epoch: (signal_tensor, stage_label, patient_features_dict)
用于 OSA-Adapt 论文的真实 GPU 训练流水线。
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# OSA 严重程度字符串到整数的映射
SEVERITY_MAP = {
    "normal": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}


class PSGDataset(Dataset):
    """从 PKL 文件加载 PSG 信号的 PyTorch Dataset。

    每个样本为一个 epoch: (signal_tensor, stage_label, patient_features_dict)
    """

    def __init__(
        self,
        patient_ids: List[str],
        pkl_dir: str,
        severity_data: Dict[str, Dict],
        demographics_generator: "DemographicsGenerator",
        channel_name: str = "C4-M1",
        fallback_channel_prefix: str = "C",
    ):
        """
        Args:
            patient_ids: 要加载的患者 ID 列表
            pkl_dir: PKL 文件目录路径
            severity_data: 患者严重程度数据字典 (patient_id -> {ahi, osa_severity, ...})
            demographics_generator: 合成人口学数据生成器
            channel_name: 目标 EEG 通道名（默认 C4-M1）
            fallback_channel_prefix: 通道不可用时的回退前缀（默认 C）
        """
        self.pkl_dir = pkl_dir
        self.severity_data = severity_data
        self.demographics_generator = demographics_generator
        self.channel_name = channel_name
        self.fallback_channel_prefix = fallback_channel_prefix

        # 每个成功加载的患者的数据
        # patient_data[i] = {signals, sleep_stages, ahi, severity, demographics, patient_id}
        self.patient_data: List[Dict] = []
        # 扁平化 epoch 索引: [(patient_idx, epoch_idx), ...]
        self.epoch_index: List[Tuple[int, int]] = []
        # patient_id -> 在 patient_data 中的索引
        self._patient_id_to_idx: Dict[str, int] = {}
        # 文件名前缀 -> 完整路径的索引（一次性构建）
        self._pkl_index: Dict[str, str] = self._build_pkl_index()

        self._load_patients(patient_ids)

    def _build_pkl_index(self) -> Dict[str, str]:
        """一次性构建 PKL 文件索引：数字前缀 -> 完整路径。"""
        index = {}
        if not os.path.isdir(self.pkl_dir):
            return index
        for fname in os.listdir(self.pkl_dir):
            if not fname.endswith(".pkl"):
                continue
            # 提取数字前缀（如 001_xxx.pkl -> 001）
            prefix = fname.split("_", 1)[0]
            full_path = os.path.join(self.pkl_dir, fname)
            index[prefix] = full_path
            # 也用完整文件名（不含扩展名）作为键
            index[fname[:-4]] = full_path
        return index

    def _find_pkl_file(self, patient_id: str) -> Optional[str]:
        """查找与 patient_id 对应的 PKL 文件。

        支持两种匹配方式：
        1. patient_id 直接作为键（如 patient_001）
        2. patient_id 的数字部分作为前缀（如 patient_001 -> 001）
        """
        # 直接匹配
        if patient_id in self._pkl_index:
            return self._pkl_index[patient_id]
        # 提取数字部分匹配（如 patient_001 -> 001）
        parts = patient_id.split("_")
        if len(parts) >= 2:
            numeric_part = parts[-1]
            if numeric_part in self._pkl_index:
                return self._pkl_index[numeric_part]
        return None

    def _select_channel(self, signals: Dict[str, np.ndarray]) -> Optional[str]:
        """选择 EEG 通道，必要时回退到第一个匹配前缀的通道。"""
        if self.channel_name in signals:
            return self.channel_name
        # 回退：查找第一个以 fallback_channel_prefix 开头的通道
        for ch_name in signals:
            if ch_name.startswith(self.fallback_channel_prefix):
                logger.warning(
                    "通道 '%s' 不可用，回退到 '%s'", self.channel_name, ch_name
                )
                return ch_name
        return None

    def _load_patients(self, patient_ids: List[str]) -> None:
        """加载所有患者的 PKL 数据，构建扁平化 epoch 索引。"""
        for patient_id in patient_ids:
            try:
                self._load_single_patient(patient_id)
            except Exception as e:
                logger.warning("跳过患者 %s: %s", patient_id, e)

    def _load_single_patient(self, patient_id: str) -> None:
        """加载单个患者的 PKL 文件。"""
        pkl_path = self._find_pkl_file(patient_id)
        if pkl_path is None:
            logger.warning("未找到患者 %s 的 PKL 文件", patient_id)
            return

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning("无法加载 PKL 文件 %s: %s", pkl_path, e)
            return

        # 验证必需字段
        if "signals" not in data or not isinstance(data.get("signals"), dict):
            logger.warning("PKL 文件 %s 缺少 signals 字段", pkl_path)
            return
        if "sleep_stages" not in data:
            logger.warning("PKL 文件 %s 缺少 sleep_stages 字段", pkl_path)
            return

        signals = data["signals"]
        sleep_stages = np.asarray(data["sleep_stages"])

        # 选择通道
        selected_channel = self._select_channel(signals)
        if selected_channel is None:
            logger.warning(
                "PKL 文件 %s 无可用 EEG 通道，跳过", pkl_path
            )
            return

        channel_signals = signals[selected_channel]
        if channel_signals.ndim != 2 or channel_signals.shape[1] != 3000:
            logger.warning(
                "PKL 文件 %s 通道 %s 形状异常: %s，跳过",
                pkl_path, selected_channel, channel_signals.shape,
            )
            return

        n_epochs = channel_signals.shape[0]
        if len(sleep_stages) != n_epochs:
            logger.warning(
                "PKL 文件 %s sleep_stages 长度 (%d) 与信号 epochs (%d) 不匹配，跳过",
                pkl_path, len(sleep_stages), n_epochs,
            )
            return

        # 获取 AHI 和 severity
        ahi = float(data.get("ahi", 0.0))
        osa_severity_str = str(data.get("osa_severity", "normal")).lower()
        severity = SEVERITY_MAP.get(osa_severity_str, 0)

        # 也可以从 severity_data 获取（优先使用 severity_data）
        if patient_id in self.severity_data:
            sev_info = self.severity_data[patient_id]
            ahi = float(sev_info.get("ahi", ahi))
            sev_str = str(sev_info.get("osa_severity", osa_severity_str)).lower()
            severity = SEVERITY_MAP.get(sev_str, severity)

        # 生成合成人口学数据
        demographics = self.demographics_generator.generate(patient_id, ahi, severity)

        # 过滤掉无效标签的epoch (sleep_stages == -1)
        # 只保留有效的睡眠分期标签 (0-4: Wake, N1, N2, N3, REM)
        valid_mask = np.array([
            0 <= int(sleep_stages[ei]) <= 4
            for ei in range(n_epochs)
        ])
        
        if not valid_mask.any():
            logger.warning(
                "PKL 文件 %s 没有有效的睡眠分期标签，跳过", pkl_path
            )
            return
        
        # Per-patient z-score 归一化（与预训练脚本保持一致）
        # 原始 EEG 信号值范围极小（约 ±0.0005），不归一化会导致
        # 梯度过小、模型无法学习
        valid_data = channel_signals[valid_mask]
        patient_mean = valid_data.mean()
        patient_std = valid_data.std()
        
        # 应用归一化
        normalized_signals = (channel_signals - patient_mean) / (patient_std + 1e-8)
        
        # 存储患者数据
        patient_idx = len(self.patient_data)
        self._patient_id_to_idx[patient_id] = patient_idx
        self.patient_data.append({
            "signals": normalized_signals.astype(np.float32),
            "sleep_stages": sleep_stages,
            "ahi": ahi,
            "severity": severity,
            "demographics": demographics,
            "patient_id": patient_id,
            "valid_mask": valid_mask,  # 保存有效epoch的mask
        })

        # 构建扁平化 epoch 索引 - 只包含有效标签的epoch
        for epoch_idx in range(n_epochs):
            if valid_mask[epoch_idx]:
                self.epoch_index.append((patient_idx, epoch_idx))

    def __len__(self) -> int:
        """返回总 epoch 数。"""
        return len(self.epoch_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
        """
        Returns:
            signal: shape [1, 3000] 单通道 EEG 信号
            stage_label: int, 0-4 (W/N1/N2/N3/REM)
            patient_features: dict with keys {ahi, severity, age, sex, bmi}
                              每个值为标量 tensor
        """
        patient_idx, epoch_idx = self.epoch_index[idx]
        patient = self.patient_data[patient_idx]

        # 信号: [1, 3000]
        signal = torch.from_numpy(
            patient["signals"][epoch_idx].copy()
        ).float().unsqueeze(0)

        # 睡眠分期标签
        stage_label = int(patient["sleep_stages"][epoch_idx])

        # 患者特征
        demographics = patient["demographics"]
        patient_features = {
            "ahi": torch.tensor(patient["ahi"], dtype=torch.float32),
            "severity": torch.tensor(patient["severity"], dtype=torch.long),  # 修复: embedding需要long类型
            "age": torch.tensor(demographics["age"], dtype=torch.float32),
            "sex": torch.tensor(demographics["sex"], dtype=torch.long),  # 修复: embedding需要long类型
            "bmi": torch.tensor(demographics["bmi"], dtype=torch.float32),
        }

        return signal, stage_label, patient_features

    def get_patient_epoch_indices(self, patient_id: str) -> List[int]:
        """返回指定患者的所有 epoch 索引，用于 per-patient 评估。"""
        if patient_id not in self._patient_id_to_idx:
            return []
        patient_idx = self._patient_id_to_idx[patient_id]
        return [
            i for i, (pidx, _) in enumerate(self.epoch_index)
            if pidx == patient_idx
        ]
