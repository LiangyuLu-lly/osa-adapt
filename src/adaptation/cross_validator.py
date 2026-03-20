"""
5折患者级别分层交叉验证器

确保：
1. 同一患者的所有epoch仅出现在同一折（患者级别划分）
2. 各折中OSA严重程度分布与总体一致（分层）
3. 使用固定种子保证可复现

Requirements: 6.1, 6.2
"""

from typing import List, Tuple, Dict
import random
from collections import defaultdict


class CrossValidator:
    """
    5折患者级别分层交叉验证

    确保：
    1. 同一患者的所有epoch仅出现在同一折
    2. 各折中OSA严重程度分布与总体一致
    3. 使用固定种子保证可复现
    """

    def __init__(self, n_folds: int = 5, seed: int = 42):
        if n_folds < 2:
            raise ValueError(f"n_folds 必须 >= 2，当前值: {n_folds}")
        self.n_folds = n_folds
        self.seed = seed

    def split(
        self,
        patient_ids: List[str],
        severity_labels: List[int],
    ) -> List[Tuple[List[str], List[str]]]:
        """
        返回 n_folds 个 (train_ids, test_ids) 元组。

        按OSA严重程度分层，在每个严重程度组内用固定种子打乱后，
        以round-robin方式将患者分配到各折，确保各折严重程度分布
        与总体一致。

        Args:
            patient_ids: 患者ID列表
            severity_labels: 对应的严重程度标签 (0=Normal, 1=Mild, 2=Moderate, 3=Severe)

        Returns:
            长度为 n_folds 的列表，每个元素为 (train_ids, test_ids) 元组

        Raises:
            ValueError: 当 patient_ids 和 severity_labels 长度不一致时
            ValueError: 当患者数量少于 n_folds 时
        """
        if len(patient_ids) != len(severity_labels):
            raise ValueError(
                f"patient_ids 长度 ({len(patient_ids)}) 与 "
                f"severity_labels 长度 ({len(severity_labels)}) 不一致"
            )

        if len(patient_ids) < self.n_folds:
            raise ValueError(
                f"患者数量 ({len(patient_ids)}) 少于折数 ({self.n_folds})"
            )

        rng = random.Random(self.seed)

        # 按严重程度分组
        groups: Dict[int, List[str]] = defaultdict(list)
        for pid, sev in zip(patient_ids, severity_labels):
            groups[sev].append(pid)

        # 初始化各折的患者集合
        folds: List[List[str]] = [[] for _ in range(self.n_folds)]

        # 在每个严重程度组内打乱，然后round-robin分配到各折
        for sev in sorted(groups.keys()):
            patients_in_group = list(groups[sev])
            rng.shuffle(patients_in_group)
            for i, pid in enumerate(patients_in_group):
                folds[i % self.n_folds].append(pid)

        # 构建 (train_ids, test_ids) 元组
        all_ids = set(patient_ids)
        result: List[Tuple[List[str], List[str]]] = []
        for fold_idx in range(self.n_folds):
            test_ids = folds[fold_idx]
            train_ids = sorted(all_ids - set(test_ids))
            result.append((train_ids, test_ids))

        return result
