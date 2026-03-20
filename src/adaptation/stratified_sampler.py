"""
严重程度分层少样本采样器

按OSA严重程度（Normal/Mild/Moderate/Severe）进行分层采样，
确保各组在适应集中的比例与总体比例一致。

Requirements: 5.1, 5.2, 5.3
"""

from typing import List, Dict, Tuple
import random
from collections import Counter, defaultdict


class SeverityStratifiedFewShotSampler:
    """
    严重程度分层少样本采样器

    按OSA严重程度（Normal/Mild/Moderate/Severe）进行分层采样，
    确保各组在适应集中的比例与总体比例一致。

    使用固定随机种子保证可复现性（Req 5.3）。
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def sample(
        self,
        patient_ids: List[str],
        severity_labels: List[int],  # 0-3
        budget: int,
    ) -> List[str]:
        """
        从候选集中按严重程度分层选择指定数量的患者。

        Args:
            patient_ids: 候选患者ID列表
            severity_labels: 对应的严重程度标签 (0=Normal, 1=Mild, 2=Moderate, 3=Severe)
            budget: 要选择的患者总数

        Returns:
            选中的患者ID列表

        Raises:
            ValueError: 当 patient_ids 和 severity_labels 长度不一致时
        """
        if len(patient_ids) != len(severity_labels):
            raise ValueError(
                f"patient_ids 长度 ({len(patient_ids)}) 与 "
                f"severity_labels 长度 ({len(severity_labels)}) 不一致"
            )

        if budget <= 0:
            return []

        # 候选不足时返回全部
        if budget >= len(patient_ids):
            return list(patient_ids)

        rng = random.Random(self.seed)

        # 按严重程度分组
        groups: Dict[int, List[str]] = defaultdict(list)
        for pid, sev in zip(patient_ids, severity_labels):
            groups[sev].append(pid)

        # 存在的组
        existing_groups = sorted(groups.keys())
        n_groups = len(existing_groups)

        if n_groups == 0:
            return []

        total = len(patient_ids)

        # 计算每组的分配数量
        allocation = self._allocate(groups, existing_groups, total, budget)

        # 从每组中随机采样
        selected: List[str] = []
        for sev in existing_groups:
            group_patients = groups[sev]
            n_select = min(allocation[sev], len(group_patients))
            rng.shuffle(group_patients)
            selected.extend(group_patients[:n_select])

        return selected

    def _allocate(
        self,
        groups: Dict[int, List[str]],
        existing_groups: List[int],
        total: int,
        budget: int,
    ) -> Dict[int, int]:
        """
        计算每组的分配数量。

        策略（Req 5.1 + 5.2）：
        1. 按比例分配（Req 5.1）
        2. 如果预算不足以每组至少2人，则先保证每组至少1人，
           剩余按比例分配（Req 5.2）
        """
        n_groups = len(existing_groups)
        allocation: Dict[int, int] = {}

        # 每组的比例
        group_sizes = {sev: len(groups[sev]) for sev in existing_groups}

        # 判断是否预算不足：budget < n_groups * 2
        min_per_group_target = 2
        budget_tight = budget < n_groups * min_per_group_target

        if budget_tight:
            # Req 5.2: 先保证每组至少1人
            guaranteed = min(n_groups, budget)
            for i, sev in enumerate(existing_groups):
                if i < guaranteed:
                    allocation[sev] = 1
                else:
                    allocation[sev] = 0

            remainder = budget - guaranteed
            if remainder > 0:
                # 剩余按比例分配
                self._distribute_remainder(
                    allocation, groups, existing_groups, total, remainder
                )
        else:
            # Req 5.1: 按比例分配
            # 使用最大余数法（Largest Remainder Method）保证总数精确
            raw_alloc = {}
            for sev in existing_groups:
                raw_alloc[sev] = (group_sizes[sev] / total) * budget

            # 先取整数部分
            for sev in existing_groups:
                allocation[sev] = int(raw_alloc[sev])

            # 分配剩余
            allocated_so_far = sum(allocation.values())
            remainder = budget - allocated_so_far

            if remainder > 0:
                # 按小数部分降序分配
                fractional = {
                    sev: raw_alloc[sev] - int(raw_alloc[sev])
                    for sev in existing_groups
                }
                sorted_by_frac = sorted(
                    existing_groups,
                    key=lambda s: fractional[s],
                    reverse=True,
                )
                for i in range(remainder):
                    sev = sorted_by_frac[i % len(sorted_by_frac)]
                    allocation[sev] += 1

        # 确保不超过每组实际人数
        for sev in existing_groups:
            allocation[sev] = min(allocation[sev], len(groups[sev]))

        # 如果因为组人数限制导致总数不足，将剩余分配给有余量的组
        current_total = sum(allocation.values())
        if current_total < budget:
            deficit = budget - current_total
            for sev in existing_groups:
                if deficit <= 0:
                    break
                available = len(groups[sev]) - allocation[sev]
                add = min(available, deficit)
                allocation[sev] += add
                deficit -= add

        return allocation

    def _distribute_remainder(
        self,
        allocation: Dict[int, int],
        groups: Dict[int, List[str]],
        existing_groups: List[int],
        total: int,
        remainder: int,
    ) -> None:
        """将剩余预算按比例分配给各组。"""
        group_sizes = {sev: len(groups[sev]) for sev in existing_groups}

        # 按比例计算每组应得的额外份额
        raw_extra = {}
        for sev in existing_groups:
            raw_extra[sev] = (group_sizes[sev] / total) * remainder

        # 最大余数法
        int_extra = {sev: int(raw_extra[sev]) for sev in existing_groups}
        for sev in existing_groups:
            allocation[sev] += int_extra[sev]

        still_remaining = remainder - sum(int_extra.values())
        if still_remaining > 0:
            fractional = {
                sev: raw_extra[sev] - int_extra[sev]
                for sev in existing_groups
            }
            sorted_by_frac = sorted(
                existing_groups,
                key=lambda s: fractional[s],
                reverse=True,
            )
            for i in range(still_remaining):
                sev = sorted_by_frac[i % len(sorted_by_frac)]
                allocation[sev] += 1
