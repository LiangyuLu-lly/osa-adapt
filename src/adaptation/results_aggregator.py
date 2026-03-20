"""
结果聚合器模块

将多次运行（5 fold × 5 seed）的结果聚合为论文格式的 JSON，
输出与现有 main_results_summary.json 格式兼容。

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import json
import logging
import os
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# evaluate_fold 输出的基础指标键
BASE_METRICS = ["acc", "kappa", "macro_f1", "n1_f1", "severe_acc", "severe_n1_f1"]


class ResultsAggregator:
    """将多次运行结果聚合为论文格式的 JSON。"""

    def aggregate(
        self,
        all_run_results: List[Dict],
        no_adapt_baseline: Dict[str, float],
    ) -> Dict[str, float]:
        """聚合 25 次运行（5 fold × 5 seed）的结果。

        Args:
            all_run_results: 每次运行的指标字典列表，
                每个字典包含 {acc, kappa, macro_f1, n1_f1, severe_acc, severe_n1_f1}
            no_adapt_baseline: no_adapt 方法的聚合指标（用于计算 delta）

        Returns:
            {acc, acc_std, kappa, macro_f1, n1_f1,
             acc_delta, n1_f1_delta,
             severe_acc, severe_n1_f1,
             severe_n1_f1_delta, severe_acc_delta}
        """
        if not all_run_results:
            return {
                "acc": 0.0, "acc_std": 0.0, "kappa": 0.0, "macro_f1": 0.0,
                "n1_f1": 0.0, "acc_delta": 0.0, "n1_f1_delta": 0.0,
                "severe_acc": 0.0, "severe_n1_f1": 0.0,
                "severe_n1_f1_delta": 0.0, "severe_acc_delta": 0.0,
            }

        # 计算每个指标的均值
        means = {}
        for key in BASE_METRICS:
            values = [r[key] for r in all_run_results]
            means[key] = float(np.mean(values))

        # acc 的标准差
        acc_std = float(np.std([r["acc"] for r in all_run_results]))

        # delta = method - baseline
        acc_delta = means["acc"] - no_adapt_baseline.get("acc", 0.0)
        n1_f1_delta = means["n1_f1"] - no_adapt_baseline.get("n1_f1", 0.0)
        severe_acc_delta = means["severe_acc"] - no_adapt_baseline.get("severe_acc", 0.0)
        severe_n1_f1_delta = means["severe_n1_f1"] - no_adapt_baseline.get("severe_n1_f1", 0.0)

        return {
            "acc": means["acc"],
            "acc_std": acc_std,
            "kappa": means["kappa"],
            "macro_f1": means["macro_f1"],
            "n1_f1": means["n1_f1"],
            "acc_delta": acc_delta,
            "n1_f1_delta": n1_f1_delta,
            "severe_acc": means["severe_acc"],
            "severe_n1_f1": means["severe_n1_f1"],
            "severe_n1_f1_delta": severe_n1_f1_delta,
            "severe_acc_delta": severe_acc_delta,
        }

    def aggregate_all_methods(
        self,
        results_by_method: Dict[str, List[Dict]],
    ) -> Dict[str, Dict[str, float]]:
        """聚合所有方法的结果，自动计算 delta。

        先聚合 no_adapt 获取基线，再聚合其他方法并计算 delta。

        Args:
            results_by_method: {method_name: [run_results_list]}

        Returns:
            {method_name: {acc, acc_std, kappa, ..., acc_delta, ...}}
        """
        # 先聚合 no_adapt 基线（delta 为 0）
        no_adapt_results = results_by_method.get("no_adapt", [])
        zero_baseline = {k: 0.0 for k in BASE_METRICS}
        no_adapt_aggregated = self.aggregate(no_adapt_results, zero_baseline)

        # 提取基线均值用于其他方法的 delta 计算
        baseline = {
            "acc": no_adapt_aggregated["acc"],
            "n1_f1": no_adapt_aggregated["n1_f1"],
            "severe_acc": no_adapt_aggregated["severe_acc"],
            "severe_n1_f1": no_adapt_aggregated["severe_n1_f1"],
        }

        aggregated = {"no_adapt": no_adapt_aggregated}

        for method, runs in results_by_method.items():
            if method == "no_adapt":
                continue
            aggregated[method] = self.aggregate(runs, baseline)

        return aggregated

    def save_json(
        self,
        aggregated: Dict[str, Dict[str, Dict]],
        output_path: str,
    ) -> None:
        """保存为与现有格式兼容的 JSON。

        Args:
            aggregated: {model_name: {method_name: {metrics...}}}
            output_path: 输出文件路径
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)

        logger.info(f"结果已保存到 {output_path}")

    def aggregate_per_budget(
        self,
        results_by_budget: Dict[int, List[Dict]],
        no_adapt_baseline: Dict[str, float],
    ) -> Dict[int, Dict[str, float]]:
        """按 budget 分组聚合，用于数据效率曲线。

        Args:
            results_by_budget: {budget: [run_results_list]}
            no_adapt_baseline: no_adapt 基线指标

        Returns:
            {budget: {acc, acc_std, kappa, ..., acc_delta, ...}}
        """
        aggregated = {}
        for budget, runs in results_by_budget.items():
            aggregated[budget] = self.aggregate(runs, no_adapt_baseline)
        return aggregated
