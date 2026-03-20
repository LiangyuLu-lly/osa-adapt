"""
睡眠分期评估器模块

计算 per-patient 和聚合级别的分期指标（accuracy, kappa, macro_f1, per-stage F1）。
支持按 OSA 严重程度子组（severity >= 2）单独计算指标。

Requirements: 5.1, 5.2, 5.3
"""

import numpy as np
from typing import Dict, List

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

import logging

logger = logging.getLogger(__name__)


class SleepStageEvaluator:
    """睡眠分期评估器，计算 per-patient 和聚合指标。"""

    STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

    def evaluate_patient(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        severity: int,
    ) -> Dict[str, float]:
        """计算单个患者的指标。

        Args:
            y_true: 真实标签数组，值域 0-4
            y_pred: 预测标签数组，值域 0-4
            severity: 患者 OSA 严重程度 (0=normal, 1=mild, 2=moderate, 3=severe)

        Returns:
            {accuracy, kappa, macro_f1, w_f1, n1_f1, n2_f1, n3_f1, rem_f1, severity}
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # 空预测数组：返回全零指标
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                "accuracy": 0.0,
                "kappa": 0.0,
                "macro_f1": 0.0,
                "w_f1": 0.0,
                "n1_f1": 0.0,
                "n2_f1": 0.0,
                "n3_f1": 0.0,
                "rem_f1": 0.0,
                "severity": float(severity),
            }

        accuracy = accuracy_score(y_true, y_pred)

        kappa_val = cohen_kappa_score(y_true, y_pred)
        kappa = float(kappa_val) if np.isfinite(kappa_val) else 0.0

        macro_f1 = f1_score(
            y_true, y_pred, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0
        )

        # per-stage F1，使用 labels=[0,1,2,3,4] 确保所有 5 个 stage 都有输出
        per_stage_f1 = f1_score(
            y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0
        )

        return {
            "accuracy": float(accuracy),
            "kappa": float(kappa),
            "macro_f1": float(macro_f1),
            "w_f1": float(per_stage_f1[0]),
            "n1_f1": float(per_stage_f1[1]),
            "n2_f1": float(per_stage_f1[2]),
            "n3_f1": float(per_stage_f1[3]),
            "rem_f1": float(per_stage_f1[4]),
            "severity": float(severity),
        }

    def evaluate_fold(
        self,
        patient_results: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """聚合一个 fold 的 per-patient 结果。

        Args:
            patient_results: evaluate_patient 返回的字典列表

        Returns:
            {acc, kappa, macro_f1, n1_f1,
             severe_acc, severe_n1_f1}
        """
        if not patient_results:
            return {
                "acc": 0.0,
                "kappa": 0.0,
                "macro_f1": 0.0,
                "n1_f1": 0.0,
                "severe_acc": 0.0,
                "severe_n1_f1": 0.0,
            }

        # 全体患者聚合
        acc = float(np.mean([r["accuracy"] for r in patient_results]))
        kappa = float(np.mean([r["kappa"] for r in patient_results]))
        macro_f1 = float(np.mean([r["macro_f1"] for r in patient_results]))
        n1_f1 = float(np.mean([r["n1_f1"] for r in patient_results]))

        # 重度子组 (severity >= 2)
        severe_results = [r for r in patient_results if r["severity"] >= 2]

        if severe_results:
            severe_acc = float(np.mean([r["accuracy"] for r in severe_results]))
            severe_n1_f1 = float(np.mean([r["n1_f1"] for r in severe_results]))
        else:
            severe_acc = 0.0
            severe_n1_f1 = 0.0

        return {
            "acc": acc,
            "kappa": kappa,
            "macro_f1": macro_f1,
            "n1_f1": n1_f1,
            "severe_acc": severe_acc,
            "severe_n1_f1": severe_n1_f1,
        }
