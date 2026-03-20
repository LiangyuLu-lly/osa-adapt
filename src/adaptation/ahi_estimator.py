"""
AHI估计器 (AHI Estimator) — 解决循环依赖问题

审稿人意见 #1: Severity Conditioner需要AHI作为输入，但AHI需要完成睡眠分期才能计算。
解决方案: Two-Pass Inference
  - Pass 1: 使用基础模型（无FiLM条件化）进行粗略分期，从分期结果估计AHI
  - Pass 2: 使用估计的AHI进行FiLM条件化分期

AHI估计方法:
  从睡眠分期结果中提取睡眠架构特征（N1比例、睡眠效率、觉醒指数等），
  通过线性回归模型估计AHI。该回归模型在训练集上拟合。
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AHIEstimator:
    """
    从睡眠分期结果估计AHI值。

    基于睡眠架构特征的线性回归模型:
    - N1比例 (N1%): OSA患者N1比例升高
    - 睡眠效率 (SE): OSA患者睡眠效率降低
    - Wake比例 (W%): OSA患者觉醒增多
    - N3比例 (N3%): 重度OSA患者深睡减少
    - REM比例 (REM%): OSA患者REM减少
    - 阶段转换频率 (transition_rate): OSA患者睡眠碎片化

    使用场景:
    - Two-Pass Inference的Pass 1: 从基础模型的粗略分期估计AHI
    - 当真实AHI不可用时的回退方案
    """

    # 睡眠阶段索引: W=0, N1=1, N2=2, N3=3, REM=4
    STAGE_W = 0
    STAGE_N1 = 1
    STAGE_N2 = 2
    STAGE_N3 = 3
    STAGE_REM = 4
    NUM_STAGES = 5

    def __init__(self):
        """初始化AHI估计器（未拟合状态）"""
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.is_fitted: bool = False
        # 特征标准化参数
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

    def extract_sleep_features(self, stage_predictions: np.ndarray) -> np.ndarray:
        """
        从睡眠分期预测中提取睡眠架构特征。

        Args:
            stage_predictions: 分期预测数组, shape [num_epochs], 值为0-4

        Returns:
            特征向量, shape [6]:
                [n1_ratio, sleep_efficiency, wake_ratio,
                 n3_ratio, rem_ratio, transition_rate]
        """
        total_epochs = len(stage_predictions)
        if total_epochs == 0:
            return np.zeros(6)

        # 各阶段比例
        stage_counts = np.bincount(
            stage_predictions.astype(int).clip(0, self.NUM_STAGES - 1),
            minlength=self.NUM_STAGES,
        )
        stage_ratios = stage_counts / total_epochs

        n1_ratio = stage_ratios[self.STAGE_N1]
        wake_ratio = stage_ratios[self.STAGE_W]
        n3_ratio = stage_ratios[self.STAGE_N3]
        rem_ratio = stage_ratios[self.STAGE_REM]

        # 睡眠效率 = 非Wake epoch / 总epoch
        sleep_efficiency = 1.0 - wake_ratio

        # 阶段转换频率 = 相邻epoch阶段不同的次数 / 总epoch数
        if total_epochs > 1:
            transitions = np.sum(stage_predictions[1:] != stage_predictions[:-1])
            transition_rate = transitions / (total_epochs - 1)
        else:
            transition_rate = 0.0

        return np.array([
            n1_ratio,
            sleep_efficiency,
            wake_ratio,
            n3_ratio,
            rem_ratio,
            transition_rate,
        ])

    def fit(
        self,
        stage_predictions_list: list,
        true_ahi_values: np.ndarray,
    ) -> Dict:
        """
        在训练集上拟合AHI估计模型。

        使用岭回归（L2正则化）从睡眠架构特征预测AHI。

        Args:
            stage_predictions_list: 每个患者的分期预测列表
            true_ahi_values: 对应的真实AHI值, shape [n_patients]

        Returns:
            拟合统计信息字典
        """
        n_patients = len(stage_predictions_list)
        if n_patients == 0:
            raise ValueError("训练集为空，无法拟合AHI估计器")

        # 提取特征矩阵
        features = np.array([
            self.extract_sleep_features(preds)
            for preds in stage_predictions_list
        ])  # [n_patients, 6]

        true_ahi = np.asarray(true_ahi_values, dtype=float)

        # 标准化特征
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        features_norm = (features - self.feature_mean) / self.feature_std

        # 岭回归: (X^T X + λI)^{-1} X^T y
        ridge_lambda = 1.0
        n_features = features_norm.shape[1]
        XtX = features_norm.T @ features_norm + ridge_lambda * np.eye(n_features)
        Xty = features_norm.T @ true_ahi
        self.coefficients = np.linalg.solve(XtX, Xty)
        self.intercept = true_ahi.mean() - features_norm.mean(axis=0) @ self.coefficients

        self.is_fitted = True

        # 计算训练集R²
        predicted = features_norm @ self.coefficients + self.intercept
        ss_res = np.sum((true_ahi - predicted) ** 2)
        ss_tot = np.sum((true_ahi - true_ahi.mean()) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)

        # 计算MAE
        mae = np.mean(np.abs(true_ahi - predicted))

        logger.info(
            "AHI估计器拟合完成: n=%d, R²=%.3f, MAE=%.1f",
            n_patients, r_squared, mae,
        )

        return {
            "n_patients": n_patients,
            "r_squared": float(r_squared),
            "mae": float(mae),
            "coefficients": self.coefficients.tolist(),
            "intercept": float(self.intercept),
        }

    def estimate(self, stage_predictions: np.ndarray) -> float:
        """
        从分期预测估计单个患者的AHI值。

        Args:
            stage_predictions: 分期预测数组, shape [num_epochs]

        Returns:
            估计的AHI值（下限截断为0）
        """
        if not self.is_fitted:
            raise RuntimeError("AHI估计器尚未拟合，请先调用fit()")

        features = self.extract_sleep_features(stage_predictions)
        features_norm = (features - self.feature_mean) / self.feature_std
        ahi_est = float(features_norm @ self.coefficients + self.intercept)

        # AHI不能为负
        return max(0.0, ahi_est)

    def estimate_batch(self, stage_predictions_list: list) -> np.ndarray:
        """
        批量估计AHI值。

        Args:
            stage_predictions_list: 每个患者的分期预测列表

        Returns:
            估计的AHI值数组, shape [n_patients]
        """
        return np.array([
            self.estimate(preds) for preds in stage_predictions_list
        ])

    def ahi_to_severity(self, ahi: float) -> int:
        """
        将AHI值转换为OSA严重程度类别。

        AASM标准:
        - Normal: AHI < 5 → 0
        - Mild: 5 ≤ AHI < 15 → 1
        - Moderate: 15 ≤ AHI < 30 → 2
        - Severe: AHI ≥ 30 → 3

        Args:
            ahi: AHI值

        Returns:
            严重程度类别 (0-3)
        """
        if ahi < 5:
            return 0
        elif ahi < 15:
            return 1
        elif ahi < 30:
            return 2
        else:
            return 3
