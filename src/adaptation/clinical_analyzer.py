"""
临床影响分析器

评估域适应对临床相关指标的改善：
- TST估计误差 (Bland-Altman分析)
- OSA严重程度分级一致性 (Cohen's Kappa)
- N1召回率改善（按严重程度分组）
- 性别公平性改善

Requirements: 10.1, 10.2, 10.3, 10.4
"""

from typing import Dict, List

import numpy as np


class ClinicalAnalyzer:
    """
    临床影响分析

    评估域适应对临床相关指标的改善：
    - TST估计误差 (Bland-Altman分析)
    - OSA严重程度分级一致性
    - N1召回率改善
    - 性别公平性改善
    """

    def bland_altman_analysis(
        self,
        true_tst: np.ndarray,
        predicted_tst: np.ndarray,
    ) -> Dict:
        """
        Bland-Altman分析TST估计一致性

        Args:
            true_tst: 真实TST值数组
            predicted_tst: 预测TST值数组

        Returns:
            mean_diff: 均值偏差 = mean(predicted - true)
            std_diff: 标准差 (sample std, ddof=1)
            loa_upper: 95%一致性上界 (mean + 1.96*std)
            loa_lower: 95%一致性下界 (mean - 1.96*std)
            n_samples: 样本数
        """
        true_tst = np.asarray(true_tst, dtype=np.float64)
        predicted_tst = np.asarray(predicted_tst, dtype=np.float64)

        diff = predicted_tst - true_tst
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        return {
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "loa_upper": loa_upper,
            "loa_lower": loa_lower,
            "n_samples": len(diff),
        }

    def severity_classification_agreement(
        self,
        true_severity: np.ndarray,
        predicted_severity: np.ndarray,
    ) -> Dict:
        """
        OSA严重程度分级一致性 (Cohen's Kappa)

        手动实现Cohen's Kappa，不依赖sklearn。

        Args:
            true_severity: 真实严重程度标签数组
            predicted_severity: 预测严重程度标签数组

        Returns:
            kappa: Cohen's Kappa系数
            agreement_rate: 总体一致率
            confusion_matrix: 混淆矩阵 (list of lists)
        """
        true_severity = np.asarray(true_severity, dtype=np.int64)
        predicted_severity = np.asarray(predicted_severity, dtype=np.int64)

        n = len(true_severity)
        labels = np.union1d(true_severity, predicted_severity)
        n_labels = len(labels)

        # 构建标签到索引的映射
        label_to_idx = {label: i for i, label in enumerate(labels)}

        # 构建混淆矩阵
        cm = np.zeros((n_labels, n_labels), dtype=np.int64)
        for t, p in zip(true_severity, predicted_severity):
            cm[label_to_idx[t], label_to_idx[p]] += 1

        # 总体一致率
        agreement_rate = float(np.trace(cm)) / n

        # Cohen's Kappa
        row_sums = cm.sum(axis=1)  # 真实标签的边际分布
        col_sums = cm.sum(axis=0)  # 预测标签的边际分布
        expected_agreement = float(np.sum(row_sums * col_sums)) / (n * n)

        if expected_agreement == 1.0:
            kappa = 1.0
        else:
            kappa = (agreement_rate - expected_agreement) / (1.0 - expected_agreement)

        return {
            "kappa": float(kappa),
            "agreement_rate": float(agreement_rate),
            "confusion_matrix": cm.tolist(),
        }

    def n1_improvement_by_severity(
        self,
        results_before: Dict[str, Dict],
        results_after: Dict[str, Dict],
        severity_groups: Dict[int, List[str]],
    ) -> Dict:
        """
        按严重程度组计算N1召回率改善

        Args:
            results_before: 适应前结果，patient_id -> {"n1_recall": float, ...}
            results_after: 适应后结果，patient_id -> {"n1_recall": float, ...}
            severity_groups: severity_level -> list of patient_ids

        Returns:
            severity_level -> {"before": float, "after": float, "improvement": float}
        """
        result = {}
        for severity_level, patient_ids in severity_groups.items():
            before_recalls = []
            after_recalls = []
            for pid in patient_ids:
                if pid in results_before and pid in results_after:
                    before_recalls.append(results_before[pid]["n1_recall"])
                    after_recalls.append(results_after[pid]["n1_recall"])

            if before_recalls:
                mean_before = float(np.mean(before_recalls))
                mean_after = float(np.mean(after_recalls))
                improvement = mean_after - mean_before
            else:
                mean_before = 0.0
                mean_after = 0.0
                improvement = 0.0

            result[severity_level] = {
                "before": mean_before,
                "after": mean_after,
                "improvement": improvement,
            }

        return result

    def gender_fairness_analysis(
        self,
        results_before: Dict[str, Dict],
        results_after: Dict[str, Dict],
        gender_labels: Dict[str, int],
    ) -> Dict:
        """
        性别公平性分析：计算适应前后男女性能差距变化

        Args:
            results_before: 适应前结果，patient_id -> {"accuracy": float, ...}
            results_after: 适应后结果，patient_id -> {"accuracy": float, ...}
            gender_labels: patient_id -> 0 (female) 或 1 (male)

        Returns:
            before_gap: 适应前性别差距 |male_acc - female_acc|
            after_gap: 适应后性别差距
            gap_change: after_gap - before_gap (负值表示改善)
            per_gender: 每个性别的详细指标
        """
        # 按性别分组
        female_ids = [pid for pid, g in gender_labels.items() if g == 0]
        male_ids = [pid for pid, g in gender_labels.items() if g == 1]

        # 计算适应前各性别准确率
        before_female = [results_before[pid]["accuracy"] for pid in female_ids if pid in results_before]
        before_male = [results_before[pid]["accuracy"] for pid in male_ids if pid in results_before]

        # 计算适应后各性别准确率
        after_female = [results_after[pid]["accuracy"] for pid in female_ids if pid in results_after]
        after_male = [results_after[pid]["accuracy"] for pid in male_ids if pid in results_after]

        before_female_acc = float(np.mean(before_female)) if before_female else 0.0
        before_male_acc = float(np.mean(before_male)) if before_male else 0.0
        after_female_acc = float(np.mean(after_female)) if after_female else 0.0
        after_male_acc = float(np.mean(after_male)) if after_male else 0.0

        before_gap = abs(before_male_acc - before_female_acc)
        after_gap = abs(after_male_acc - after_female_acc)
        gap_change = after_gap - before_gap

        return {
            "before_gap": before_gap,
            "after_gap": after_gap,
            "gap_change": gap_change,
            "per_gender": {
                "female": {
                    "before_accuracy": before_female_acc,
                    "after_accuracy": after_female_acc,
                    "n_patients": len(before_female),
                },
                "male": {
                    "before_accuracy": before_male_acc,
                    "after_accuracy": after_male_acc,
                    "n_patients": len(before_male),
                },
            },
        }
