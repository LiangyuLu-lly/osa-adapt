"""
统计检验模块 — OSA-Adapt框架

实现论文所需的统计检验功能：
- Bootstrap 95% 置信区间（百分位法）
- Wilcoxon 符号秩检验（配对比较）
- Bonferroni 多重比较校正
- Cohen's d 效应量

Requirements: 11.1, 11.2, 11.3, 11.4
"""

import numpy as np
from typing import Callable, Dict, List, Optional
from scipy.stats import wilcoxon
import logging

logger = logging.getLogger(__name__)


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap 置信区间（百分位法）。

    对数据进行有放回重采样，计算统计量的分布，
    然后取百分位数作为置信区间边界。

    Args:
        data: 输入数据数组
        statistic_fn: 统计量函数，默认为 np.mean
        n_bootstrap: 重采样次数，默认 1000
        confidence: 置信水平，默认 0.95
        seed: 随机种子

    Returns:
        {"estimate": float, "ci_lower": float, "ci_upper": float, "n_bootstrap": int}
    """
    data = np.asarray(data)
    rng = np.random.RandomState(seed)
    n = len(data)

    estimate = float(statistic_fn(data))

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample = data[idx]
        try:
            stat = statistic_fn(sample)
            if np.isfinite(stat):
                bootstrap_stats.append(stat)
        except Exception:
            continue

    if not bootstrap_stats:
        return {
            "estimate": estimate,
            "ci_lower": estimate,
            "ci_upper": estimate,
            "n_bootstrap": n_bootstrap,
        }

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
    }


def wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict:
    """
    Wilcoxon 符号秩检验（配对样本）。

    用于比较两组配对观测值是否存在显著差异。

    Args:
        x: 第一组观测值
        y: 第二组观测值（与 x 等长）

    Returns:
        {"statistic": float, "p_value": float, "n_samples": int}
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n_samples = len(x)

    diff = x - y
    nonzero = diff != 0

    if nonzero.sum() < 2:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_samples": n_samples,
        }

    try:
        stat, p_val = wilcoxon(diff[nonzero])
        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "n_samples": n_samples,
        }
    except Exception as e:
        logger.warning(f"Wilcoxon检验失败: {e}")
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_samples": n_samples,
        }


def bonferroni_correction(
    p_values: List[float],
    n_comparisons: Optional[int] = None,
) -> List[float]:
    """
    Bonferroni 多重比较校正。

    corrected_p = min(original_p × n_comparisons, 1.0)

    Args:
        p_values: 原始 p 值列表
        n_comparisons: 比较次数。若为 None，使用 len(p_values)

    Returns:
        校正后的 p 值列表
    """
    if n_comparisons is None:
        n_comparisons = len(p_values)

    return [min(p * n_comparisons, 1.0) for p in p_values]


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """
    Cohen's d 效应量。

    d = (mean1 - mean2) / pooled_std
    pooled_std = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))

    Args:
        group1: 第一组数据
        group2: 第二组数据

    Returns:
        Cohen's d 值
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    n1 = len(group1)
    n2 = len(group2)

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # 池化标准差
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


# ================================================================
# 患者级配对统计检验（审稿人意见 #3）
# ================================================================
#
# 审稿人指出：使用25次运行（5-fold × 5-seed）的Wilcoxon检验存在
# 独立性问题，因为同一患者在不同seed下的结果高度相关。
# 正确做法：在患者级别（N=328）进行配对检验。
# 每个患者有一个"适应前准确率"和一个"适应后准确率"，
# 这两个值是自然配对的。


def patient_level_wilcoxon(
    patient_metrics_before: np.ndarray,
    patient_metrics_after: np.ndarray,
) -> Dict:
    """
    患者级Wilcoxon符号秩检验。

    对每个患者的适应前后指标进行配对比较（N=患者数）。
    这比运行级检验（N=25）更合理，因为：
    1. 患者间的观测是独立的
    2. 样本量更大（N=328 vs N=25），统计功效更高
    3. 避免了同一患者在不同fold/seed中重复出现的伪复制问题

    Args:
        patient_metrics_before: 每个患者的适应前指标, shape [n_patients]
        patient_metrics_after: 每个患者的适应后指标, shape [n_patients]

    Returns:
        包含检验结果的字典:
            statistic: Wilcoxon统计量
            p_value: p值
            n_patients: 患者数
            mean_improvement: 平均改善量
            median_improvement: 中位改善量
            pct_improved: 改善的患者比例
    """
    before = np.asarray(patient_metrics_before)
    after = np.asarray(patient_metrics_after)
    n_patients = len(before)

    improvement = after - before
    mean_improvement = float(np.mean(improvement))
    median_improvement = float(np.median(improvement))
    pct_improved = float(np.mean(improvement > 0) * 100)

    # Wilcoxon检验
    result = wilcoxon_test(after, before)

    return {
        "statistic": result["statistic"],
        "p_value": result["p_value"],
        "n_patients": n_patients,
        "mean_improvement": mean_improvement,
        "median_improvement": median_improvement,
        "pct_improved": pct_improved,
    }


def patient_level_bootstrap_comparison(
    patient_metrics_before: np.ndarray,
    patient_metrics_after: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    患者级Bootstrap配对比较。

    对适应前后的改善量（after - before）计算Bootstrap置信区间。
    如果CI不包含0，则改善具有统计显著性。

    Args:
        patient_metrics_before: 每个患者的适应前指标
        patient_metrics_after: 每个患者的适应后指标
        n_bootstrap: Bootstrap重采样次数
        confidence: 置信水平
        seed: 随机种子

    Returns:
        包含Bootstrap CI的字典
    """
    improvement = np.asarray(patient_metrics_after) - np.asarray(patient_metrics_before)

    ci_result = bootstrap_ci(
        improvement,
        statistic_fn=np.mean,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    ci_result["significant"] = (
        ci_result["ci_lower"] > 0 or ci_result["ci_upper"] < 0
    )

    return ci_result
