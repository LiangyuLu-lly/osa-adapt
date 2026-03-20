"""
统计检验模块

实现模型对比所需的统计检验：
- McNemar检验（分类任务对比）
- Wilcoxon符号秩检验（回归任务对比）
- Bonferroni校正
- Bootstrap置信区间
- Cohen's d效应大小

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import wilcoxon
from dataclasses import dataclass
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StatTestResult:
    """统计测试结果"""
    statistic: float
    p_value: float
    test_name: str
    significant: bool
    alpha: float = 0.05


@dataclass
class ConfidenceInterval:
    """置信区间"""
    lower: float
    upper: float
    mean: float
    confidence_level: float
    method: str


@dataclass
class EffectSize:
    """效应大小"""
    value: float
    interpretation: str
    method: str


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, float]:
    """
    McNemar检验：比较两个分类模型的性能差异。

    基于两个模型预测不一致的样本构建2x2列联表：
    - b: A正确 B错误的数量
    - c: A错误 B正确的数量
    统计量: chi2 = (|b - c| - 1)² / (b + c)

    Args:
        y_true: 真实标签 [N]
        y_pred_a: 模型A预测 [N]
        y_pred_b: 模型B预测 [N]

    Returns:
        {'statistic': float, 'p_value': float}
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A对B错, c: A错B对
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    if b + c == 0:
        return {'statistic': 0.0, 'p_value': 1.0}

    # McNemar with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    from scipy.stats import chi2 as chi2_dist
    p_value = float(1 - chi2_dist.cdf(chi2, df=1))

    return {'statistic': float(chi2), 'p_value': p_value}


def wilcoxon_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
) -> Dict[str, float]:
    """
    Wilcoxon符号秩检验：比较两个回归模型的误差分布。

    Args:
        errors_a: 模型A的绝对误差 [N]
        errors_b: 模型B的绝对误差 [N]

    Returns:
        {'statistic': float, 'p_value': float}
    """
    diff = errors_a - errors_b
    # 移除零差异
    nonzero = diff != 0
    if nonzero.sum() < 2:
        return {'statistic': 0.0, 'p_value': 1.0}

    try:
        stat, p_val = wilcoxon(diff[nonzero])
        return {'statistic': float(stat), 'p_value': float(p_val)}
    except Exception as e:
        logger.warning(f"Wilcoxon检验失败: {e}")
        return {'statistic': 0.0, 'p_value': 1.0}


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Bonferroni多重比较校正。

    校正后的显著性阈值 = alpha / n_tests
    校正后的p值 = min(p * n_tests, 1.0)

    Args:
        p_values: 原始p值列表
        alpha: 显著性水平

    Returns:
        {
            'corrected_p_values': List[float],
            'corrected_alpha': float,
            'significant_before': List[bool],
            'significant_after': List[bool],
            'n_significant_before': int,
            'n_significant_after': int,
        }
    """
    n = len(p_values)
    if n == 0:
        return {
            'corrected_p_values': [],
            'corrected_alpha': alpha,
            'significant_before': [],
            'significant_after': [],
            'n_significant_before': 0,
            'n_significant_after': 0,
        }

    corrected_alpha = alpha / n
    corrected_p = [min(p * n, 1.0) for p in p_values]
    sig_before = [p < alpha for p in p_values]
    sig_after = [p < corrected_alpha for p in p_values]

    return {
        'corrected_p_values': corrected_p,
        'corrected_alpha': corrected_alpha,
        'significant_before': sig_before,
        'significant_after': sig_after,
        'n_significant_before': sum(sig_before),
        'n_significant_after': sum(sig_after),
    }


class StatisticalValidator:
    """
    统计验证器 - 提供严格的统计验证和显著性测试
    
    实现功能：
    - Wilcoxon符号秩检验
    - Bootstrap置信区间计算
    - Cohen's d效应大小
    - Bonferroni多重比较校正
    - 配对和非配对统计测试
    """
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000):
        """
        初始化统计验证器
        
        Args:
            alpha: 显著性水平，默认0.05
            n_bootstrap: Bootstrap迭代次数，默认1000
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        
    def wilcoxon_signed_rank_test(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray,
        paired: bool = True
    ) -> StatTestResult:
        """
        Wilcoxon符号秩检验
        
        Args:
            group1: 第一组数据
            group2: 第二组数据  
            paired: 是否为配对数据，默认True
            
        Returns:
            StatTestResult: 包含统计量、p值等信息
        """
        try:
            if paired:
                # 配对Wilcoxon符号秩检验
                if len(group1) != len(group2):
                    raise ValueError("配对检验要求两组数据长度相同")
                
                diff = group1 - group2
                # 移除零差异
                nonzero_diff = diff[diff != 0]
                
                if len(nonzero_diff) < 2:
                    logger.warning("有效差异样本数量不足，返回默认结果")
                    return StatTestResult(
                        statistic=0.0,
                        p_value=1.0,
                        test_name="Wilcoxon Signed-Rank Test (Paired)",
                        significant=False,
                        alpha=self.alpha
                    )
                
                statistic, p_value = wilcoxon(nonzero_diff)
                
            else:
                # 非配对Wilcoxon秩和检验 (Mann-Whitney U)
                from scipy.stats import mannwhitneyu
                statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            
            test_name = "Wilcoxon Signed-Rank Test (Paired)" if paired else "Mann-Whitney U Test (Unpaired)"
            
            return StatTestResult(
                statistic=float(statistic),
                p_value=float(p_value),
                test_name=test_name,
                significant=p_value < self.alpha,
                alpha=self.alpha
            )
            
        except Exception as e:
            logger.error(f"Wilcoxon检验失败: {e}")
            return StatTestResult(
                statistic=0.0,
                p_value=1.0,
                test_name="Wilcoxon Test (Failed)",
                significant=False,
                alpha=self.alpha
            )
    
    def bootstrap_confidence_interval(
        self, 
        data: np.ndarray, 
        confidence: float = 0.95,
        statistic_func: callable = np.mean
    ) -> ConfidenceInterval:
        """
        Bootstrap置信区间计算
        
        Args:
            data: 输入数据
            confidence: 置信水平，默认0.95
            statistic_func: 统计量函数，默认为均值
            
        Returns:
            ConfidenceInterval: 置信区间信息
        """
        try:
            n = len(data)
            bootstrap_stats = []
            
            # Bootstrap重采样
            np.random.seed(42)  # 确保可重现性
            for _ in range(self.n_bootstrap):
                # 有放回抽样
                bootstrap_sample = np.random.choice(data, size=n, replace=True)
                bootstrap_stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(bootstrap_stat)
            
            bootstrap_stats = np.array(bootstrap_stats)
            
            # 计算置信区间
            alpha_level = 1 - confidence
            lower_percentile = (alpha_level / 2) * 100
            upper_percentile = (1 - alpha_level / 2) * 100
            
            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)
            mean_stat = np.mean(bootstrap_stats)
            
            return ConfidenceInterval(
                lower=float(ci_lower),
                upper=float(ci_upper),
                mean=float(mean_stat),
                confidence_level=confidence,
                method="Bootstrap"
            )
            
        except Exception as e:
            logger.error(f"Bootstrap置信区间计算失败: {e}")
            # 返回基于正态分布的近似置信区间
            mean_val = float(np.mean(data))
            std_val = float(np.std(data, ddof=1))
            margin = 1.96 * std_val / np.sqrt(len(data))  # 95%置信区间
            
            return ConfidenceInterval(
                lower=mean_val - margin,
                upper=mean_val + margin,
                mean=mean_val,
                confidence_level=confidence,
                method="Normal Approximation (Fallback)"
            )
    
    def cohens_d_effect_size(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray,
        pooled: bool = True
    ) -> EffectSize:
        """
        Cohen's d效应大小计算
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            pooled: 是否使用合并标准差，默认True
            
        Returns:
            EffectSize: 效应大小信息
        """
        try:
            mean1 = np.mean(group1)
            mean2 = np.mean(group2)
            
            if pooled:
                # 合并标准差
                n1, n2 = len(group1), len(group2)
                var1 = np.var(group1, ddof=1)
                var2 = np.var(group2, ddof=1)
                pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                d = (mean1 - mean2) / pooled_std
            else:
                # 使用第一组的标准差
                std1 = np.std(group1, ddof=1)
                d = (mean1 - mean2) / std1
            
            # 效应大小解释
            abs_d = abs(d)
            if abs_d < 0.2:
                interpretation = "negligible"  # 可忽略
            elif abs_d < 0.5:
                interpretation = "small"      # 小效应
            elif abs_d < 0.8:
                interpretation = "medium"     # 中等效应
            else:
                interpretation = "large"      # 大效应
            
            return EffectSize(
                value=float(d),
                interpretation=interpretation,
                method="Cohen's d (Pooled)" if pooled else "Cohen's d (Group 1 SD)"
            )
            
        except Exception as e:
            logger.error(f"Cohen's d计算失败: {e}")
            return EffectSize(
                value=0.0,
                interpretation="unknown",
                method="Cohen's d (Failed)"
            )
    
    def bonferroni_correction(self, p_values: List[float]) -> Dict[str, Union[List[float], float, int]]:
        """
        Bonferroni多重比较校正
        
        Args:
            p_values: 原始p值列表
            
        Returns:
            Dict: 校正结果
        """
        return bonferroni_correction(p_values, self.alpha)
    
    def run_comprehensive_comparison(
        self,
        method_results: Dict[str, np.ndarray],
        baseline_method: str = None
    ) -> Dict[str, Dict]:
        """
        运行综合统计比较
        
        Args:
            method_results: 方法名称到结果数组的映射
            baseline_method: 基线方法名称，如果为None则使用第一个方法
            
        Returns:
            Dict: 综合比较结果
        """
        methods = list(method_results.keys())
        if len(methods) < 2:
            raise ValueError("至少需要两个方法进行比较")
        
        if baseline_method is None:
            baseline_method = methods[0]
        
        if baseline_method not in methods:
            raise ValueError(f"基线方法 {baseline_method} 不在方法列表中")
        
        baseline_data = method_results[baseline_method]
        comparison_results = {}
        p_values = []
        
        # 与基线方法进行配对比较
        for method in methods:
            if method == baseline_method:
                continue
                
            method_data = method_results[method]
            
            # Wilcoxon符号秩检验
            wilcoxon_result = self.wilcoxon_signed_rank_test(method_data, baseline_data, paired=True)
            
            # Cohen's d效应大小
            effect_size = self.cohens_d_effect_size(method_data, baseline_data)
            
            # Bootstrap置信区间
            diff_data = method_data - baseline_data
            ci_diff = self.bootstrap_confidence_interval(diff_data)
            
            comparison_results[f"{method}_vs_{baseline_method}"] = {
                'wilcoxon_test': wilcoxon_result,
                'effect_size': effect_size,
                'difference_ci': ci_diff,
                'method_mean': float(np.mean(method_data)),
                'baseline_mean': float(np.mean(baseline_data)),
                'mean_difference': float(np.mean(diff_data))
            }
            
            p_values.append(wilcoxon_result.p_value)
        
        # Bonferroni校正
        if p_values:
            bonferroni_results = self.bonferroni_correction(p_values)
            comparison_results['multiple_comparison_correction'] = bonferroni_results
        
        return comparison_results
    
    def generate_statistical_report(
        self,
        comparison_results: Dict,
        output_path: str = "results/statistical_significance.json"
    ) -> Dict:
        """
        生成统计显著性报告
        
        Args:
            comparison_results: 比较结果
            output_path: 输出文件路径
            
        Returns:
            Dict: 完整的统计报告
        """
        report = {
            'metadata': {
                'analyzer': 'StatisticalValidator',
                'alpha_level': self.alpha,
                'n_bootstrap': self.n_bootstrap,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            },
            'statistical_tests': {},
            'effect_sizes': {},
            'confidence_intervals': {},
            'multiple_comparison_correction': {},
            'summary': {}
        }
        
        # 整理结果
        significant_comparisons = 0
        total_comparisons = 0
        
        for comparison_name, results in comparison_results.items():
            if comparison_name == 'multiple_comparison_correction':
                report['multiple_comparison_correction'] = results
                continue
            
            # 统计测试结果
            wilcoxon_result = results['wilcoxon_test']
            report['statistical_tests'][comparison_name] = {
                'test_name': wilcoxon_result.test_name,
                'statistic': wilcoxon_result.statistic,
                'p_value': wilcoxon_result.p_value,
                'significant': wilcoxon_result.significant,
                'alpha': wilcoxon_result.alpha
            }
            
            # 效应大小
            effect_size = results['effect_size']
            report['effect_sizes'][comparison_name] = {
                'cohens_d': effect_size.value,
                'interpretation': effect_size.interpretation,
                'method': effect_size.method
            }
            
            # 置信区间
            ci_diff = results['difference_ci']
            report['confidence_intervals'][comparison_name] = {
                'difference_ci_lower': ci_diff.lower,
                'difference_ci_upper': ci_diff.upper,
                'difference_mean': ci_diff.mean,
                'confidence_level': ci_diff.confidence_level,
                'method': ci_diff.method
            }
            
            # 统计汇总
            if wilcoxon_result.significant:
                significant_comparisons += 1
            total_comparisons += 1
        
        # 生成汇总
        report['summary'] = {
            'total_comparisons': total_comparisons,
            'significant_comparisons': significant_comparisons,
            'significance_rate': significant_comparisons / total_comparisons if total_comparisons > 0 else 0,
            'alpha_level': self.alpha,
            'bonferroni_corrected_alpha': self.alpha / total_comparisons if total_comparisons > 0 else self.alpha
        }
        
        # 保存报告
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 自定义JSON编码器处理dataclass
        def json_encoder(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=json_encoder)
        
        logger.info(f"统计显著性报告已保存到: {output_path}")
        return report
