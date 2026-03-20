"""
RescueResultAnalyzer — 挽救实验结果分析器

统计分析与可视化：
- Wilcoxon 符号秩检验（配对比较）
- Bonferroni 校正（多重比较）
- Cohen's d 效应量 + 95% 置信区间
- JSON 序列化/反序列化（round-trip 一致性）
- LaTeX 表格生成

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import json
import logging
import math
import os
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.adaptation.statistical_tests import (
    bootstrap_ci,
    bonferroni_correction,
    cohens_d,
    wilcoxon_test,
)

logger = logging.getLogger(__name__)


def _cohens_d_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """计算 Cohen's d 及其 95% 置信区间。

    使用非中心 t 分布的近似公式：
        SE(d) ≈ sqrt(n1+n2)/(n1*n2) + d²/(2*(n1+n2-2))

    Args:
        group1: 第一组数据
        group2: 第二组数据
        confidence: 置信水平

    Returns:
        (d, ci_lower, ci_upper)
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    n1, n2 = len(group1), len(group2)

    if n1 + n2 <= 2 or n1 < 2 or n2 < 2:
        # 样本量不足以计算池化标准差或 CI，返回 0
        d = 0.0 if (n1 < 2 or n2 < 2) else cohens_d(group1, group2)
        return d, d, d

    d = cohens_d(group1, group2)

    # Hedges & Olkin (1985) 近似标准误
    se = math.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2)))

    from scipy import stats
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    return d, d - z * se, d + z * se


class RescueResultAnalyzer:
    """挽救实验结果分析器。

    接收实验结果列表（ExperimentResult 字典或对象），执行统计检验、
    生成 LaTeX 表格，并支持 JSON 序列化/反序列化。

    Attributes:
        results: 原始实验结果列表（字典形式）
    """

    # 核心指标名称
    METRICS = ["accuracy", "kappa", "macro_f1", "n1_f1"]

    # 指标显示名称（用于 LaTeX 表格）
    METRIC_LABELS = {
        "accuracy": "Accuracy",
        "kappa": "Cohen's $\\kappa$",
        "macro_f1": "Macro-F1",
        "n1_f1": "N1-F1",
    }

    def __init__(self, results: List[Union[Dict[str, Any], Any]]):
        """初始化分析器。

        支持两种输入格式：
        1. 字典列表 — 每个字典至少包含 ``config`` 和四个核心指标
        2. ExperimentResult 对象列表 — 自动转换为字典

        Args:
            results: ExperimentResult 字典或对象列表。每个条目至少包含
                     ``config`` (含 model_name, method, budget, fold, seed)
                     和四个核心指标 (accuracy, kappa, macro_f1, n1_f1)。
        """
        self.results: List[Dict[str, Any]] = self._normalize_results(results)

    @staticmethod
    def _normalize_results(
        results: List[Union[Dict[str, Any], Any]],
    ) -> List[Dict[str, Any]]:
        """将结果列表统一转换为字典列表。

        如果元素具有 ``to_dict`` 方法（如 ExperimentResult dataclass），
        则调用该方法；如果是 dataclass 则使用 ``asdict``；否则假定已是字典。

        Args:
            results: 混合类型的结果列表

        Returns:
            字典列表
        """
        normalized: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, dict):
                normalized.append(r)
            elif hasattr(r, "to_dict"):
                normalized.append(r.to_dict())
            else:
                try:
                    normalized.append(asdict(r))
                except TypeError:
                    # 最后的回退：直接使用 __dict__
                    normalized.append(vars(r))
        return normalized

    # ----------------------------------------------------------
    # 分组辅助
    # ----------------------------------------------------------

    def _group_by(
        self, keys: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """按指定键分组结果。

        Args:
            keys: config 中用于分组的键列表

        Returns:
            {group_key: [result_dicts]}
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in self.results:
            cfg = r.get("config", {})
            parts = [str(cfg.get(k, "")) for k in keys]
            key = "_".join(parts)
            groups.setdefault(key, []).append(r)
        return groups

    def _get_metric_values(
        self,
        result_list: List[Dict[str, Any]],
        metric: str,
    ) -> np.ndarray:
        """从结果列表中提取指定指标的值数组。"""
        return np.array([r.get(metric, 0.0) for r in result_list], dtype=float)

    # ----------------------------------------------------------
    # 统计检验 (Req 5.1, 5.2, 5.3)
    # ----------------------------------------------------------

    def statistical_tests(
        self,
        metric: str = "accuracy",
        group_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """执行所有统计检验。

        对每对方法进行 Wilcoxon 符号秩检验，应用 Bonferroni 校正，
        并计算 Cohen's d 效应量及 95% CI。

        Args:
            metric: 用于比较的指标名称
            group_keys: 分组键（默认 ["model_name", "method", "budget"]）

        Returns:
            {
                "pairwise": [
                    {
                        "group_a": str,
                        "group_b": str,
                        "wilcoxon_statistic": float,
                        "p_value": float,
                        "p_value_corrected": float,
                        "cohens_d": float,
                        "cohens_d_ci_lower": float,
                        "cohens_d_ci_upper": float,
                        "n_samples": int,
                    },
                    ...
                ],
                "n_comparisons": int,
                "metric": str,
                "correction_method": "bonferroni",
            }
        """
        if group_keys is None:
            group_keys = ["model_name", "method"]

        groups = self._group_by(group_keys)
        group_names = sorted(groups.keys())
        pairs = list(combinations(group_names, 2))
        n_comparisons = len(pairs)

        pairwise_results: List[Dict[str, Any]] = []
        raw_p_values: List[float] = []

        for ga, gb in pairs:
            vals_a = self._get_metric_values(groups[ga], metric)
            vals_b = self._get_metric_values(groups[gb], metric)

            # 对齐长度（取较短的）
            min_len = min(len(vals_a), len(vals_b))
            vals_a_paired = vals_a[:min_len]
            vals_b_paired = vals_b[:min_len]

            # Wilcoxon 符号秩检验 (Req 5.1)
            wt = wilcoxon_test(vals_a_paired, vals_b_paired)

            # Cohen's d + 95% CI (Req 5.3)
            d, d_lo, d_hi = _cohens_d_ci(vals_a, vals_b)

            raw_p_values.append(wt["p_value"])
            pairwise_results.append({
                "group_a": ga,
                "group_b": gb,
                "wilcoxon_statistic": wt["statistic"],
                "p_value": wt["p_value"],
                "p_value_corrected": 0.0,  # 稍后填充
                "cohens_d": d,
                "cohens_d_ci_lower": d_lo,
                "cohens_d_ci_upper": d_hi,
                "n_samples": min_len,
            })

        # Bonferroni 校正 (Req 5.2)
        if raw_p_values:
            corrected = bonferroni_correction(raw_p_values, n_comparisons)
            for i, pw in enumerate(pairwise_results):
                pw["p_value_corrected"] = corrected[i]

        return {
            "pairwise": pairwise_results,
            "n_comparisons": n_comparisons,
            "metric": metric,
            "correction_method": "bonferroni",
        }

    # ----------------------------------------------------------
    # LaTeX 表格生成 (Req 5.4)
    # ----------------------------------------------------------

    def generate_latex_tables(self, output_dir: str) -> None:
        """生成 LaTeX 格式的结果表格。

        生成两个表格文件：
        1. main_results.tex — 各方法在各预算下的核心指标（均值 ± 标准差）
        2. statistical_tests.tex — 配对统计检验结果

        Args:
            output_dir: 输出目录路径
        """
        os.makedirs(output_dir, exist_ok=True)
        self._generate_main_results_table(output_dir)
        self._generate_statistical_tests_table(output_dir)

    def _generate_main_results_table(self, output_dir: str) -> None:
        """生成主结果表格 (main_results.tex)。"""
        groups = self._group_by(["model_name", "method", "budget"])

        # 收集所有 (model, method, budget) 组合
        combos: List[Tuple[str, str, int]] = []
        for r in self.results:
            cfg = r.get("config", {})
            combo = (
                cfg.get("model_name", ""),
                cfg.get("method", ""),
                cfg.get("budget", 0),
            )
            if combo not in combos:
                combos.append(combo)
        combos.sort()

        # 构建 LaTeX
        n_metrics = len(self.METRICS)
        col_spec = "ll" + "r" * n_metrics
        header_cells = " & ".join(
            self.METRIC_LABELS.get(m, m) for m in self.METRICS
        )

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Main adaptation results (mean $\\pm$ std)}",
            "\\label{tab:main_results}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"Method & Budget & {header_cells} \\\\",
            "\\midrule",
        ]

        for model, method, budget in combos:
            key = f"{model}_{method}_{budget}"
            group = groups.get(key, [])
            cells = []
            for metric in self.METRICS:
                vals = self._get_metric_values(group, metric)
                if len(vals) > 0:
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    cells.append(f"{mean:.3f} $\\pm$ {std:.3f}")
                else:
                    cells.append("--")
            metric_str = " & ".join(cells)
            display_method = method.replace("_", "\\_")
            lines.append(f"{display_method} & {budget} & {metric_str} \\\\")

        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]

        path = os.path.join(output_dir, "main_results.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("LaTeX 主结果表格已保存: %s", path)

    def _generate_statistical_tests_table(self, output_dir: str) -> None:
        """生成统计检验表格 (statistical_tests.tex)。"""
        test_results = self.statistical_tests(metric="accuracy")
        pairwise = test_results["pairwise"]

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Pairwise statistical comparisons (Accuracy)}",
            "\\label{tab:stat_tests}",
            "\\begin{tabular}{llrrrrr}",
            "\\toprule",
            "Method A & Method B & $W$ & $p$ & $p_{\\text{corr}}$ "
            "& $d$ & 95\\% CI \\\\",
            "\\midrule",
        ]

        for pw in pairwise:
            ga = pw["group_a"].replace("_", "\\_")
            gb = pw["group_b"].replace("_", "\\_")
            w = pw["wilcoxon_statistic"]
            p = pw["p_value"]
            pc = pw["p_value_corrected"]
            d = pw["cohens_d"]
            d_lo = pw["cohens_d_ci_lower"]
            d_hi = pw["cohens_d_ci_upper"]

            # 格式化 p 值
            p_str = f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"
            pc_str = f"{pc:.4f}" if pc >= 0.0001 else f"{pc:.2e}"

            lines.append(
                f"{ga} & {gb} & {w:.1f} & {p_str} & {pc_str} "
                f"& {d:.3f} & [{d_lo:.3f}, {d_hi:.3f}] \\\\"
            )

        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]

        path = os.path.join(output_dir, "statistical_tests.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("LaTeX 统计检验表格已保存: %s", path)

    # ----------------------------------------------------------
    # JSON 序列化/反序列化 (Req 5.5, 5.6)
    # ----------------------------------------------------------

    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """递归地将对象转换为 JSON 可序列化的类型。

        处理 numpy 类型（int64, float64, ndarray 等）以确保
        JSON 序列化不会失败。

        Args:
            obj: 任意对象

        Returns:
            JSON 可序列化的对象
        """
        if isinstance(obj, dict):
            return {k: RescueResultAnalyzer._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [RescueResultAnalyzer._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def to_json(self) -> str:
        """将分析器状态（含所有结果）序列化为 JSON 字符串。

        确保所有 numpy 类型被正确转换为 Python 原生类型，
        以保证 round-trip 一致性 (Req 5.5)。

        Returns:
            JSON 字符串
        """
        serializable_results = self._make_json_serializable(self.results)
        payload = {"results": serializable_results}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "RescueResultAnalyzer":
        """从 JSON 字符串反序列化创建分析器。

        Args:
            json_str: JSON 字符串

        Returns:
            RescueResultAnalyzer 实例
        """
        payload = json.loads(json_str)
        return cls(results=payload["results"])

    def save_json(self, path: str) -> None:
        """将结果保存到 JSON 文件。

        Args:
            path: 输出文件路径
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        logger.info("结果已保存到 JSON: %s", path)

    @classmethod
    def load_json(cls, path: str) -> "RescueResultAnalyzer":
        """从 JSON 文件加载分析器。

        Args:
            path: JSON 文件路径

        Returns:
            RescueResultAnalyzer 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    # ----------------------------------------------------------
    # 图表生成 (Req 7.1, 7.2, 7.3, 7.4, 7.5)
    # ----------------------------------------------------------

    def generate_figures(self, output_dir: str) -> None:
        """生成论文级图表。

        委托给 scripts/rescue_generate_figures.py 中的图表生成函数。
        生成数据效率曲线、严重程度分层柱状图、混淆矩阵热力图、消融瀑布图。
        所有图表 300 DPI，PNG + PDF 双格式。

        Args:
            output_dir: 图表输出目录路径
        """
        from pathlib import Path

        # 延迟导入，避免循环依赖
        from scripts.rescue_generate_figures import (
            generate_all_figures,
            plot_ablation_waterfall,
            plot_confusion_matrix_heatmaps,
            plot_data_efficiency_curves,
            plot_severity_stratified_bar,
            setup_publication_style,
        )

        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir)

        setup_publication_style()

        logger.info("开始生成论文级图表，输出目录: %s", output_dir)

        # 使用分析器中已有的结果数据
        results = self.results

        # 图表 1: 数据效率曲线 (Req 7.2)
        logger.info("[1/4] 生成数据效率曲线...")
        plot_data_efficiency_curves(results, output_path)

        # 图表 2: 严重程度分层柱状图 (Req 7.3)
        logger.info("[2/4] 生成严重程度分层柱状图...")
        plot_severity_stratified_bar(results, output_path)

        # 图表 3: 混淆矩阵热力图 (Req 7.4)
        logger.info("[3/4] 生成混淆矩阵热力图...")
        plot_confusion_matrix_heatmaps(results, output_path)

        # 图表 4: 消融瀑布图 (Req 7.5)
        # 消融瀑布图需要 results_dir 来加载 ablation_summary.json
        # 默认使用 output_dir 的父目录作为 results_dir
        results_dir = output_path.parent
        logger.info("[4/4] 生成消融瀑布图...")
        plot_ablation_waterfall(results_dir, output_path, results)

        logger.info("所有论文级图表已保存到: %s", output_dir)
