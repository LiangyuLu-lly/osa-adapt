"""
消融实验运行器 — AblationRunner

系统性移除OSA-Adapt框架的各个组件，评估每个组件的贡献。
支持的消融组件：
- 严重程度条件化（退化为标准FiLM）
- N1感知损失（退化为标准交叉熵）
- 分层采样（退化为随机采样）
- 渐进式适应（退化为直接微调，跳过Phase 1）

Requirements: 8.1, 8.2, 8.3
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .statistical_tests import wilcoxon_test, cohens_d

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """消融实验结果"""

    full_model_metrics: Dict[str, float]
    ablation_metrics: Dict[str, Dict[str, float]]  # component_name -> metrics
    statistical_tests: Dict[str, Dict]  # component_name -> {p_value, effect_size, ...}


class AblationRunner:
    """消融实验运行器

    系统性移除OSA-Adapt的各个组件，评估每个组件的贡献。
    对每种消融配置，与完整模型进行Wilcoxon符号秩检验比较。
    """

    ABLATION_COMPONENTS = [
        "no_severity_conditioning",
        "no_n1_loss",
        "no_stratified_sampling",
        "no_progressive_adaptation",
    ]

    # 消融组件的可读名称（用于表格展示）
    COMPONENT_DISPLAY_NAMES = {
        "no_severity_conditioning": "w/o Severity Conditioning",
        "no_n1_loss": "w/o N1-Aware Loss",
        "no_stratified_sampling": "w/o Stratified Sampling",
        "no_progressive_adaptation": "w/o Progressive Adaptation",
    }

    def __init__(self, output_dir: str = "experiments/paper3_osa_adapt/ablation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def define_ablations(self) -> Dict[str, Dict]:
        """定义每种消融配置的参数修改。

        Returns:
            字典，键为消融组件名，值为该消融对应的配置修改。
        """
        return {
            "no_severity_conditioning": {
                "description": "移除严重程度条件化，使用固定零向量代替条件向量",
                "use_severity_conditioning": False,
                "use_n1_loss": True,
                "use_stratified_sampling": True,
                "use_progressive_adaptation": True,
            },
            "no_n1_loss": {
                "description": "移除N1感知损失，使用标准交叉熵损失",
                "use_severity_conditioning": True,
                "use_n1_loss": False,
                "use_stratified_sampling": True,
                "use_progressive_adaptation": True,
            },
            "no_stratified_sampling": {
                "description": "移除分层采样，使用随机采样",
                "use_severity_conditioning": True,
                "use_n1_loss": True,
                "use_stratified_sampling": False,
                "use_progressive_adaptation": True,
            },
            "no_progressive_adaptation": {
                "description": "移除渐进式适应，跳过Phase 1直接微调",
                "use_severity_conditioning": True,
                "use_n1_loss": True,
                "use_stratified_sampling": True,
                "use_progressive_adaptation": False,
            },
        }

    def run_ablation_study(
        self,
        full_model_results: Dict[str, float],
        ablation_results: Dict[str, Dict[str, float]],
    ) -> AblationResult:
        """分析消融结果，执行统计检验。

        对每种消融配置，将其指标与完整模型进行比较，
        使用Wilcoxon符号秩检验评估差异显著性，并计算Cohen's d效应量。

        Args:
            full_model_results: 完整模型的指标字典，
                键为指标名（如 "accuracy", "kappa"），值为浮点数或数组。
                若值为数组（多折/多种子结果），则执行配对检验。
            ablation_results: 各消融配置的指标字典，
                键为消融组件名，值为该消融的指标字典。

        Returns:
            AblationResult 包含完整模型指标、消融指标和统计检验结果。
        """
        statistical_tests = {}

        for component_name, abl_metrics in ablation_results.items():
            component_stats = {}

            # 找到完整模型和消融模型共有的指标
            common_metrics = set(full_model_results.keys()) & set(abl_metrics.keys())

            for metric_name in sorted(common_metrics):
                full_val = full_model_results[metric_name]
                abl_val = abl_metrics[metric_name]

                full_arr = np.atleast_1d(np.asarray(full_val, dtype=float))
                abl_arr = np.atleast_1d(np.asarray(abl_val, dtype=float))

                # 计算差值（完整模型 - 消融模型）
                if len(full_arr) == len(abl_arr) and len(full_arr) >= 2:
                    # 配对样本：执行Wilcoxon检验
                    test_result = wilcoxon_test(full_arr, abl_arr)
                    effect = cohens_d(full_arr, abl_arr)
                    component_stats[metric_name] = {
                        "full_mean": float(np.mean(full_arr)),
                        "ablation_mean": float(np.mean(abl_arr)),
                        "difference": float(np.mean(full_arr) - np.mean(abl_arr)),
                        "p_value": test_result["p_value"],
                        "statistic": test_result["statistic"],
                        "effect_size": effect,
                        "n_samples": test_result["n_samples"],
                    }
                else:
                    # 标量或长度不匹配：仅计算差值，不做检验
                    full_mean = float(np.mean(full_arr))
                    abl_mean = float(np.mean(abl_arr))
                    component_stats[metric_name] = {
                        "full_mean": full_mean,
                        "ablation_mean": abl_mean,
                        "difference": full_mean - abl_mean,
                        "p_value": None,
                        "statistic": None,
                        "effect_size": None,
                        "n_samples": max(len(full_arr), len(abl_arr)),
                    }

            statistical_tests[component_name] = component_stats

        return AblationResult(
            full_model_metrics=full_model_results,
            ablation_metrics=ablation_results,
            statistical_tests=statistical_tests,
        )

    def generate_ablation_table(self, result: AblationResult) -> str:
        """生成消融对比的LaTeX表格。

        表格包含完整模型和各消融配置的指标对比，
        以及统计检验结果（p值和效应量）。

        Args:
            result: AblationResult 消融实验结果。

        Returns:
            LaTeX格式的表格字符串。
        """
        # 收集所有指标名
        all_metrics = set()
        for abl_metrics in result.ablation_metrics.values():
            all_metrics.update(abl_metrics.keys())
        all_metrics = sorted(all_metrics)

        if not all_metrics:
            return "% 无消融结果数据\n"

        # 构建LaTeX表格
        n_metrics = len(all_metrics)
        col_spec = "l" + "c" * n_metrics
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]

        # 表头
        header_cells = ["Configuration"] + [
            self._format_metric_name(m) for m in all_metrics
        ]
        lines.append(" & ".join(header_cells) + r" \\")
        lines.append(r"\midrule")

        # 完整模型行
        full_cells = ["Full Model (OSA-Adapt)"]
        for metric in all_metrics:
            val = result.full_model_metrics.get(metric)
            full_cells.append(self._format_value(val))
        lines.append(" & ".join(full_cells) + r" \\")
        lines.append(r"\midrule")

        # 各消融行
        for component in self.ABLATION_COMPONENTS:
            if component not in result.ablation_metrics:
                continue

            display_name = self.COMPONENT_DISPLAY_NAMES.get(component, component)
            row_cells = [display_name]

            for metric in all_metrics:
                abl_metrics = result.ablation_metrics[component]
                val = abl_metrics.get(metric)
                val_str = self._format_value(val)

                # 添加显著性标记
                stats = result.statistical_tests.get(component, {})
                metric_stats = stats.get(metric, {})
                p_value = metric_stats.get("p_value")
                if p_value is not None and p_value < 0.05:
                    val_str += r"$^{*}$"

                row_cells.append(val_str)

            lines.append(" & ".join(row_cells) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}\small",
            r"\item $^{*}$ indicates $p < 0.05$ (Wilcoxon signed-rank test).",
            r"\end{tablenotes}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    @staticmethod
    def _format_metric_name(metric: str) -> str:
        """将指标名格式化为表格表头。"""
        name_map = {
            "accuracy": "Accuracy",
            "kappa": "Kappa",
            "macro_f1": "Macro F1",
            "n1_recall": "N1 Recall",
            "n1_f1": "N1 F1",
        }
        return name_map.get(metric, metric.replace("_", " ").title())

    @staticmethod
    def _format_value(val) -> str:
        """将数值格式化为表格单元格字符串。"""
        if val is None:
            return "--"
        arr = np.atleast_1d(np.asarray(val, dtype=float))
        mean = float(np.mean(arr))
        if len(arr) > 1:
            std = float(np.std(arr, ddof=1))
            return f"{mean:.3f} $\\pm$ {std:.3f}"
        return f"{mean:.3f}"
