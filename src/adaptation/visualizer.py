"""
论文级可视化 — AdaptationVisualizer

生成符合Q2+期刊投稿标准的图表和LaTeX表格：
- 数据效率曲线（含95% CI阴影区域）
- 消融柱状图（分组柱状图）
- 严重程度分层多面板图
- Bland-Altman散点图
- LaTeX格式表格

所有图表使用300 DPI、serif字体，同时输出PNG和PDF格式。

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptationVisualizer:
    """论文级可视化生成器。

    配置matplotlib全局默认值（DPI、字体），提供多种图表生成方法，
    并统一保存为PNG+PDF双格式。
    """

    def __init__(
        self,
        output_dir: str,
        dpi: int = 300,
        font_family: str = "serif",
    ):
        """初始化可视化器，配置matplotlib默认参数。

        Args:
            output_dir: 图表输出目录
            dpi: 图表分辨率，默认300
            font_family: 字体族，默认serif
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.font_family = font_family

        # 配置matplotlib全局默认值
        plt.rcParams.update({
            "font.family": font_family,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.grid": True,
            "grid.alpha": 0.3,
        })

    def plot_data_efficiency_curve(
        self,
        results_df: pd.DataFrame,
        metric: str,
        methods: List[str],
        save_name: str,
    ) -> plt.Figure:
        """绘制数据效率曲线（含95% CI阴影区域）。

        横轴为数据预算（适应集患者数），纵轴为性能指标。
        每种方法一条折线，阴影区域表示95%置信区间。

        Args:
            results_df: 结果DataFrame，需包含列:
                - 'data_budget': 数据预算
                - 'method': 方法名
                - '{metric}': 指标值
                - '{metric}_ci_lower' (可选): CI下界
                - '{metric}_ci_upper' (可选): CI上界
            metric: 指标名（如 'accuracy', 'kappa'）
            methods: 要绘制的方法列表
            save_name: 保存文件名（不含扩展名）

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        ci_lower_col = f"{metric}_ci_lower"
        ci_upper_col = f"{metric}_ci_upper"
        has_ci = ci_lower_col in results_df.columns and ci_upper_col in results_df.columns

        for method in methods:
            method_data = results_df[results_df["method"] == method].sort_values("data_budget")
            if method_data.empty:
                continue

            budgets = method_data["data_budget"].values
            values = method_data[metric].values

            ax.plot(budgets, values, marker="o", label=method, linewidth=1.5)

            if has_ci:
                ci_low = method_data[ci_lower_col].values
                ci_high = method_data[ci_upper_col].values
                ax.fill_between(budgets, ci_low, ci_high, alpha=0.2)

        ax.set_xlabel("Number of Adaptation Patients")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Data Efficiency: {metric.replace('_', ' ').title()}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        self._save_figure(fig, save_name)
        return fig

    def plot_ablation_bar(
        self,
        ablation_result,
        metric: str,
        save_name: str,
    ) -> plt.Figure:
        """绘制消融实验分组柱状图。

        展示完整模型和各消融配置在指定指标上的对比。

        Args:
            ablation_result: AblationResult对象，包含:
                - full_model_metrics: 完整模型指标
                - ablation_metrics: 各消融配置指标
            metric: 要绘制的指标名
            save_name: 保存文件名（不含扩展名）

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = ["Full Model"]
        values = []
        full_val = ablation_result.full_model_metrics.get(metric, 0.0)
        full_val = float(np.mean(np.atleast_1d(full_val)))
        values.append(full_val)

        for component, metrics in ablation_result.ablation_metrics.items():
            display_name = component.replace("no_", "w/o ").replace("_", " ").title()
            labels.append(display_name)
            val = metrics.get(metric, 0.0)
            values.append(float(np.mean(np.atleast_1d(val))))

        x = np.arange(len(labels))
        colors = ["#2196F3"] + ["#FF9800"] * (len(labels) - 1)
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Ablation Study: {metric.replace('_', ' ').title()}")

        # 在柱子上方标注数值
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        fig.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_severity_stratified_panel(
        self,
        results_before: Dict[str, float],
        results_after: Dict[str, float],
        severity_groups: List[str],
        metrics: List[str],
        save_name: str,
    ) -> plt.Figure:
        """绘制严重程度分层多面板图。

        每个指标一个子面板，展示适应前后各严重程度组的性能变化。

        Args:
            results_before: 适应前结果，键为 "{severity}_{metric}" 格式
            results_after: 适应后结果，键为 "{severity}_{metric}" 格式
            severity_groups: 严重程度组名列表，如 ["Normal", "Mild", "Moderate", "Severe"]
            metrics: 指标列表，如 ["accuracy", "kappa", "n1_recall"]
            save_name: 保存文件名（不含扩展名）

        Returns:
            matplotlib Figure对象
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        x = np.arange(len(severity_groups))
        width = 0.35

        for idx, (metric, ax) in enumerate(zip(metrics, axes)):
            before_vals = []
            after_vals = []
            for sev in severity_groups:
                key = f"{sev}_{metric}"
                before_vals.append(results_before.get(key, 0.0))
                after_vals.append(results_after.get(key, 0.0))

            ax.bar(x - width / 2, before_vals, width, label="Before", color="#EF5350", alpha=0.8)
            ax.bar(x + width / 2, after_vals, width, label="After", color="#66BB6A", alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(severity_groups, fontsize=8)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(metric.replace("_", " ").title())
            if idx == 0:
                ax.legend(fontsize=8)

        fig.suptitle("Performance by OSA Severity: Before vs After Adaptation", fontsize=11)
        fig.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_bland_altman(
        self,
        true_tst: np.ndarray,
        predicted_tst: np.ndarray,
        save_name: str,
    ) -> plt.Figure:
        """绘制Bland-Altman散点图。

        横轴为两次测量的均值，纵轴为差值（predicted - true）。
        包含均值偏差线和95%一致性界限（LoA）线。

        Args:
            true_tst: 真实TST值数组
            predicted_tst: 预测TST值数组
            save_name: 保存文件名（不含扩展名）

        Returns:
            matplotlib Figure对象
        """
        true_tst = np.asarray(true_tst, dtype=np.float64)
        predicted_tst = np.asarray(predicted_tst, dtype=np.float64)

        mean_vals = (true_tst + predicted_tst) / 2.0
        diff_vals = predicted_tst - true_tst
        mean_diff = float(np.mean(diff_vals))
        std_diff = float(np.std(diff_vals, ddof=1))
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(mean_vals, diff_vals, alpha=0.5, s=20, color="#1976D2", edgecolors="none")

        # 均值偏差线
        ax.axhline(y=mean_diff, color="black", linestyle="-", linewidth=1,
                    label=f"Mean diff = {mean_diff:.2f}")
        # 95% LoA
        ax.axhline(y=loa_upper, color="red", linestyle="--", linewidth=1,
                    label=f"+1.96 SD = {loa_upper:.2f}")
        ax.axhline(y=loa_lower, color="red", linestyle="--", linewidth=1,
                    label=f"-1.96 SD = {loa_lower:.2f}")

        ax.set_xlabel("Mean of True and Predicted TST (min)")
        ax.set_ylabel("Difference (Predicted - True) (min)")
        ax.set_title("Bland-Altman Plot: TST Estimation Agreement")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        self._save_figure(fig, save_name)
        return fig

    def generate_latex_table(
        self,
        results_df: pd.DataFrame,
        caption: str,
        label: str,
        save_name: str,
    ) -> str:
        """生成LaTeX格式表格文件。

        将DataFrame转换为LaTeX表格并保存到文件。

        Args:
            results_df: 结果DataFrame
            caption: 表格标题
            label: LaTeX标签（用于引用）
            save_name: 保存文件名（不含扩展名，自动添加.tex）

        Returns:
            LaTeX表格字符串
        """
        n_cols = len(results_df.columns)
        col_spec = "l" + "c" * (n_cols - 1)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]

        # 表头
        header = " & ".join(str(c) for c in results_df.columns) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        # 数据行
        for _, row in results_df.iterrows():
            cells = []
            for val in row:
                if isinstance(val, float):
                    cells.append(f"{val:.3f}")
                else:
                    cells.append(str(val))
            lines.append(" & ".join(cells) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        latex_str = "\n".join(lines)

        # 保存到文件
        save_path = self.output_dir / f"{save_name}.tex"
        save_path.write_text(latex_str, encoding="utf-8")
        logger.info(f"LaTeX表格已保存: {save_path}")

        return latex_str

    def _save_figure(self, fig: plt.Figure, save_name: str) -> None:
        """保存图表为PNG和PDF双格式。

        Args:
            fig: matplotlib Figure对象
            save_name: 文件名（不含扩展名）
        """
        for ext in ("png", "pdf"):
            path = self.output_dir / f"{save_name}.{ext}"
            fig.savefig(str(path), dpi=self.dpi, bbox_inches="tight", format=ext)
            logger.info(f"图表已保存: {path}")

        plt.close(fig)
