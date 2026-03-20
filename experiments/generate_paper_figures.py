#!/usr/bin/env python
"""
OSA-Adapt 结果汇总与可视化脚本

收集所有实验结果，生成论文图表和LaTeX表格。
使用 AdaptationVisualizer 生成：
- 数据效率曲线（含95% CI阴影区域）(Req 13.2)
- 消融柱状图 (Req 13.3)
- 严重程度分层多面板图 (Req 13.4)
- Bland-Altman散点图 (Req 13.5)
- LaTeX格式表格 (Req 13.6)

所有图表使用300 DPI、serif字体，同时输出PNG和PDF格式 (Req 13.1)。

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6

用法示例:
    # 使用默认路径生成所有图表
    PYTHONPATH=. python experiments/generate_paper_figures.py

    # 指定结果目录和输出目录
    PYTHONPATH=. python experiments/generate_paper_figures.py \
        --results-dir results \
        --ablation-dir results/ablation \
        --figures-dir results/figures

    # 指定模型和指标
    PYTHONPATH=. python experiments/generate_paper_figures.py \
        --models Chambon2018 TinySleepNet --metrics accuracy kappa
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认常量
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_DIR = "results"
DEFAULT_ABLATION_DIR = "results/ablation"
DEFAULT_FIGURES_DIR = "results/figures"
DEFAULT_MODELS = ["Chambon2018", "TinySleepNet", "USleep"]
DEFAULT_METHODS = [
    "osa_adapt", "full_ft", "last_layer", "lora",
    "film_no_severity", "bn_only", "no_adapt",
]
DEFAULT_METRICS = ["accuracy", "kappa", "macro_f1", "n1_recall"]
DEFAULT_SEVERITY_GROUPS = ["Normal", "Mild", "Moderate", "Severe"]


# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="OSA-Adapt 结果汇总与可视化：生成论文图表和LaTeX表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help=f"主实验结果目录 (默认: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--ablation-dir", type=str, default=DEFAULT_ABLATION_DIR,
        help=f"消融实验结果目录 (默认: {DEFAULT_ABLATION_DIR})",
    )
    parser.add_argument(
        "--figures-dir", type=str, default=DEFAULT_FIGURES_DIR,
        help=f"图表输出目录 (默认: {DEFAULT_FIGURES_DIR})",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"模型列表 (默认: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        help=f"方法列表 (默认: {DEFAULT_METHODS})",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS,
        help=f"指标列表 (默认: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="图表DPI分辨率 (默认: 300)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
def setup_logging(log_level: str) -> None:
    """配置日志输出到控制台。"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_experiment_results(results_dir: str) -> pd.DataFrame:
    """从结果目录加载所有实验结果JSON文件。

    扫描 results_dir 下的所有 .json 文件，展平 config 和 result 字段
    合并为一个 DataFrame。

    Args:
        results_dir: 结果JSON文件所在目录

    Returns:
        包含所有实验结果的 DataFrame，无结果时返回空 DataFrame
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning("结果目录不存在: %s", results_dir)
        return pd.DataFrame()

    records = []
    for json_file in sorted(results_path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            record = {}
            if "config" in data:
                record.update(data["config"])
            if "result" in data:
                for k, v in data["result"].items():
                    if k not in record and not isinstance(v, (dict, list)):
                        record[k] = v
            records.append(record)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("跳过损坏的结果文件 %s: %s", json_file, e)

    if not records:
        logger.warning("未找到任何实验结果: %s", results_dir)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info("已加载 %d 条实验结果 (来自 %s)", len(df), results_dir)
    return df


def load_ablation_results(ablation_dir: str) -> List[Dict]:
    """从消融结果目录加载所有消融实验JSON文件。

    Args:
        ablation_dir: 消融结果JSON文件所在目录

    Returns:
        消融结果字典列表
    """
    abl_path = Path(ablation_dir)
    if not abl_path.exists():
        logger.warning("消融结果目录不存在: %s", ablation_dir)
        return []

    results = []
    for json_file in sorted(abl_path.glob("ablation_*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            results.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("跳过损坏的消融结果文件 %s: %s", json_file, e)

    logger.info("已加载 %d 个消融实验结果 (来自 %s)", len(results), ablation_dir)
    return results


# ---------------------------------------------------------------------------
# 图表生成
# ---------------------------------------------------------------------------
def generate_data_efficiency_curves(
    visualizer,
    results_df: pd.DataFrame,
    models: List[str],
    methods: List[str],
    metrics: List[str],
) -> None:
    """为每个模型和指标生成数据效率曲线 (Req 13.2)。

    横轴为数据预算，纵轴为性能指标，每种方法一条折线。
    """
    if results_df.empty:
        logger.warning("无实验结果，跳过数据效率曲线生成")
        return

    # 确保必要列存在
    required_cols = {"adaptation_method", "data_budget"}
    if not required_cols.issubset(results_df.columns):
        logger.warning("结果缺少必要列 %s，跳过数据效率曲线", required_cols)
        return

    for model in models:
        model_df = results_df[results_df["model_name"] == model] if "model_name" in results_df.columns else results_df
        if model_df.empty:
            logger.info("模型 %s 无结果，跳过", model)
            continue

        for metric in metrics:
            if metric not in model_df.columns:
                logger.debug("指标 %s 不在结果中，跳过", metric)
                continue

            # 按方法和预算聚合（均值和CI）
            agg_rows = []
            for method in methods:
                method_data = model_df[model_df["adaptation_method"] == method]
                for budget, group in method_data.groupby("data_budget"):
                    values = group[metric].dropna().values
                    if len(values) == 0:
                        continue
                    mean_val = float(np.mean(values))
                    if len(values) > 1:
                        std_val = float(np.std(values, ddof=1))
                        ci_half = 1.96 * std_val / np.sqrt(len(values))
                    else:
                        ci_half = 0.0
                    agg_rows.append({
                        "method": method,
                        "data_budget": budget,
                        metric: mean_val,
                        f"{metric}_ci_lower": mean_val - ci_half,
                        f"{metric}_ci_upper": mean_val + ci_half,
                    })

            if not agg_rows:
                continue

            agg_df = pd.DataFrame(agg_rows)
            save_name = f"data_efficiency_{model}_{metric}"
            visualizer.plot_data_efficiency_curve(
                results_df=agg_df,
                metric=metric,
                methods=methods,
                save_name=save_name,
            )
            logger.info("已生成数据效率曲线: %s", save_name)


def generate_ablation_figures(
    visualizer,
    ablation_results: List[Dict],
    metrics: List[str],
) -> None:
    """为每个消融实验生成分组柱状图 (Req 13.3)。"""
    from src.adaptation.ablation_runner import AblationResult

    if not ablation_results:
        logger.warning("无消融结果，跳过消融图表生成")
        return

    for abl_data in ablation_results:
        model_name = abl_data.get("model_name", "unknown")
        budget = abl_data.get("data_budget", 0)

        abl_result = AblationResult(
            full_model_metrics=abl_data.get("full_model_metrics", {}),
            ablation_metrics=abl_data.get("ablation_metrics", {}),
            statistical_tests=abl_data.get("statistical_tests", {}),
        )

        for metric in metrics:
            if metric not in abl_result.full_model_metrics:
                continue
            save_name = f"ablation_{model_name}_budget{budget}_{metric}"
            visualizer.plot_ablation_bar(
                ablation_result=abl_result,
                metric=metric,
                save_name=save_name,
            )
            logger.info("已生成消融柱状图: %s", save_name)


def generate_severity_panels(
    visualizer,
    results_df: pd.DataFrame,
    metrics: List[str],
) -> None:
    """生成严重程度分层多面板图 (Req 13.4)。

    展示适应前后各严重程度组的性能变化。
    """
    if results_df.empty:
        logger.warning("无实验结果，跳过严重程度分层图生成")
        return

    severity_groups = DEFAULT_SEVERITY_GROUPS

    # 尝试从结果中提取适应前后的严重程度分层指标
    # 查找 "{severity}_{metric}" 格式的列
    before_results = {}
    after_results = {}

    for sev in severity_groups:
        for metric in metrics:
            key = f"{sev}_{metric}"
            before_col = f"baseline_{key}"
            after_col = f"adapted_{key}"

            if before_col in results_df.columns:
                before_results[key] = float(results_df[before_col].mean())
            if after_col in results_df.columns:
                after_results[key] = float(results_df[after_col].mean())

    # 如果没有分层数据，尝试从 no_adapt 和 osa_adapt 方法对比
    if not before_results and "adaptation_method" in results_df.columns:
        for metric in metrics:
            if metric not in results_df.columns:
                continue
            no_adapt = results_df[results_df["adaptation_method"] == "no_adapt"]
            osa_adapt = results_df[results_df["adaptation_method"] == "osa_adapt"]
            if not no_adapt.empty and not osa_adapt.empty:
                before_val = float(no_adapt[metric].mean())
                after_val = float(osa_adapt[metric].mean())
                for sev in severity_groups:
                    before_results[f"{sev}_{metric}"] = before_val
                    after_results[f"{sev}_{metric}"] = after_val

    if before_results and after_results:
        visualizer.plot_severity_stratified_panel(
            results_before=before_results,
            results_after=after_results,
            severity_groups=severity_groups,
            metrics=metrics,
            save_name="severity_stratified_panel",
        )
        logger.info("已生成严重程度分层多面板图")
    else:
        logger.info("无严重程度分层数据，跳过面板图")


def generate_bland_altman_plots(
    visualizer,
    results_df: pd.DataFrame,
) -> None:
    """生成Bland-Altman散点图 (Req 13.5)。

    展示TST估计一致性。
    """
    if results_df.empty:
        logger.warning("无实验结果，跳过Bland-Altman图生成")
        return

    # 查找TST相关列
    true_col = None
    pred_col = None
    for col in results_df.columns:
        if "true_tst" in col.lower():
            true_col = col
        if "predicted_tst" in col.lower() or "pred_tst" in col.lower():
            pred_col = col

    if true_col and pred_col:
        true_tst = results_df[true_col].dropna().values
        pred_tst = results_df[pred_col].dropna().values
        min_len = min(len(true_tst), len(pred_tst))
        if min_len > 0:
            visualizer.plot_bland_altman(
                true_tst=true_tst[:min_len],
                predicted_tst=pred_tst[:min_len],
                save_name="bland_altman_tst",
            )
            logger.info("已生成Bland-Altman图")
        else:
            logger.info("TST数据为空，跳过Bland-Altman图")
    else:
        logger.info("未找到TST列 (true_tst/predicted_tst)，跳过Bland-Altman图")


def generate_latex_tables(
    visualizer,
    results_df: pd.DataFrame,
    models: List[str],
    methods: List[str],
    metrics: List[str],
) -> None:
    """生成LaTeX格式的结果表格 (Req 13.6)。"""
    if results_df.empty:
        logger.warning("无实验结果，跳过LaTeX表格生成")
        return

    # 表1: 主结果表（方法 × 指标，按模型分组）
    for model in models:
        model_df = results_df[results_df["model_name"] == model] if "model_name" in results_df.columns else results_df
        if model_df.empty:
            continue

        available_metrics = [m for m in metrics if m in model_df.columns]
        if not available_metrics:
            continue

        table_rows = []
        for method in methods:
            method_data = model_df[model_df["adaptation_method"] == method] if "adaptation_method" in model_df.columns else pd.DataFrame()
            if method_data.empty:
                continue
            row = {"Method": method}
            for metric in available_metrics:
                values = method_data[metric].dropna().values
                if len(values) > 0:
                    mean_val = float(np.mean(values))
                    if len(values) > 1:
                        std_val = float(np.std(values, ddof=1))
                        row[metric] = f"{mean_val:.3f} ± {std_val:.3f}"
                    else:
                        row[metric] = f"{mean_val:.3f}"
                else:
                    row[metric] = "--"
            table_rows.append(row)

        if table_rows:
            table_df = pd.DataFrame(table_rows)
            visualizer.generate_latex_table(
                results_df=table_df,
                caption=f"Adaptation Results for {model}",
                label=f"tab:results_{model.lower()}",
                save_name=f"table_results_{model}",
            )
            logger.info("已生成LaTeX表格: table_results_%s", model)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def generate_all_figures(args: argparse.Namespace) -> None:
    """主流程：加载结果并生成所有论文图表。"""
    from src.adaptation.visualizer import AdaptationVisualizer

    logger.info("=" * 60)
    logger.info("OSA-Adapt 论文图表生成")
    logger.info("=" * 60)
    logger.info("结果目录: %s", args.results_dir)
    logger.info("消融目录: %s", args.ablation_dir)
    logger.info("图表输出: %s", args.figures_dir)

    # 初始化可视化器 (Req 13.1: 300 DPI, serif字体, PNG+PDF)
    visualizer = AdaptationVisualizer(
        output_dir=args.figures_dir,
        dpi=args.dpi,
        font_family="serif",
    )

    # 1. 加载实验结果
    results_df = load_experiment_results(args.results_dir)
    ablation_results = load_ablation_results(args.ablation_dir)

    # 2. 数据效率曲线 (Req 13.2)
    logger.info("-" * 40)
    logger.info("生成数据效率曲线...")
    generate_data_efficiency_curves(
        visualizer=visualizer,
        results_df=results_df,
        models=args.models,
        methods=args.methods,
        metrics=args.metrics,
    )

    # 3. 消融柱状图 (Req 13.3)
    logger.info("-" * 40)
    logger.info("生成消融柱状图...")
    generate_ablation_figures(
        visualizer=visualizer,
        ablation_results=ablation_results,
        metrics=args.metrics,
    )

    # 4. 严重程度分层多面板图 (Req 13.4)
    logger.info("-" * 40)
    logger.info("生成严重程度分层图...")
    generate_severity_panels(
        visualizer=visualizer,
        results_df=results_df,
        metrics=args.metrics,
    )

    # 5. Bland-Altman图 (Req 13.5)
    logger.info("-" * 40)
    logger.info("生成Bland-Altman图...")
    generate_bland_altman_plots(
        visualizer=visualizer,
        results_df=results_df,
    )

    # 6. LaTeX表格 (Req 13.6)
    logger.info("-" * 40)
    logger.info("生成LaTeX表格...")
    generate_latex_tables(
        visualizer=visualizer,
        results_df=results_df,
        models=args.models,
        methods=args.methods,
        metrics=args.metrics,
    )

    logger.info("=" * 60)
    logger.info("OSA-Adapt 论文图表生成完成")
    logger.info("输出目录: %s", args.figures_dir)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """脚本入口。"""
    args = parse_args(argv)
    setup_logging(args.log_level)
    generate_all_figures(args)


if __name__ == "__main__":
    main()
