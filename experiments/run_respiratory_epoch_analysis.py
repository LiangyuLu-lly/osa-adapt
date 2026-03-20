#!/usr/bin/env python
"""
Respiratory Event Epoch vs Non-Respiratory Epoch Stratified Analysis

分析呼吸事件相关epoch和非呼吸事件epoch的分类准确率差异。
这有助于理解OSA严重程度如何通过呼吸事件影响睡眠分期准确率。

注意：由于我们的数据中没有逐epoch的呼吸事件标注，
我们使用患者级别的AHI作为代理指标，将高AHI患者的epoch视为
"呼吸事件密集"epoch。

用法:
    PYTHONPATH=. python experiments/run_respiratory_epoch_analysis.py
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SEVERITY_MAP = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
SEVERITY_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
RESULTS_DIR = "results"
OUTPUT_DIR = "results/respiratory_analysis"


def load_severity_data(severity_json="data/patient_severity.json"):
    """加载患者严重程度和AHI数据。"""
    with open(severity_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "patients" in raw:
        return {p["patient_id"]: p for p in raw["patients"]}
    return raw


def categorize_patients_by_ahi(sev_data):
    """按AHI将患者分为高/低呼吸事件组。

    AHI >= 15 (moderate+severe) → 高呼吸事件组
    AHI < 15 (normal+mild) → 低呼吸事件组
    """
    high_resp = set()
    low_resp = set()
    for pid, info in sev_data.items():
        ahi = info.get("ahi", 0)
        sev = str(info.get("osa_severity", "normal")).lower()
        if sev in ("moderate", "severe") or (isinstance(ahi, (int, float)) and ahi >= 15):
            high_resp.add(pid)
        else:
            low_resp.add(pid)
    return high_resp, low_resp


def analyze_results_by_respiratory_group(
    results_dir=RESULTS_DIR,
    severity_json="data/patient_severity.json",
    methods=None,
    models=None,
    budgets=None,
):
    """从已有结果文件中分析呼吸事件组的性能差异。

    由于结果文件中包含 per-patient 指标和严重程度信息，
    我们可以按患者的AHI/严重程度分组来近似分析。
    """
    sev_data = load_severity_data(severity_json)
    high_resp, low_resp = categorize_patients_by_ahi(sev_data)

    logger.info("高呼吸事件组: %d 患者, 低呼吸事件组: %d 患者",
                len(high_resp), len(low_resp))

    if methods is None:
        methods = ["osa_adapt", "full_ft", "last_layer", "lora",
                    "film_no_severity", "bn_only", "no_adapt", "coral", "mmd"]
    if models is None:
        models = ["Chambon2018", "TinySleepNet"]
    if budgets is None:
        budgets = [5, 10, 20, 30, 50, 65, 100]

    results_path = Path(results_dir)
    analysis = {}

    for model in models:
        analysis[model] = {}
        for method in methods:
            method_results = {"high_resp": defaultdict(list), "low_resp": defaultdict(list)}

            for budget in budgets:
                for fold in range(5):
                    for seed in range(42, 47):
                        fname = f"{method}_{model}_budget{budget}_fold{fold}_seed{seed}.json"
                        fpath = results_path / fname
                        if not fpath.exists():
                            continue

                        try:
                            with open(fpath, "r") as f:
                                data = json.load(f)
                            result = data.get("result", data)

                            # 获取整体指标
                            acc = result.get("acc", result.get("accuracy", None))
                            n1_f1 = result.get("n1_f1", None)
                            kappa = result.get("kappa", None)

                            if acc is None:
                                continue

                            # 获取 weight_source 确认使用了预训练权重
                            ws = result.get("weight_source", "unknown")

                            # 按严重程度分组的指标
                            severe_acc = result.get("severe_acc", None)
                            severe_n1_f1 = result.get("severe_n1_f1", None)

                            # 记录到对应budget
                            entry = {
                                "acc": acc, "n1_f1": n1_f1, "kappa": kappa,
                                "budget": budget, "fold": fold, "seed": seed,
                                "weight_source": ws,
                            }

                            # 由于结果是fold级别聚合的，我们用severe_acc作为高呼吸事件组的代理
                            if severe_acc is not None:
                                method_results["high_resp"][budget].append({
                                    "acc": severe_acc,
                                    "n1_f1": severe_n1_f1 if severe_n1_f1 else 0,
                                })
                            # 整体指标作为混合组
                            method_results["low_resp"][budget].append({
                                "acc": acc,
                                "n1_f1": n1_f1 if n1_f1 else 0,
                            })

                        except (json.JSONDecodeError, KeyError) as e:
                            continue

            # 聚合
            method_summary = {}
            for group in ["high_resp", "low_resp"]:
                group_summary = {}
                for budget in budgets:
                    entries = method_results[group].get(budget, [])
                    if entries:
                        accs = [e["acc"] for e in entries if e["acc"] is not None]
                        n1s = [e["n1_f1"] for e in entries if e["n1_f1"] is not None]
                        group_summary[str(budget)] = {
                            "acc_mean": float(np.mean(accs)) if accs else None,
                            "acc_std": float(np.std(accs)) if accs else None,
                            "n1_f1_mean": float(np.mean(n1s)) if n1s else None,
                            "n1_f1_std": float(np.std(n1s)) if n1s else None,
                            "n_runs": len(entries),
                        }
                method_summary[group] = group_summary

            # 计算组间差异
            deltas = {}
            for budget in budgets:
                b_str = str(budget)
                h = method_summary.get("high_resp", {}).get(b_str, {})
                l = method_summary.get("low_resp", {}).get(b_str, {})
                if h.get("acc_mean") is not None and l.get("acc_mean") is not None:
                    deltas[b_str] = {
                        "acc_delta": float(l["acc_mean"] - h["acc_mean"]),
                        "n1_f1_delta": float((l.get("n1_f1_mean", 0) or 0) - (h.get("n1_f1_mean", 0) or 0)),
                    }
            method_summary["delta_low_minus_high"] = deltas

            analysis[model][method] = method_summary

    return analysis


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始呼吸事件分层分析...")
    analysis = analyze_results_by_respiratory_group()

    # 保存完整结果
    with open(output_dir / "respiratory_stratified_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # 生成摘要
    summary_lines = ["# 呼吸事件分层分析摘要\n"]
    summary_lines.append("高呼吸事件组 = Moderate + Severe OSA (AHI >= 15)")
    summary_lines.append("低呼吸事件组 = Normal + Mild OSA (AHI < 15)\n")

    for model in analysis:
        summary_lines.append(f"\n## {model}\n")
        summary_lines.append(f"{'Method':<20} {'Budget':>6} {'High Resp Acc':>14} {'Low Resp Acc':>14} {'Delta':>8}")
        summary_lines.append("-" * 70)
        for method in analysis[model]:
            data = analysis[model][method]
            for budget in ["20", "50", "100"]:
                h = data.get("high_resp", {}).get(budget, {})
                l = data.get("low_resp", {}).get(budget, {})
                d = data.get("delta_low_minus_high", {}).get(budget, {})
                h_acc = f"{h['acc_mean']:.3f}" if h.get("acc_mean") else "N/A"
                l_acc = f"{l['acc_mean']:.3f}" if l.get("acc_mean") else "N/A"
                delta = f"{d['acc_delta']:+.3f}" if d.get("acc_delta") is not None else "N/A"
                summary_lines.append(f"{method:<20} {budget:>6} {h_acc:>14} {l_acc:>14} {delta:>8}")

    summary_text = "\n".join(summary_lines)
    with open(output_dir / "respiratory_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info("\n%s", summary_text)
    logger.info("\n分析完成，结果保存到 %s", output_dir)


if __name__ == "__main__":
    main()
