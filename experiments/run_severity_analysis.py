#!/usr/bin/env python
"""
Per-Severity-Group Adaptation Results Analysis

从已有的实验结果JSON文件中，按OSA严重程度分组统计适应效果。
生成每个严重程度组（Normal/Mild/Moderate/Severe）的详细指标。

用法:
    PYTHONPATH=. python experiments/run_severity_stratified_analysis.py
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
SEVERITY_JSON = "data/patient_severity.json"
OUTPUT_DIR = "results/severity_stratified"

SEVERITY_MAP = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
SEVERITY_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
MODELS = ["Chambon2018", "TinySleepNet"]
METHODS = ["osa_adapt", "full_ft", "last_layer", "lora", "film_no_severity",
           "bn_only", "no_adapt", "coral", "mmd"]
BUDGETS = [5, 10, 20, 30, 50, 65, 100]


def load_severity_data():
    with open(SEVERITY_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "patients" in raw:
        return {p["patient_id"]: p for p in raw["patients"]}
    return raw


def get_patient_severity(pid, sev_data):
    if pid in sev_data:
        s = str(sev_data[pid].get("osa_severity", "normal")).lower()
        return SEVERITY_MAP.get(s, 0)
    return hash(pid) % 4


def analyze_results():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)

    sev_data = load_severity_data()
    results_path = Path(RESULTS_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        logger.error("结果目录不存在: %s", RESULTS_DIR)
        return

    # 收集所有结果
    all_results = {}
    for f in sorted(results_path.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            # 提取config和result
            if "config" in data and "result" in data:
                config = data["config"]
                result = data["result"]
            else:
                config = data
                result = data

            key = f.stem
            all_results[key] = {"config": config, "result": result, "file": str(f)}
        except Exception as e:
            logger.warning("读取失败 %s: %s", f, e)

    logger.info("加载了 %d 个结果文件", len(all_results))

    # 按 model × method × budget 聚合，提取 per-severity 指标
    # 结果文件中的 patient_results 包含每个患者的指标
    severity_analysis = {}

    for model in MODELS:
        severity_analysis[model] = {}
        for method in METHODS:
            severity_analysis[model][method] = {}
            for budget in BUDGETS:
                # 收集所有 fold × seed 的结果
                sev_metrics = defaultdict(lambda: defaultdict(list))

                pattern = f"{method}_{model}_budget{budget}_fold*_seed*.json"
                matching = list(results_path.glob(pattern))

                for rf in matching:
                    try:
                        with open(rf, "r", encoding="utf-8") as fp:
                            data = json.load(fp)
                        result = data.get("result", data)

                        # 检查是否有 patient_results
                        patient_results = result.get("patient_results", [])
                        if not patient_results:
                            # 使用聚合指标
                            for sev_name in ["severe", "moderate", "mild", "normal"]:
                                acc_key = f"{sev_name}_acc"
                                n1_key = f"{sev_name}_n1_f1"
                                if acc_key in result:
                                    sev_idx = SEVERITY_MAP.get(sev_name, 0)
                                    sev_metrics[sev_idx]["acc"].append(result[acc_key])
                                if n1_key in result:
                                    sev_idx = SEVERITY_MAP.get(sev_name, 0)
                                    sev_metrics[sev_idx]["n1_f1"].append(result[n1_key])
                            # 也记录总体指标
                            if "acc" in result:
                                sev_metrics["overall"]["acc"].append(result["acc"])
                            if "n1_f1" in result:
                                sev_metrics["overall"]["n1_f1"].append(result.get("n1_f1", 0))
                            if "kappa" in result:
                                sev_metrics["overall"]["kappa"].append(result["kappa"])
                            if "macro_f1" in result:
                                sev_metrics["overall"]["macro_f1"].append(result["macro_f1"])
                        else:
                            # 有 per-patient 结果
                            for pr in patient_results:
                                pid = pr.get("patient_id", "")
                                sev = pr.get("severity", get_patient_severity(pid, sev_data))
                                for metric in ["accuracy", "n1_f1", "kappa", "macro_f1"]:
                                    if metric in pr:
                                        sev_metrics[sev][metric].append(pr[metric])
                    except Exception as e:
                        logger.warning("解析失败 %s: %s", rf, e)

                # 计算统计量
                budget_stats = {}
                for sev_key, metrics in sev_metrics.items():
                    sev_label = SEVERITY_NAMES.get(sev_key, str(sev_key))
                    budget_stats[sev_label] = {}
                    for metric_name, values in metrics.items():
                        if values:
                            budget_stats[sev_label][metric_name] = {
                                "mean": float(np.mean(values)),
                                "std": float(np.std(values)),
                                "n": len(values),
                            }

                severity_analysis[model][method][f"budget_{budget}"] = budget_stats

    # 保存完整分析
    with open(output_path / "severity_stratified_full.json", "w", encoding="utf-8") as f:
        json.dump(severity_analysis, f, indent=2, ensure_ascii=False)

    # 生成摘要表格
    summary = generate_summary_table(severity_analysis)
    with open(output_path / "severity_stratified_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("分析完成，结果保存到 %s", output_path)
    return severity_analysis


def generate_summary_table(analysis):
    """生成跨budget平均的severity分层摘要。"""
    summary = {}
    for model in MODELS:
        summary[model] = {}
        for method in METHODS:
            method_data = analysis.get(model, {}).get(method, {})
            # 跨所有budget聚合
            sev_agg = defaultdict(lambda: defaultdict(list))
            for budget_key, budget_stats in method_data.items():
                for sev_label, metrics in budget_stats.items():
                    for metric_name, stats in metrics.items():
                        if "mean" in stats:
                            sev_agg[sev_label][metric_name].append(stats["mean"])

            summary[model][method] = {}
            for sev_label, metrics in sev_agg.items():
                summary[model][method][sev_label] = {}
                for metric_name, values in metrics.items():
                    summary[model][method][sev_label][metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }

    return summary


if __name__ == "__main__":
    analyze_results()
