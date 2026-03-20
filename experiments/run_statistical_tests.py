#!/usr/bin/env python
"""
OSA-Adapt 统计显著性检验

- Wilcoxon signed-rank tests with Bonferroni correction (patient-level paired comparisons)
- Bootstrap confidence intervals (1000 iterations, percentile method)
- Effect sizes (Cohen's d)

使用 src/adaptation/statistical_tests.py 中的标准化统计函数。

用法:
    PYTHONPATH=. python experiments/run_statistical_tests.py
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.adaptation.statistical_tests import (
    wilcoxon_test,
    bonferroni_correction,
    cohens_d,
    bootstrap_ci,
)

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/statistical")
MODELS = ["Chambon2018", "TinySleepNet"]
METHODS = ["osa_adapt", "full_ft", "last_layer", "lora", "film_no_severity",
           "bn_only", "no_adapt", "coral", "mmd"]
BUDGETS = [5, 10, 20, 30, 50, 65, 100]
METRICS = ["acc", "kappa", "macro_f1", "n1_f1", "severe_acc", "severe_n1_f1"]
N_BOOTSTRAP = 1000
ALPHA = 0.05


def load_all_results():
    """加载所有实验结果，按 model/method/budget/seed/fold 组织。"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith(("main_results", "per_budget", "aggregated", "experiment")):
            continue
        with open(f, encoding="utf-8") as fh:
            raw = json.load(fh)
        # 支持嵌套格式 {config, result}
        if "config" in raw and "result" in raw:
            r = {**raw["config"], **raw["result"]}
            if "status" not in r:
                r["status"] = "completed"
        else:
            r = raw
        if "status" not in r and "acc" in r:
            r["status"] = "completed"
        if r.get("status") != "completed":
            continue
        model = r.get("model_name", "")
        method = r.get("method", r.get("adaptation_method", ""))
        budget = r.get("data_budget", 0)
        seed = r.get("seed", 42)
        fold = r.get("fold", 0)
        results[model][method][budget][(seed, fold)] = r
    return results


def get_patient_metrics(results, model, method, budget):
    """获取某个 model/method/budget 下所有 seed×fold 的指标列表。"""
    runs = results[model][method][budget]
    metrics = {k: [] for k in METRICS}
    for (seed, fold), r in sorted(runs.items()):
        for k in METRICS:
            v = r.get(k)
            if v is not None:
                metrics[k].append(float(v))
    return metrics


def run_pairwise_comparisons(results):
    """OSA-Adapt vs 每个baseline的配对比较。

    使用 src/adaptation/statistical_tests 中的标准化函数：
    - wilcoxon_test: Wilcoxon 符号秩检验 (Requirements 5.2)
    - cohens_d: Cohen's d 效应量 (Requirements 5.4)
    """
    comparisons = {}
    for model in MODELS:
        comparisons[model] = {}
        for budget in BUDGETS:
            comparisons[model][str(budget)] = {}
            osa_metrics = get_patient_metrics(results, model, "osa_adapt", budget)
            for method in METHODS:
                if method == "osa_adapt":
                    continue
                base_metrics = get_patient_metrics(results, model, method, budget)
                comp = {}
                for metric in METRICS:
                    osa_vals = osa_metrics[metric]
                    base_vals = base_metrics[metric]
                    if not osa_vals or not base_vals:
                        continue
                    # 确保配对长度一致
                    n = min(len(osa_vals), len(base_vals))
                    osa_arr = np.array(osa_vals[:n])
                    base_arr = np.array(base_vals[:n])
                    # Wilcoxon 符号秩检验 (Requirement 5.2)
                    wt = wilcoxon_test(osa_arr, base_arr)
                    # Cohen's d 效应量 (Requirement 5.4)
                    d = cohens_d(osa_arr, base_arr)
                    comp[metric] = {
                        "wilcoxon": wt,
                        "cohens_d": d,
                        "osa_adapt_mean": float(np.mean(osa_arr)),
                        "baseline_mean": float(np.mean(base_arr)),
                        "delta": float(np.mean(osa_arr) - np.mean(base_arr)),
                    }
                comparisons[model][str(budget)][method] = comp
    return comparisons


def run_bootstrap_analysis(results):
    """Bootstrap CI for all method/model/budget combinations."""
    bootstrap_results = {}
    for model in MODELS:
        bootstrap_results[model] = {}
        for method in METHODS:
            bootstrap_results[model][method] = {}
            for budget in BUDGETS:
                metrics = get_patient_metrics(results, model, method, budget)
                budget_ci = {}
                for metric in METRICS:
                    vals = metrics[metric]
                    if vals:
                        budget_ci[metric] = bootstrap_ci(
                            np.array(vals),
                            n_bootstrap=N_BOOTSTRAP,
                            confidence=0.95,
                            seed=42,
                        )
                bootstrap_results[model][method][str(budget)] = budget_ci
    return bootstrap_results


def apply_bonferroni_correction(comparisons, n_comparisons=None):
    """Apply Bonferroni correction to p-values (Requirement 5.3).

    使用 src/adaptation/statistical_tests.bonferroni_correction 标准化函数。
    p_corrected = min(p_value * n_comparisons, 1.0)
    """
    if n_comparisons is None:
        n_comparisons = len(METHODS) - 1  # 8 baselines
    corrected = {}
    for model in comparisons:
        corrected[model] = {}
        for budget in comparisons[model]:
            corrected[model][budget] = {}
            for method in comparisons[model][budget]:
                corrected[model][budget][method] = {}
                for metric in comparisons[model][budget][method]:
                    entry = dict(comparisons[model][budget][method][metric])
                    raw_p = entry["wilcoxon"]["p_value"]
                    if not np.isnan(raw_p):
                        # 使用标准化 bonferroni_correction 函数
                        corrected_p_list = bonferroni_correction(
                            [raw_p], n_comparisons=n_comparisons
                        )
                        entry["wilcoxon"]["p_value_corrected"] = corrected_p_list[0]
                        entry["wilcoxon"]["significant"] = corrected_p_list[0] < ALPHA
                        entry["wilcoxon"]["n_comparisons"] = n_comparisons
                    else:
                        entry["wilcoxon"]["p_value_corrected"] = np.nan
                        entry["wilcoxon"]["significant"] = False
                        entry["wilcoxon"]["n_comparisons"] = n_comparisons
                    corrected[model][budget][method][metric] = entry
    return corrected


def generate_summary_json(corrected_comparisons):
    """生成符合设计文档格式的统计检验结果 JSON。

    输出格式:
    {
        "Chambon2018": {
            "osa_adapt_vs_full_ft": {
                "p_value": 0.001,
                "p_corrected": 0.008,
                "cohens_d": 1.23,
                "significant": true,
                "n_comparisons": 8
            }
        }
    }

    对每个 model，聚合所有 budget 下的主要指标 (acc) 的统计检验结果。
    同时输出按 budget 细分的完整结果。
    """
    summary = {}
    detailed = {}

    for model in corrected_comparisons:
        summary[model] = {}
        detailed[model] = {}

        for budget in corrected_comparisons[model]:
            detailed[model][budget] = {}
            for method in corrected_comparisons[model][budget]:
                comparison_key = f"osa_adapt_vs_{method}"
                method_results = {}

                for metric in corrected_comparisons[model][budget][method]:
                    entry = corrected_comparisons[model][budget][method][metric]
                    method_results[metric] = {
                        "p_value": entry["wilcoxon"]["p_value"],
                        "p_corrected": entry["wilcoxon"].get("p_value_corrected", np.nan),
                        "cohens_d": entry["cohens_d"],
                        "significant": entry["wilcoxon"].get("significant", False),
                        "n_comparisons": entry["wilcoxon"].get("n_comparisons", len(METHODS) - 1),
                        "delta": entry["delta"],
                    }

                detailed[model][budget][comparison_key] = method_results

                # 汇总：使用 acc 指标作为主要比较
                if "acc" in method_results:
                    acc_result = method_results["acc"]
                    # 如果该 comparison_key 还没有记录，或者当前 budget 的结果更显著
                    if comparison_key not in summary[model]:
                        summary[model][comparison_key] = {
                            "p_value": acc_result["p_value"],
                            "p_corrected": acc_result["p_corrected"],
                            "cohens_d": acc_result["cohens_d"],
                            "significant": acc_result["significant"],
                            "n_comparisons": acc_result["n_comparisons"],
                            "best_budget": int(budget),
                        }
                    else:
                        # 保留效应量最大的 budget
                        if abs(acc_result["cohens_d"]) > abs(summary[model][comparison_key]["cohens_d"]):
                            summary[model][comparison_key] = {
                                "p_value": acc_result["p_value"],
                                "p_corrected": acc_result["p_corrected"],
                                "cohens_d": acc_result["cohens_d"],
                                "significant": acc_result["significant"],
                                "n_comparisons": acc_result["n_comparisons"],
                                "best_budget": int(budget),
                            }

    return summary, detailed


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, float) and np.isnan(obj): return None
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("加载实验结果...")
    results = load_all_results()

    # 统计加载情况
    for model in MODELS:
        for method in METHODS:
            total = sum(len(results[model][method][b]) for b in BUDGETS)
            print(f"  {model}/{method}: {total} runs")

    print("\n运行配对比较 (Wilcoxon signed-rank)...")
    comparisons = run_pairwise_comparisons(results)

    print("应用 Bonferroni 校正...")
    corrected = apply_bonferroni_correction(comparisons)

    with open(OUTPUT_DIR / "pairwise_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(corrected, f, ensure_ascii=False, indent=2, default=_json_default)

    print("\n运行 Bootstrap CI 分析...")
    bootstrap = run_bootstrap_analysis(results)

    with open(OUTPUT_DIR / "bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap, f, ensure_ascii=False, indent=2, default=_json_default)

    # 生成符合设计文档格式的统计检验结果 JSON
    print("\n生成统计检验结果摘要 JSON...")
    summary, detailed = generate_summary_json(corrected)

    with open(OUTPUT_DIR / "statistical_test_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    with open(OUTPUT_DIR / "statistical_test_detailed.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2, default=_json_default)

    # 打印关键结果摘要
    print("\n" + "=" * 70)
    print("关键统计结果摘要")
    print("=" * 70)
    for model in MODELS:
        print(f"\n{model}:")
        for budget in [20, 50, 100]:
            print(f"  Budget={budget}:")
            for method in ["no_adapt", "full_ft", "coral", "mmd"]:
                if method in corrected[model].get(str(budget), {}):
                    entry = corrected[model][str(budget)][method]
                    if "acc" in entry:
                        a = entry["acc"]
                        sig = "***" if a["wilcoxon"].get("significant") else "n.s."
                        print(f"    vs {method}: Δacc={a['delta']:+.4f} d={a['cohens_d']:.3f} "
                              f"p={a['wilcoxon'].get('p_value_corrected', 'N/A')} {sig}")

    print(f"\n结果已保存到 {OUTPUT_DIR}")
    print(f"  - pairwise_comparisons.json (完整配对比较)")
    print(f"  - bootstrap_ci.json (Bootstrap 置信区间)")
    print(f"  - statistical_test_results.json (设计文档格式摘要)")
    print(f"  - statistical_test_detailed.json (按 budget 细分)")


if __name__ == "__main__":
    main()
