#!/usr/bin/env python
"""
Confusion Matrix Analysis for OSA-Adapt

对每个模型×方法×严重程度组生成混淆矩阵，分析误分类模式。
重点关注N1的误分类方向（N1→W vs N1→N2）在不同严重程度下的变化。

用法:
    PYTHONPATH=. python experiments/run_confusion_matrix_analysis.py
"""

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
SEVERITY_MAP = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
SEVERITY_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
OUTPUT_DIR = "results/confusion_matrix"


def run_confusion_analysis(
    models=("Chambon2018", "TinySleepNet"),
    methods=("osa_adapt", "no_adapt", "full_ft"),
    budget=20,
    n_folds=5,
    seed=42,
    pkl_dir="data/preprocessed",
    severity_json="data/patient_severity.json",
):
    """运行混淆矩阵分析。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)

    from src.adaptation.psg_dataset import PSGDataset
    from src.adaptation.demographics_generator import DemographicsGenerator
    from src.adaptation.model_builder import build_model
    from src.adaptation.weight_loader import WeightLoader
    from src.adaptation.severity_conditioner import SeverityConditioner
    from src.adaptation.wrapped_models import FiLMWrappedChambon, FiLMWrappedTinySleepNet
    from src.adaptation.progressive_adapter import ProgressiveAdapter
    from src.adaptation.severity_aware_loss import SeverityAwareN1Loss
    from src.adaptation.baselines import create_baseline
    from src.adaptation.stratified_sampler import SeverityStratifiedFewShotSampler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载数据
    with open(severity_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sev_data = {p["patient_id"]: p for p in raw["patients"]} if "patients" in raw else raw

    pkl_path = Path(pkl_dir)
    pids = sorted({f"patient_{f.stem.split('_', 1)[0].zfill(3)}" for f in pkl_path.glob("*.pkl")})
    sev_labels = []
    for pid in pids:
        if pid in sev_data:
            s = str(sev_data[pid].get("osa_severity", "normal")).lower()
            sev_labels.append(SEVERITY_MAP.get(s, 0))
        else:
            sev_labels.append(hash(pid) % 4)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(skf.split(pids, sev_labels))

    METHOD_MAP = {
        "full_ft": "full_finetune", "last_layer": "last_layer_finetune",
        "lora": "lora_adaptation", "film_no_severity": "standard_film",
        "bn_only": "bn_only_adaptation", "no_adapt": "no_adaptation",
        "coral": "coral_adaptation", "mmd": "mmd_adaptation",
    }

    all_confusion = {}
    dg = DemographicsGenerator()

    for model_name in models:
        all_confusion[model_name] = {}
        for method in methods:
            logger.info("=== %s × %s ===", model_name, method)

            # 收集所有fold的预测
            all_y_true = []
            all_y_pred = []
            all_severities = []

            for fi, (train_idx, test_idx) in enumerate(folds):
                train_ids = [pids[i] for i in train_idx]
                test_ids = [pids[i] for i in test_idx]

                torch.manual_seed(seed)
                np.random.seed(seed)

                # 创建数据集
                test_ds = PSGDataset(patient_ids=test_ids, pkl_dir=pkl_dir,
                                     severity_data=sev_data, demographics_generator=dg)
                train_ds = PSGDataset(patient_ids=train_ids, pkl_dir=pkl_dir,
                                      severity_data=sev_data, demographics_generator=dg)

                # 构建模型
                base = build_model(model_name)
                WeightLoader.load_weights(base, model_name, fold=fi)

                if method == "osa_adapt":
                    cond = SeverityConditioner(condition_dim=64)
                    if model_name == "Chambon2018":
                        model = FiLMWrappedChambon(base, cond)
                    else:
                        model = FiLMWrappedTinySleepNet(base, cond)
                    model = model.to(device)

                    # 采样适应集
                    train_sevs = [sev_labels[i] for i in train_idx]
                    sampler = SeverityStratifiedFewShotSampler(seed=seed)
                    adapt_ids = sampler.sample(train_ids, train_sevs, budget)
                    n_val = max(1, len(adapt_ids) // 5)
                    rng = np.random.RandomState(seed)
                    sh = list(adapt_ids); rng.shuffle(sh)
                    val_ids = sh[:n_val]; at_ids = sh[n_val:] or list(adapt_ids)

                    from experiments.run_main_experiment import (
                        create_dataloader, create_subset_dataloader, TypeFixingDataLoader
                    )
                    ul = TypeFixingDataLoader(create_dataloader(train_ds, 128, shuffle=False))
                    tl = TypeFixingDataLoader(create_subset_dataloader(train_ds, at_ids, 128))
                    vl = TypeFixingDataLoader(create_subset_dataloader(train_ds, val_ids, 128, shuffle=False))

                    loss_fn = SeverityAwareN1Loss()
                    adapter = ProgressiveAdapter(model=model, conditioner=cond, loss_fn=loss_fn)
                    adapter.phase1_bn_adapt(ul)
                    adapter.phase2_film_finetune(tl, vl)
                    eval_model = model

                elif method == "no_adapt":
                    cond = SeverityConditioner(condition_dim=64)
                    if model_name == "Chambon2018":
                        model = FiLMWrappedChambon(base, cond)
                    else:
                        model = FiLMWrappedTinySleepNet(base, cond)
                    model = model.to(device)
                    eval_model = model

                else:
                    base = base.to(device)
                    baseline_name = METHOD_MAP.get(method)
                    if baseline_name is None:
                        continue

                    train_sevs = [sev_labels[i] for i in train_idx]
                    sampler = SeverityStratifiedFewShotSampler(seed=seed)
                    adapt_ids = sampler.sample(train_ids, train_sevs, budget)
                    n_val = max(1, len(adapt_ids) // 5)
                    rng = np.random.RandomState(seed)
                    sh = list(adapt_ids); rng.shuffle(sh)
                    val_ids = sh[:n_val]; at_ids = sh[n_val:] or list(adapt_ids)

                    from experiments.run_main_experiment import (
                        create_subset_dataloader, TypeFixingDataLoader
                    )
                    tl = TypeFixingDataLoader(create_subset_dataloader(train_ds, at_ids, 128))
                    vl = TypeFixingDataLoader(create_subset_dataloader(train_ds, val_ids, 128, shuffle=False))

                    bl = create_baseline(baseline_name)
                    result = bl.adapt(base, tl, vl)
                    eval_model = result.pop("adapted_model", base)

                # 预测测试集
                eval_model.eval()
                for pid in test_ids:
                    indices = test_ds.get_patient_epoch_indices(pid)
                    if not indices:
                        continue
                    sev = SEVERITY_MAP.get(
                        str(sev_data.get(pid, {}).get("osa_severity", "normal")).lower(), 0
                    ) if pid in sev_data else hash(pid) % 4

                    with torch.no_grad():
                        for s in range(0, len(indices), 256):
                            bi = indices[s:s+256]
                            sigs, labs = [], []
                            fl = {k: [] for k in ["ahi", "severity", "age", "sex", "bmi"]}
                            for idx in bi:
                                sig, lab, pf = test_ds[idx]
                                sigs.append(sig); labs.append(lab)
                                for k in fl: fl[k].append(pf[k])
                            x = torch.stack(sigs).to(device)
                            pf = {k: torch.stack(v).to(device) for k, v in fl.items()}
                            pf["severity"] = pf["severity"].long()
                            pf["sex"] = pf["sex"].long()
                            out = eval_model(x, pf)
                            pred = out.argmax(dim=1).cpu().numpy()
                            for l, p in zip(labs, pred):
                                if l >= 0:
                                    all_y_true.append(l)
                                    all_y_pred.append(p)
                                    all_severities.append(sev)

                del eval_model, base, train_ds, test_ds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("  Fold %d 完成", fi)

            # 计算混淆矩阵
            y_true = np.array(all_y_true)
            y_pred = np.array(all_y_pred)
            severities = np.array(all_severities)

            method_results = {
                "overall": {
                    "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(5)).tolist(),
                    "n_samples": len(y_true),
                    "accuracy": float((y_true == y_pred).mean()),
                },
            }

            # Per-severity confusion matrices
            for sev_idx, sev_name in SEVERITY_NAMES.items():
                mask = severities == sev_idx
                if mask.sum() > 0:
                    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=range(5))
                    # N1 误分类分析
                    n1_true = y_true[mask] == 1
                    n1_total = n1_true.sum()
                    if n1_total > 0:
                        n1_preds = y_pred[mask][n1_true]
                        n1_to_w = (n1_preds == 0).sum()
                        n1_to_n2 = (n1_preds == 2).sum()
                        n1_to_n3 = (n1_preds == 3).sum()
                        n1_to_rem = (n1_preds == 4).sum()
                        n1_correct = (n1_preds == 1).sum()
                    else:
                        n1_to_w = n1_to_n2 = n1_to_n3 = n1_to_rem = n1_correct = 0

                    method_results[sev_name] = {
                        "confusion_matrix": cm.tolist(),
                        "n_samples": int(mask.sum()),
                        "accuracy": float((y_true[mask] == y_pred[mask]).mean()),
                        "n1_analysis": {
                            "total_n1_epochs": int(n1_total),
                            "n1_correct": int(n1_correct),
                            "n1_to_W": int(n1_to_w),
                            "n1_to_N2": int(n1_to_n2),
                            "n1_to_N3": int(n1_to_n3),
                            "n1_to_REM": int(n1_to_rem),
                            "n1_recall": float(n1_correct / max(n1_total, 1)),
                        },
                    }

            all_confusion[model_name][method] = method_results
            logger.info("  Overall acc=%.4f, n=%d", method_results["overall"]["accuracy"], len(y_true))

    # 保存
    with open(output_path / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(all_confusion, f, indent=2, ensure_ascii=False)

    logger.info("混淆矩阵分析完成，保存到 %s", output_path)
    return all_confusion


if __name__ == "__main__":
    run_confusion_analysis()
