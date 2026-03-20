#!/usr/bin/env python
"""
Two-Pass AHI Estimator 评估脚本

评估 Ridge Regression AHI 估计器的性能：
- MAE, RMSE, Pearson r
- 4-class severity classification accuracy
- Within-one-class accuracy
- 对比 estimated vs ground-truth AHI conditioning 的适应效果差异

用法:
    PYTHONPATH=. python experiments/run_ahi_estimator_eval.py
"""
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

DEFAULT_PKL_DIR = "data/preprocessed"
DEFAULT_SEVERITY_JSON = "data/patient_severity.json"
DEFAULT_OUTPUT_DIR = "results/ahi_estimator"
MODELS = ["Chambon2018", "TinySleepNet"]
SEVERITY_MAP = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
SEVERITY_THRESHOLDS = [5, 15, 30]  # AHI thresholds for Normal/Mild/Moderate/Severe


def ahi_to_severity(ahi):
    """AHI → severity class (0-3)."""
    if ahi < 5: return 0
    elif ahi < 15: return 1
    elif ahi < 30: return 2
    else: return 3


def extract_sleep_features(predictions, n_classes=5):
    """从睡眠分期预测中提取6个睡眠架构特征。"""
    preds = np.array(predictions)
    total = len(preds)
    if total == 0:
        return np.zeros(6)
    # W=0, N1=1, N2=2, N3=3, REM=4
    n1_ratio = np.mean(preds == 1)
    wake_ratio = np.mean(preds == 0)
    n3_ratio = np.mean(preds == 3)
    rem_ratio = np.mean(preds == 4)
    sleep_mask = preds > 0
    sleep_efficiency = np.mean(sleep_mask)
    transitions = np.sum(preds[1:] != preds[:-1]) / max(total - 1, 1)
    return np.array([n1_ratio, sleep_efficiency, wake_ratio, n3_ratio, rem_ratio, transitions])


def run_ahi_estimation(model_name, train_ds, test_ds, train_ids, test_ids,
                       sev_data, device, seed=42, bs=256, fold=None):
    """对一个fold运行AHI估计评估。"""
    from src.adaptation.model_builder import build_model
    from src.adaptation.weight_loader import WeightLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 构建base model (无FiLM)
    base = build_model(model_name)
    WeightLoader.load_weights(base, model_name, fold=fold)
    base = base.to(device)
    base.eval()

    def predict_patient_base(ds, pid):
        """用base model预测单个患者。"""
        idx = ds.get_patient_epoch_indices(pid)
        if not idx:
            return np.array([])
        preds = []
        with torch.no_grad():
            for s in range(0, len(idx), bs):
                bi = idx[s:s+bs]
                sigs = []
                for i in bi:
                    sig, lab, pf = ds[i]
                    sigs.append(sig)
                x = torch.stack(sigs).to(device)
                out = base(x)
                pred = out.argmax(dim=1).cpu().numpy()
                preds.extend(pred.tolist())
        return np.array(preds)

    # 提取训练集特征和AHI标签
    train_features, train_ahis = [], []
    for pid in train_ids:
        if pid not in sev_data:
            continue
        ahi = sev_data[pid].get("ahi", sev_data[pid].get("AHI", 0))
        if ahi is None:
            continue
        preds = predict_patient_base(train_ds, pid)
        if len(preds) == 0:
            continue
        feats = extract_sleep_features(preds)
        train_features.append(feats)
        train_ahis.append(float(ahi))

    if len(train_features) < 5:
        return None

    X_train = np.array(train_features)
    y_train = np.array(train_ahis)

    # 训练Ridge回归
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # 在测试集上评估
    test_features, test_ahis, test_pids_valid = [], [], []
    for pid in test_ids:
        if pid not in sev_data:
            continue
        ahi = sev_data[pid].get("ahi", sev_data[pid].get("AHI", 0))
        if ahi is None:
            continue
        preds = predict_patient_base(test_ds, pid)
        if len(preds) == 0:
            continue
        feats = extract_sleep_features(preds)
        test_features.append(feats)
        test_ahis.append(float(ahi))
        test_pids_valid.append(pid)

    if len(test_features) < 2:
        return None

    X_test = np.array(test_features)
    y_test = np.array(test_ahis)
    y_pred = ridge.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # AHI不能为负

    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if np.std(y_test) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_test, y_pred)[0, 1])
    else:
        pearson_r = 0.0

    # Severity classification
    true_sev = np.array([ahi_to_severity(a) for a in y_test])
    pred_sev = np.array([ahi_to_severity(a) for a in y_pred])
    exact_acc = float(np.mean(true_sev == pred_sev))
    within_one = float(np.mean(np.abs(true_sev - pred_sev) <= 1))

    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson_r": pearson_r,
        "exact_severity_acc": exact_acc,
        "within_one_class_acc": within_one,
        "n_train": len(train_features),
        "n_test": len(test_features),
        "y_test_mean_ahi": float(np.mean(y_test)),
        "y_pred_mean_ahi": float(np.mean(y_pred)),
    }


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pkl-dir", default=DEFAULT_PKL_DIR)
    p.add_argument("--severity-json", default=DEFAULT_SEVERITY_JSON)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)

    from src.adaptation.cross_validator import CrossValidator
    from src.adaptation.psg_dataset import PSGDataset
    from src.adaptation.demographics_generator import DemographicsGenerator

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 加载severity数据
    with open(args.severity_json, encoding="utf-8") as f:
        raw = json.load(f)
    # 转换为 {patient_id: {...}} 格式
    if "patients" in raw:
        sev_data = {p["patient_id"]: p for p in raw["patients"]}
    else:
        sev_data = raw

    # 获取患者ID
    pkl_dir = Path(args.pkl_dir)
    pids = sorted(set(f"patient_{f.stem.split('_', 1)[0].zfill(3)}" for f in pkl_dir.glob("*.pkl")))
    logger.info("患者数: %d", len(pids))

    sev_labels = []
    for pid in pids:
        if pid in sev_data:
            s = str(sev_data[pid].get("osa_severity", "normal")).lower()
            sev_labels.append(SEVERITY_MAP.get(s, 0))
        else:
            sev_labels.append(hash(pid) % 4)

    cv = CrossValidator(n_folds=args.n_folds, seed=42)
    folds = cv.split(pids, sev_labels)

    all_results = {}
    for model_name in MODELS:
        logger.info("=" * 50)
        logger.info("模型: %s", model_name)
        fold_results = []
        for fi, (tr_ids, te_ids) in enumerate(folds):
            logger.info("  Fold %d: train=%d, test=%d", fi, len(tr_ids), len(te_ids))
            dg = DemographicsGenerator()
            tr_ds = PSGDataset(patient_ids=tr_ids, pkl_dir=args.pkl_dir,
                               severity_data=sev_data, demographics_generator=dg)
            te_ds = PSGDataset(patient_ids=te_ids, pkl_dir=args.pkl_dir,
                               severity_data=sev_data, demographics_generator=dg)

            result = run_ahi_estimation(model_name, tr_ds, te_ds, tr_ids, te_ids,
                                        sev_data, device, seed=args.seed, fold=fi)
            if result:
                result["fold"] = fi
                fold_results.append(result)
                logger.info("    MAE=%.2f RMSE=%.2f r=%.3f exact_acc=%.3f within1=%.3f",
                            result["mae"], result["rmse"], result["pearson_r"],
                            result["exact_severity_acc"], result["within_one_class_acc"])

            del tr_ds, te_ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 聚合
        if fold_results:
            agg = {"n_folds": len(fold_results)}
            for k in ["mae", "rmse", "pearson_r", "exact_severity_acc", "within_one_class_acc"]:
                vals = [r[k] for r in fold_results]
                agg[k] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
            agg["folds"] = fold_results
            all_results[model_name] = agg
            logger.info("  %s 平均: MAE=%.2f±%.2f RMSE=%.2f r=%.3f exact=%.3f within1=%.3f",
                        model_name, agg["mae"], agg["mae_std"], agg["rmse"],
                        agg["pearson_r"], agg["exact_severity_acc"], agg["within_one_class_acc"])

    with open(outdir / "ahi_estimator_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=_json_default)

    logger.info("\n结果已保存到 %s", outdir)


if __name__ == "__main__":
    main()
