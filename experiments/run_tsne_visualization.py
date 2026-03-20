#!/usr/bin/env python
"""
Conditioning Vector Visualization (t-SNE / UMAP)

可视化 SeverityConditioner 生成的 conditioning vector 在不同 OSA 严重程度下的分布。
生成 t-SNE 和 UMAP 降维图，展示 conditioning vector 是否能有效区分不同严重程度。

用法:
    PYTHONPATH=. python experiments/run_tsne_visualization.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

SEVERITY_MAP = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
SEVERITY_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
SEVERITY_COLORS = {0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}
OUTPUT_DIR = "results/tsne"


def extract_conditioning_vectors(
    model_name: str = "Chambon2018",
    fold: int = 0,
    budget: int = 20,
    seed: int = 42,
    pkl_dir: str = "data/preprocessed",
    severity_json: str = "data/patient_severity.json",
):
    """提取 conditioning vectors 并进行可视化。"""
    from src.adaptation.psg_dataset import PSGDataset
    from src.adaptation.demographics_generator import DemographicsGenerator
    from src.adaptation.model_builder import build_model
    from src.adaptation.weight_loader import WeightLoader
    from src.adaptation.severity_conditioner import SeverityConditioner
    from src.adaptation.wrapped_models import FiLMWrappedChambon, FiLMWrappedTinySleepNet
    from src.adaptation.progressive_adapter import ProgressiveAdapter
    from src.adaptation.severity_aware_loss import SeverityAwareN1Loss
    from src.adaptation.stratified_sampler import SeverityStratifiedFewShotSampler
    from sklearn.model_selection import StratifiedKFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds_list = list(skf.split(pids, sev_labels))
    train_idx, test_idx = folds_list[fold]
    train_ids = [pids[i] for i in train_idx]
    test_ids = [pids[i] for i in test_idx]

    torch.manual_seed(seed)
    np.random.seed(seed)

    dg = DemographicsGenerator()

    # 构建并适应模型
    base = build_model(model_name)
    WeightLoader.load_weights(base, model_name, fold=fold)
    cond = SeverityConditioner(condition_dim=64)
    if model_name == "Chambon2018":
        model = FiLMWrappedChambon(base, cond)
    else:
        model = FiLMWrappedTinySleepNet(base, cond)
    model = model.to(device)

    # 适应
    train_ds = PSGDataset(patient_ids=train_ids, pkl_dir=pkl_dir,
                          severity_data=sev_data, demographics_generator=dg)
    test_ds = PSGDataset(patient_ids=test_ids, pkl_dir=pkl_dir,
                         severity_data=sev_data, demographics_generator=dg)

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

    # 提取 conditioning vectors
    model.eval()
    all_vectors = []
    all_severities = []
    all_pids = []

    for pid in test_ids:
        indices = test_ds.get_patient_epoch_indices(pid)
        if not indices:
            continue
        sev = SEVERITY_MAP.get(
            str(sev_data.get(pid, {}).get("osa_severity", "normal")).lower(), 0
        ) if pid in sev_data else hash(pid) % 4

        # 取该患者的前10个epoch的conditioning vector
        sample_indices = indices[:min(10, len(indices))]
        with torch.no_grad():
            for idx in sample_indices:
                sig, lab, pf = test_ds[idx]
                pf_batch = {k: v.unsqueeze(0).to(device) for k, v in pf.items()}
                pf_batch["severity"] = pf_batch["severity"].long()
                pf_batch["sex"] = pf_batch["sex"].long()
                cv = cond(pf_batch)  # conditioning vector
                all_vectors.append(cv.cpu().numpy().flatten())
                all_severities.append(sev)
                all_pids.append(pid)

    vectors = np.array(all_vectors)
    severities = np.array(all_severities)

    return vectors, severities, all_pids


def plot_tsne(vectors, severities, model_name, output_dir):
    """生成 t-SNE 可视化图。"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors) - 1))
    embedded = tsne.fit_transform(vectors)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for sev_idx in sorted(SEVERITY_NAMES.keys()):
        mask = severities == sev_idx
        if mask.sum() > 0:
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=SEVERITY_COLORS[sev_idx], label=SEVERITY_NAMES[sev_idx],
                alpha=0.6, s=20, edgecolors="none",
            )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Conditioning Vector t-SNE ({model_name})")
    ax.legend(title="OSA Severity")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tsne_{model_name}.png", dpi=150)
    plt.savefig(f"{output_dir}/tsne_{model_name}.pdf")
    plt.close()
    logger.info("t-SNE 图已保存: %s/tsne_%s.png", output_dir, model_name)


def plot_umap(vectors, severities, model_name, output_dir):
    """生成 UMAP 可视化图（如果 umap-learn 可用）。"""
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn 未安装，跳过 UMAP 可视化")
        return

    reducer = UMAP(n_components=2, random_state=42)
    embedded = reducer.fit_transform(vectors)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for sev_idx in sorted(SEVERITY_NAMES.keys()):
        mask = severities == sev_idx
        if mask.sum() > 0:
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=SEVERITY_COLORS[sev_idx], label=SEVERITY_NAMES[sev_idx],
                alpha=0.6, s=20, edgecolors="none",
            )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Conditioning Vector UMAP ({model_name})")
    ax.legend(title="OSA Severity")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_{model_name}.png", dpi=150)
    plt.savefig(f"{output_dir}/umap_{model_name}.pdf")
    plt.close()
    logger.info("UMAP 图已保存: %s/umap_%s.png", output_dir, model_name)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for model_name in ["Chambon2018", "TinySleepNet"]:
        logger.info("=== 提取 %s conditioning vectors ===", model_name)
        vectors, severities, pids = extract_conditioning_vectors(model_name=model_name)
        logger.info("  提取完成: %d vectors, %d unique patients", len(vectors), len(set(pids)))

        # t-SNE
        plot_tsne(vectors, severities, model_name, str(output_dir))

        # UMAP
        plot_umap(vectors, severities, model_name, str(output_dir))

        # 统计信息
        stats = {
            "n_vectors": len(vectors),
            "n_patients": len(set(pids)),
            "severity_distribution": {
                SEVERITY_NAMES[i]: int((severities == i).sum()) for i in range(4)
            },
            "vector_dim": vectors.shape[1] if len(vectors) > 0 else 0,
            "mean_norm": float(np.linalg.norm(vectors, axis=1).mean()) if len(vectors) > 0 else 0,
        }

        # 计算不同严重程度之间的 conditioning vector 距离
        centroids = {}
        for sev_idx in range(4):
            mask = severities == sev_idx
            if mask.sum() > 0:
                centroids[sev_idx] = vectors[mask].mean(axis=0)

        if len(centroids) >= 2:
            distances = {}
            keys = sorted(centroids.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    k1, k2 = keys[i], keys[j]
                    d = float(np.linalg.norm(centroids[k1] - centroids[k2]))
                    distances[f"{SEVERITY_NAMES[k1]}-{SEVERITY_NAMES[k2]}"] = d
            stats["centroid_distances"] = distances

        all_stats[model_name] = stats

    # 保存统计
    with open(output_dir / "conditioning_vector_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    logger.info("所有可视化完成，结果保存到 %s", output_dir)


if __name__ == "__main__":
    main()
